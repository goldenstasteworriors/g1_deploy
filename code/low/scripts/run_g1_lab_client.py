"""30 Hz sensor/command bridge on the SONICMJ controller host.

Runs stop-and-go inference: holds the last commanded pose during prediction, executes a
short prefix, then observes again. Lower-body control runs independently.
Default observes and predicts only; --execute enables arm goal publication.
"""
import argparse
import base64
from collections import deque
import json
import itertools
from pathlib import Path
import sys
import threading
import time
import urllib.request
import urllib.error
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_low.skills import ARM_INDICES, action_mask, condition
from lab_low.live_marks import load_live_marks


def rpc(url, packet):
    request = urllib.request.Request(url.rstrip("/") + "/rpc",
        json.dumps(packet, allow_nan=False).encode(), {"Content-Type": "application/json"})
    # Private wired traffic must not go through a shell HTTP proxy.
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        # Preserve the worker's reason instead of hiding it behind "HTTP 400".
        with error:
            payload = error.read().decode(errors='replace')
        raise RuntimeError(f"Policy HTTP {error.code}: {payload}") from error
    if not result.get("ok"):
        raise RuntimeError(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sonic-root", type=Path, required=True)
    p.add_argument("--policy-url", default="http://192.168.123.222:8849")
    p.add_argument("--camera-host", default="127.0.0.1")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--marks", type=Path, required=True, help="JSON with skill, hand and initial targets")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--prefix", type=int, default=8)
    p.add_argument("--chunks", type=int, default=None)
    p.add_argument("--denoise-steps",type=int,default=10)
    p.add_argument("--operator-epoch", type=int, help="Reject commands after an operator stop/restart")
    args = p.parse_args()
    if not 1 <= args.prefix <= 32 or (args.chunks is not None and args.chunks < 1):
        p.error("prefix must be 1..32 and chunks must be positive")
    marks = load_live_marks(args.marks)
    condition(marks["skill"], marks["hand"])
    active = action_mask(marks["skill"], marks["hand"])
    sys.path.insert(0, str(args.sonic_root.resolve()))
    from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber
    from decoupled_wbc.control.main.constants import STATE_TOPIC_NAME, CONTROL_GOAL_TOPIC
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.camera.sensor_server import SensorClient, ImageMessageSchema
    from gear_sonic.camera.synchronization import ImageBundleBuffer
    robot = instantiate_g1_robot_model(waist_location="lower_body")
    body = robot.get_body_actuated_joint_indices()
    ros_manager = ROSManager(node_name="labwam_sensor_bridge")
    subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC) if args.execute else None
    shared_lock = threading.Lock()
    stop = threading.Event()
    packets, errors = deque(), []
    binding = str(uuid.uuid4())
    selected = {"target": None, "allow": False, "trace": None}
    mounts = ("ego_view", "left_wrist", "right_wrist")

    def sample_loop():
        camera = SensorClient()
        camera.start_client(args.camera_host, args.camera_port)
        views = ImageBundleBuffer(mounts)
        frame = sequence = 0
        armed = False
        state_msg = None
        deadline = time.monotonic()
        try:
            while not stop.is_set():
                now = time.monotonic()
                state = subscriber.get_msg()
                if state is not None:
                    state_msg = state
                raw = camera.receive_message_nonblocking(0)
                if raw is not None:
                    views.update(ImageMessageSchema.deserialize(raw).asdict(), now)
                images = views.current(now)
                if state_msg is None or images is None:
                    stop.wait(.01)
                    deadline = time.monotonic()
                    continue
                q = np.asarray(state_msg["q"])
                measured = np.r_[q[body], state_msg["left_hand_dex1_q"], state_msg["right_hand_dex1_q"]]
                if measured.shape != (31,) or not np.isfinite(measured).all():
                    raise ValueError("Invalid 31D measured state")
                encoded = {}
                for mount in mounts:
                    # SensorClient returns RGB arrays; OpenCV JPEG encoder expects BGR.
                    ok, jpg = cv2.imencode(".jpg", images["images"][mount][..., ::-1],
                                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok:
                        raise RuntimeError("JPEG encode failed")
                    encoded["observation.images." + mount] = base64.b64encode(jpg).decode()
                packet = {"frame_index": frame, "state": measured.tolist(), "images": encoded}
                with shared_lock:
                    packets.append(packet)
                    target = selected["target"]
                    allow = selected["allow"]
                    trace = selected["trace"]
                    if target is None:
                        held = state_msg.get("lab_hold_target")
                        target = np.asarray(measured[ARM_INDICES] if held is None else held, dtype=float).copy()
                        if target.shape != (16,) or not np.isfinite(target).all():
                            raise ValueError("Invalid initial hold target")
                        selected["target"] = target
                if publisher is not None and allow:
                    if state_msg.get("control_mode") != "running":
                        raise RuntimeError("Controller is not running; use its existing manual mode controls")
                    if args.operator_epoch is not None and state_msg.get("lab_operator_epoch") != args.operator_epoch:
                        raise RuntimeError("Operator epoch changed; discard this inference session")
                    if state_msg.get("lab_binding_id") == binding:
                        armed = True
                    elif armed:
                        raise RuntimeError("Controller dropped the binding; restart the skill explicitly")
                    publisher.publish({"lab_target": target.tolist(), "lab_binding_id": binding,
                        "lab_sequence": sequence, "lab_sent_monotonic": time.monotonic(),
                        "lab_operator_epoch": args.operator_epoch,
                        "lab_trace": trace,
                        "lab_skill": marks["skill"], "lab_hand": marks["hand"], "lab_reset": not armed})
                    sequence += 1
                frame += 1
                deadline = max(deadline + 1 / 30, time.monotonic())
                stop.wait(max(0, deadline - time.monotonic()))
        except Exception as error:
            errors.append(str(error))
            stop.set()
        finally:
            camera.stop_client()

    thread = threading.Thread(target=sample_loop, daemon=True)
    thread.start()
    sent_frame = None
    try:
        while not packets and not errors:
            time.sleep(.02)
        if errors:
            raise RuntimeError(errors[0])
        with shared_lock:
            first = dict(packets[-1])
        reset = rpc(args.policy_url, {**first, **marks, "op": "reset", "binding_id": binding})
        if marks.get("reference_images") and not reset.get("reference_images_used"):
            raise RuntimeError("Policy service did not accept the original marked reference images")
        print(json.dumps(dict(event="live_target_bound", binding_id=binding, marks_path=str(args.marks),
                              skill=marks["skill"], hand=marks["hand"], targets=marks["targets"],
                              reference_images_used=reset.get("reference_images_used", False),
                              observation_frame=first["frame_index"]), ensure_ascii=False), flush=True)
        sent_frame = first["frame_index"]
        with shared_lock:
            selected["allow"] = True
        iterations = itertools.count() if args.chunks is None else range(args.chunks)
        for _ in iterations:
            # Keep publishing the last target while inference is running.
            with shared_lock:
                recent = [dict(x) for x in packets if x["frame_index"] > sent_frame]
                # Prune only observations already acknowledged by the model.
                # Keep the newest measurement even when no frames are pending.
                while len(packets) > 1 and packets[0]["frame_index"] <= sent_frame:
                    packets.popleft()
            if recent:
                rpc(args.policy_url, {"op": "observe_batch", "binding_id": binding, "observations": recent})
                sent_frame = recent[-1]["frame_index"]
            result = rpc(args.policy_url, {"op": "predict", "binding_id": binding,"denoise_steps":args.denoise_steps})
            if errors:
                raise RuntimeError(errors[0])
            actions = np.asarray(result["actions"], dtype=float)
            if result.get("binding_id") != binding or actions.shape != (32, 16) or not np.isfinite(actions).all():
                raise RuntimeError("Invalid prediction")
            prediction_id = str(uuid.uuid4())
            print(json.dumps(dict(event="model_prediction", wall_time_ns=time.time_ns(),
                monotonic_ns=time.monotonic_ns(), binding_id=binding, prediction_id=prediction_id,
                observation_frame=sent_frame, execute_prefix=args.prefix, actions=actions.tolist())), flush=True)
            with shared_lock:
                measured_now = np.asarray(packets[-1]["state"])[ARM_INDICES].copy()
                held = selected["target"].copy()
            print(json.dumps({"inference_seconds": result["inference_seconds"],
                              "observation_frame": sent_frame, "publishing": args.execute,
                              "measured_arm_state": measured_now.tolist(), "hold_target": held.tolist(),
                              "first_target": np.where(active, actions[0], held).tolist(),
                              "last_target": np.where(active, actions[args.prefix-1], held).tolist()}), flush=True)
            for index, action in enumerate(actions[:args.prefix]):
                with shared_lock:
                    selected["target"] = np.where(active, action, selected["target"])
                    selected["trace"] = dict(prediction_id=prediction_id, action_index=index,
                                             model_target16=action.tolist())
                if stop.wait(1 / 30):
                    raise RuntimeError(errors[0] if errors else "Stopped")
        if errors:
            raise RuntimeError(errors[0])
    finally:
        with shared_lock:
            selected["allow"] = False
        stop.set()
        thread.join(timeout=2)
        if publisher is not None:
            publisher.publish({"lab_stop": True, "lab_binding_id": binding, "lab_operator_epoch": args.operator_epoch})
        if sent_frame is not None:
            try:
                rpc(args.policy_url, {"op": "stop", "binding_id": binding})
            except Exception:
                pass
        # Finish the ROS executor before interpreter teardown; otherwise its
        # background C++ thread can abort an otherwise successful replay.
        ROSManager.shutdown()
        if hasattr(ros_manager, "thread"):
            ros_manager.thread.join(timeout=2)


if __name__ == "__main__":
    main()
