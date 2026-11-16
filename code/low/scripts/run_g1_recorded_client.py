"""Live G1 state + recorded RGB/initial boxes/skill/prompt; no recorded actions.

Dataset RGB advances on model-action steps; 30Hz state/history include waiting holds.
The operator remains responsible for standing, initialization and d/r.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import threading
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_low.recorded_vision import RecordedDataset, RecordedEpisode
from lab_low.recorded_control import LiveCommandPump, ROSStateReader, ReadOnlyDDSStateReader
from run_g1_lab_client import rpc


def run_session(source, pump, call, *, binding, prefix=8, chunks=None, report=print):
    """Dependency-injected transport for offline checks of the actual playback loop."""
    index, completed = 0, 0
    reset = call(source.reset_packet(pump.snapshot(), binding))
    if not reset.get("reference_images_used") or reset.get("prompt") != source.task["prompt"]:
        raise RuntimeError("Policy service needs the recorded-reference protocol; restart it with current code")
    pump.enable()
    try:
        while source.remaining and (chunks is None or completed < chunks):
            # The pump holds the last commanded pose while this request is blocked.
            # Neither source.frame nor the execution-indexed history advances.
            state_before = pump.snapshot()
            result = call(dict(op="predict", binding_id=binding))
            actions = np.asarray(result["actions"], dtype=float)
            if result.get("binding_id") != binding or actions.shape != (32, 16) or not np.isfinite(actions).all():
                raise ValueError("Invalid predicted action chunk")
            count = min(prefix, source.remaining)
            prediction_id = str(uuid.uuid4())
            report(json.dumps(dict(event="model_prediction", wall_time_ns=time.time_ns(),
                monotonic_ns=time.monotonic_ns(), binding_id=binding, prediction_id=prediction_id,
                dataset_frame=source.frame, execute_prefix=count, actions=actions.tolist())))
            if hasattr(pump, "set_prediction"):
                pump.set_prediction(dict(prediction_id=prediction_id, dataset_frame=source.frame))
            states = pump.execute(actions[:count])
            observations = []
            for state in states:
                source.advance()
                index += 1
                observations.append(source.packet(state, index))
            call(dict(op="observe_batch", binding_id=binding, observations=observations))
            completed += 1
            report(json.dumps(dict(episode=source.selection["episode"], dataset_frame=source.frame,
                execution_index=index, chunks=completed, steps=count,
                inference_seconds=result.get("inference_seconds"), state_source="live_robot",
                vision_source="recorded_dataset", live_state_before=state_before.tolist(),
                live_state_after=states[-1].tolist(), predicted_first=actions[0].tolist(),
                predicted_last=actions[count-1].tolist(), completion=None)))
        reason = "recorded_episode_ended" if not source.remaining else "requested_chunk_count_reached"
        result = dict(reason=reason, dataset_frame=source.frame, executed_steps=index,
                      chunks=completed, completion=None)
        report(json.dumps(result))
        return result
    finally:
        # Stop the pump before a possibly slow model request; d is in the
        # independent operator process and never waits for either operation.
        pump.close()
        try:
            call(dict(op="stop", binding_id=binding))
        except Exception:
            pass


def run_session_async(source, pump, call, *, binding, horizon=32, delay_steps=16,
                      chunks=None, report=print, on_tick=None, encode_images=None,
                      simulate_latency=False, expected_denoise_steps=None, smoothing_steps=0,
                      freeze_simulation_on_inference=False):
    """Depth-one prefetch; official execution counter plus a 30Hz observation clock.

    The first model request holds the initial pose. Later underrun holds also
    advance state/history time, while recorded RGB and action count pause.
    Only the inference thread encodes RGB or does RPC.
    """
    from lab_low.async_execution import AsyncChunkExecutor
    from lab_low.skills import action_mask
    lock = threading.Lock()
    pending = []
    index = 0
    execution_step = 0
    started = False
    last_trace = dict(inference_mode="async",holding=True)

    def emit(record):
        with lock:
            report(json.dumps(record))

    def snapshot(step):
        records = list(pending)
        pending.clear()
        return dict(records=records,step=step,dataset_frame=source.frame)

    def generate(condition):
        began = time.monotonic()
        observations = [dict(frame_index=i,state=state.tolist(),
            images=(encode_images(images) if encode_images else source.encode(images,'.jpg')),
            dataset_frame=dataset_frame)
            for i,state,images,dataset_frame in condition['records']]
        if observations:
            call(dict(op="observe_batch",binding_id=binding,observations=observations))
        request = dict(op="predict",binding_id=binding)
        if expected_denoise_steps is not None:
            request['denoise_steps']=expected_denoise_steps
        result = call(request)
        if result.get("binding_id") != binding:
            raise RuntimeError("Discard prediction for another session")
        if expected_denoise_steps is not None and result.get('denoise_steps') != expected_denoise_steps:
            raise RuntimeError("Policy service denoise settings differ; restart the matching service")
        result = dict(result, prediction_id=str(uuid.uuid4()),
                      pipeline_seconds=result.get('pipeline_seconds',time.monotonic()-began))
        emit(dict(event="model_prediction",binding_id=binding,prediction_id=result['prediction_id'],
            input_step=condition['step'],dataset_frame=condition['dataset_frame'],
            denoise_steps=result.get('denoise_steps'),actions=result['actions'],
            inference_seconds=result['inference_seconds'],pipeline_seconds=result['pipeline_seconds'],
            inference_mode="async",wall_time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns()))
        return result

    executor = AsyncChunkExecutor(generate,horizon=horizon,delay_steps=delay_steps,
                                  simulate_latency=simulate_latency,smoothing_steps=smoothing_steps,
                                  smoothing_mask=action_mask(source.task["skill"],source.task["hand"]),
                                  initial_target=pump.target_snapshot(),
                                  freeze_simulation_on_inference=freeze_simulation_on_inference)
    try:
        reset = call(source.reset_packet(pump.snapshot(),binding))
        if not reset.get('reference_images_used') or reset.get('prompt') != source.task['prompt']:
            raise RuntimeError("Recorded reference protocol mismatch")
        pump.enable()
        def sequence():
            nonlocal index, execution_step, started, last_trace
            while source.remaining:
                if chunks is not None and len(executor.adoptions)>=chunks and not executor.buffer:
                    break
                item = executor.poll(execution_step,snapshot,clock_step=index)
                if item is None:
                    action = pump.target_snapshot()
                    trace = dict(last_trace,holding=True)
                else:
                    action,trace = item
                    last_trace = trace
                    started = True
                state = yield action,trace
                if started:
                    if item is not None:
                        source.advance()
                        execution_step += 1
                    index += 1
                    pending.append((index,state.copy(),source.current_images.copy(),source.frame))
                if on_tick is not None:
                    on_tick(dict(index=index,execution_step=execution_step,holding=item is None,state=state.copy(),
                                 action=action.copy(),trace=trace,stats=executor.stats()))
        stream = sequence()
        if hasattr(pump, "execute_stream"):
            pump.execute_stream(stream)
        else:
            # Physics/fake pumps advance their own clock one period per step.
            try:
                item = next(stream)
                while True:
                    action,trace = item
                    state = pump.execute(np.asarray([action]),traces=[trace])[0]
                    item = stream.send(state)
            except StopIteration:
                pass
        result = dict(event="async_session_complete",inference_mode="async",dataset_frame=source.frame,
                      timeline_steps=index,execution_steps=execution_step,completion=None,**executor.stats())
        emit(result)
        return result
    finally:
        # Neither a late RPC nor its result can publish after this point.
        pump.close()
        executor.close()
        try:
            call(dict(op="stop",binding_id=binding))
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--policy-url", default="http://127.0.0.1:8849")
    parser.add_argument("--state-source", choices=["ros", "dds"], default="ros")
    parser.add_argument("--interface", default="enp132s0")
    parser.add_argument("--operator-epoch", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-input", action="store_true", help="Read live state and dataset inputs; no model RPC or actuator publication")
    parser.add_argument("--prefix", type=int, default=8)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--inference-mode",choices=["sync","async"],default="sync")
    parser.add_argument("--inference-horizon",type=int,default=32)
    parser.add_argument("--inference-delay-steps",type=int,default=16)
    parser.add_argument("--denoise-steps",type=int,default=10)
    parser.add_argument("--chunk-smoothing-steps",type=int,default=0)
    args = parser.parse_args()
    if not 1 <= args.prefix <= 32 or (args.chunks is not None and args.chunks < 1):
        parser.error("prefix must be 1..32 and chunks must be positive")
    if args.execute and (args.state_source != "ros" or args.operator_epoch is None or args.check_input):
        parser.error("Execution requires the staged operator's ROS state and epoch; input check is read-only")
    sys.path.insert(0, str(args.sonic_root.resolve()))
    sys.path.insert(0, str(args.sonic_root.resolve() / "external_dependencies/unitree_sdk2_python"))
    source = RecordedEpisode(RecordedDataset(args.dataset), args.episode, args.start_frame)
    reader = pump = None
    try:
        reader = ROSStateReader() if args.state_source == "ros" else ReadOnlyDDSStateReader(args.interface)
        publish = None
        if args.execute:
            from decoupled_wbc.control.main.constants import CONTROL_GOAL_TOPIC
            from decoupled_wbc.control.utils.ros_utils import ROSMsgPublisher
            publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
            publish = publisher.publish
        binding = str(uuid.uuid4())
        pump = LiveCommandPump(reader.read, publish, skill=source.task["skill"], hand=source.task["hand"],
                               binding=binding, epoch=args.operator_epoch)
        pump.start()
        state = pump.snapshot()
        print(json.dumps(dict(episode=args.episode, start_frame=args.start_frame,
            reference_frame=source.task["reference_frame"], skill=source.task["skill"],
            hand=source.task["hand"], prompt=source.task["prompt"], targets=source.task["targets"],
            image_shapes={k:list(v.shape) for k,v in source.current_images.items()},
            state=state.tolist(), state_source="live_robot", publishing=args.execute,
            playback="model_action_steps_only", history="live_state_and_recorded_RGB_after_each_step",
            annotation_sha256=source.dataset.annotation_digest)), flush=True)
        if args.check_input:
            return
        def call(packet):
            if packet.get('op')=='predict' and args.denoise_steps is not None:
                packet=dict(packet,denoise_steps=args.denoise_steps)
            return rpc(args.policy_url,packet)
        if args.inference_mode == "async":
            run_session_async(source,pump,call,binding=binding,
                horizon=args.inference_horizon,delay_steps=args.inference_delay_steps,chunks=args.chunks,
                expected_denoise_steps=args.denoise_steps,smoothing_steps=args.chunk_smoothing_steps,report=lambda s:print(s,flush=True))
        else:
            run_session(source, pump, call, binding=binding,
                        prefix=args.prefix, chunks=args.chunks, report=lambda s: print(s, flush=True))
    finally:
        if pump is not None and pump.thread.is_alive():
            pump.close()
        if reader is not None:
            reader.close()
        source.close()


if __name__ == "__main__":
    main()
