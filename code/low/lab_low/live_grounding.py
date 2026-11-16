"""Read-only camera capture and the existing Qwen grounding HTTP contract."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np

from lab_low.live_marks import CAMERAS, encode_rgb, save_live_marks, validate_live_marks
from lab_low.skills import condition

OPERATIONS = {
    "TAKE_TUBE": "pick_up_test_tube",
    "TAKE_DROPPER_AND_ASPIRATE": "take_dropper_and_aspirate",
    "DISPENSE_TO_HELD_TUBE": "dispense_into_held_test_tube",
    "RETURN_DROPPER": "return_dropper",
    "RETURN_TUBE_TO_RACK": "return_test_tube_to_rack",
}


def request(url, path, body=None):
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data, {"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Grounding HTTP {error.code}: {error.read().decode()}") from error


def capture(config):
    sys.path.insert(0, str(Path(config["sonic_root"]).resolve()))
    from gear_sonic.camera.sensor_server import SensorClient, ImageMessageSchema
    from gear_sonic.camera.synchronization import ImageBundleBuffer
    client = SensorClient()
    client.start_client(config["camera_host"], config["camera_port"])
    views = ImageBundleBuffer(tuple(c.removeprefix("observation.images.") for c in CAMERAS))
    deadline = time.monotonic() + config.get("capture_timeout", 15.)
    try:
        while time.monotonic() < deadline:
            raw = client.receive_message_nonblocking(50)
            if raw is not None:
                views.update(ImageMessageSchema.deserialize(raw).asdict(), time.monotonic())
            bundle = views.current(time.monotonic())
            if bundle is not None:
                images = {c: bundle["images"][c.removeprefix("observation.images.")].copy() for c in CAMERAS}
                return images, dict(received_wall_time_ns=time.time_ns(), camera_host=config["camera_host"],
                                    camera_port=config["camera_port"])
        raise RuntimeError("No complete live ego/left_wrist/right_wrist bundle; check the existing camera service")
    finally:
        client.stop_client()


def proposal_marks(config, encoded, response, frame_id):
    """Reject stale images, mismatched operations, and non-grounding decisions."""
    step, prediction, observation = response["next_operation"], response["prediction"], response["observation"]
    skill, hand = config["skill"], config["hand"]
    if response["request_id"] != frame_id or prediction["request_id"] != frame_id:
        raise ValueError("Grounding request ID mismatch")
    if OPERATIONS.get(step["operation"]) != skill or step["hand"] != hand:
        raise ValueError("Grounding session step does not match the selected model skill/hand")
    if prediction["operation"] != step["operation"] or prediction["target_id"] != step["target"]["id"]:
        raise ValueError("Grounding target/operation mismatch")
    sha = hashlib.sha256(base64.b64decode(encoded[CAMERAS[0]], validate=True)).hexdigest()
    if observation["frame_id"] != frame_id or observation["input_sha256"] != sha:
        raise ValueError("Grounding proposal belongs to a different original image")
    if prediction["decision"] != "ground":
        raise ValueError(f"Qwen returned {prediction['decision']}; no executable target was produced")
    marks = dict(skill=skill, hand=hand, prompt=condition(skill, hand)[2],
                 targets=[dict(role=0, camera=CAMERAS[0], box=prediction["bbox_xyxy"], confidence=.5)],
                 reference_images=encoded)
    validate_live_marks(marks)
    return marks


def prepare(config, output, *, images=None, capture_metadata=None, call=request):
    """Capture once, ground that exact image, and atomically publish its binding.

    Explicit images/call injection supports offline checks. The deployment CLI
    always captures real cameras; it cannot fall back to dataset imagery.
    """
    skill, hand = config["skill"], config["hand"]
    condition(skill, hand)
    if images is None:
        images, capture_metadata = capture(config)
    encoded = encode_rgb(images)
    capture_id = "live_" + uuid.uuid4().hex
    archive = Path(output).parent / "captures" / capture_id
    archive.mkdir(parents=True, exist_ok=False)
    for name, data in encoded.items():
        (archive / (name.removeprefix("observation.images.") + ".png")).write_bytes(base64.b64decode(data))
    (archive / "capture.json").write_text(json.dumps(dict(capture_id=capture_id, **(capture_metadata or {})), indent=2))
    provenance = dict(capture_id=capture_id, archive=str(archive.resolve()), **(capture_metadata or {}))
    if skill == "dispense_into_held_test_tube":
        marks = dict(skill=skill, hand=hand, prompt=condition(skill, hand)[2], targets=[], reference_images=encoded)
    else:
        if images[CAMERAS[0]].shape != (480, 640, 3):
            raise ValueError("Current Qwen service expects original ego RGB at 640x480; no implicit resize")
        url = config["grounding_url"]
        health = call(url, "/health")
        if health.get("service") != "lab_grounding_resident_v1" or not health.get("model", {}).get("ready"):
            raise ValueError("The existing Qwen grounding service is not ready")
        sid = config.get("session_id")
        if sid:
            state = call(url, "/sessions/" + sid)
        else:
            packet = config["plan_packet"]
            sid = capture_id
            state = call(url, "/sessions", dict(session_id=sid, plan=packet["plan"], layout=packet["layout"]))
        step = state.get("next_operation")
        if step is None or OPERATIONS.get(step["operation"]) != skill or step["hand"] != hand:
            raise ValueError("The current high-level step does not match the selected skill/hand")
        response = call(url, "/sessions/" + sid + "/ground", dict(request_id=capture_id,
                        frame_id=capture_id, expected_revision=state["revision"], image_base64=encoded[CAMERAS[0]]))
        if response["session_id"] != sid or response["state_revision_before"] != state["revision"]:
            raise ValueError("Grounding session/revision mismatch")
        (archive / "grounding_response.json").write_text(json.dumps(response, indent=2, ensure_ascii=False))
        marks = proposal_marks(config, encoded, response, capture_id)
        provenance.update(session_id=sid, request_id=capture_id, revision=response["revision"],
                          identity_verified=False, target_id=step["target"]["id"])
    marks.update(reference_metadata=provenance,
                 reference_sha256={c: hashlib.sha256(base64.b64decode(v)).hexdigest() for c, v in encoded.items()})
    preview = images[CAMERAS[0]][..., ::-1].copy()
    for target in marks["targets"]:
        x1, y1, x2, y2 = map(int, target["box"])
        cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
    if not cv2.imwrite(str(archive / "preview.png"), preview):
        raise RuntimeError("Cannot save target preview")
    save_live_marks(output, marks)
    return dict(skill=skill, hand=hand, targets=marks["targets"], marks=str(output),
                preview=str(archive / "preview.png"), reference_metadata=provenance,
                robot_commands_sent=False, completion_receipt_sent=False)
