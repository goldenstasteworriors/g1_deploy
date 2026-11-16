"""Bind initial pixel boxes to their original RGB images, without robot I/O."""
import base64
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from lab_low.skills import condition, validate_targets

CAMERAS = tuple("observation.images." + name for name in ("ego_view", "left_wrist", "right_wrist"))


def encode_rgb(images):
    result = {}
    for name in CAMERAS:
        rgb = images[name]
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected uint8 RGB: {name}")
        ok, data = cv2.imencode(".png", rgb[..., ::-1])
        if not ok:
            raise ValueError(f"Cannot encode {name}")
        result[name] = base64.b64encode(data).decode("ascii")
    return result


def validate_live_marks(marks, *, require_reference=True):
    """Return only reset-protocol fields; provenance never enters model prompts."""
    _, _, prompt = condition(marks["skill"], marks["hand"])
    if marks.get("prompt", prompt) != prompt:
        raise ValueError("Prompt differs from the trained skill/hand command")
    targets = validate_targets(marks["skill"], marks.get("targets"))
    result = dict(skill=marks["skill"], hand=marks["hand"], prompt=prompt, targets=targets)
    reference = marks.get("reference_images")
    if reference is None:
        if targets and require_reference:
            raise ValueError("Initial boxes need their original reference_images; capture and mark a live frame first")
        return result
    if not isinstance(reference, dict) or set(reference) != set(CAMERAS):
        raise ValueError("Reference must include ego_view, left_wrist and right_wrist")
    shapes = {}
    for name, encoded in reference.items():
        raw = base64.b64decode(encoded, validate=True)
        expected = marks.get("reference_sha256", {}).get(name)
        if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Reference image checksum mismatch: {name}")
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot decode original reference image: {name}")
        shapes[name] = bgr.shape[:2]
    roles = set()
    for mark in targets:
        if "mask" in mark:
            raise ValueError("Live deployment takes initial pixel boxes, not mask file paths")
        role = mark.get("role", 0)
        if role not in (0, 1) or role in roles:
            raise ValueError("Duplicate or invalid target role")
        roles.add(role)
        h, w = shapes[mark.get("camera", CAMERAS[0])]
        box = np.asarray(mark["box"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError(f"Box must be xyxy pixels in its original {w}x{h} image")
        confidence = float(mark.get("confidence", .5))
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid target confidence")
    result["reference_images"] = reference
    return result


def load_live_marks(path):
    return validate_live_marks(json.loads(Path(path).read_text()))


def save_live_marks(path, marks):
    validate_live_marks(marks)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(marks, ensure_ascii=False) + "\n")
    temporary.replace(path)
