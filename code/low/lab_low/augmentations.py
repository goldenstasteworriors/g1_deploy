"""Train-time nuisance changes that preserve the achieved joint targets.

Camera transforms represent small camera perturbations, not relocated objects.
The same sampled trajectory is evaluated at each frame's original timestamp;
reference priors must be transformed with the reference image at time zero.
"""
import cv2
import numpy as np
from PIL import Image


def sample_local_light(config, rng):
    if not config or not config.get("enabled", False) or rng.random() >= float(config["probability"]):
        return None
    count = int(config["regions"])
    return [dict(center=rng.uniform(-.1, 1.1, (count, 2)),
                 radius=rng.uniform(*config["radius_fraction"], (count, 2)),
                 gain=rng.uniform(*config["gain"], count)) for _ in range(3)]


def local_light_field(size, parameters):
    width, height = size
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x /= max(1, width - 1)
    y /= max(1, height - 1)
    field = np.ones((height, width), dtype=np.float32)
    for (cx, cy), (rx, ry), gain in zip(parameters["center"], parameters["radius"], parameters["gain"]):
        field += gain * np.exp(-.5 * (((x-cx)/rx)**2 + ((y-cy)/ry)**2))
    # Two overlapping bright/shadow regions must not erase transparent edges.
    return field.clip(.65, 1.35)


def perturb_box(box, image_size, rng, config):
    """Bounded label noise, retaining the original center and most box area.

This is a geometric guard, not proof that neighboring objects are excluded.
Only the initial identity prior is changed; the trajectory label stays fixed.
"""
    original = np.asarray(box, dtype=np.float32)
    if rng.random() >= float(config["probability"]):
        return original.copy()
    width, height = image_size
    wh = original[2:] - original[:2]
    center = (original[:2] + original[2:]) / 2
    area = float(np.prod(wh))
    for _ in range(8):
        candidate = original.copy()
        mode = rng.choice(("shift", "expand", "trim"), p=[.4, .3, .3])
        if mode == "shift":
            delta = rng.uniform(-float(config["shift_fraction"]), float(config["shift_fraction"]), 2) * wh
            candidate += np.tile(delta, 2)
        elif mode == "expand":
            extent = wh * rng.uniform(*config["expansion"], 2) / 2
            candidate = np.concatenate((center - extent, center + extent))
        else:
            edge = int(rng.integers(4))
            amount = rng.uniform(*config["trim_fraction"]) * wh[edge % 2]
            candidate[edge] += amount if edge < 2 else -amount
        candidate[[0, 2]] = candidate[[0, 2]].clip(0, width)
        candidate[[1, 3]] = candidate[[1, 3]].clip(0, height)
        intersection = np.maximum(0, np.minimum(candidate[2:], original[2:]) - np.maximum(candidate[:2], original[:2]))
        retained = float(np.prod(intersection))
        candidate_area = float(np.prod(np.maximum(0, candidate[2:] - candidate[:2])))
        iou = retained / max(area + candidate_area - retained, 1e-8)
        contains_center = np.all(candidate[:2] <= center) and np.all(center <= candidate[2:])
        if contains_center and retained / area >= float(config["minimum_retained_area"]) and iou >= float(config["minimum_iou"]):
            return candidate.astype(np.float32)
    return original.copy()


def sample_camera(config, rng):
    if not config or not config.get("enabled", False) or rng.random() >= float(config["probability"]):
        return None
    return dict(
        offset=rng.uniform(-float(config["translation_fraction"]), float(config["translation_fraction"]), (3, 2)),
        angle=rng.uniform(-float(config["rotation_degrees"]), float(config["rotation_degrees"]), 3),
        amplitude=rng.uniform(0, float(config["temporal_translation_fraction"]), (3, 2)),
        angle_amplitude=rng.uniform(0, float(config["temporal_rotation_degrees"]), 3),
        frequency=rng.uniform(*config["frequency_hz"], (3, 3)),
        phase=rng.uniform(-np.pi, np.pi, (3, 3)),
    )


def camera_matrix(size, parameters, camera, seconds):
    width, height = size
    wave = np.sin(2*np.pi*parameters["frequency"][camera]*seconds + parameters["phase"][camera])
    shift = parameters["offset"][camera] + parameters["amplitude"][camera]*wave[:2]
    angle = parameters["angle"][camera] + parameters["angle_amplitude"][camera]*wave[2]
    matrix = cv2.getRotationMatrix2D(((width-1)/2, (height-1)/2), float(angle), 1.)
    matrix[:, 2] += shift * np.array([width, height])
    return matrix


def apply_camera(image, parameters, camera, seconds, is_mask=False):
    if parameters is None:
        return image
    result = cv2.warpAffine(np.asarray(image), camera_matrix(image.size, parameters, camera, seconds),
                            image.size, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT_101,
                            borderValue=0)
    return Image.fromarray(result)
