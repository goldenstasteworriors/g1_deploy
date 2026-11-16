"""Deployment-only Dex1 motor-angle calibration after action denormalization."""
import numpy as np


PICK_TUBE_GRIPPER_CALIBRATION = dict(
    skill="pick_up_test_tube", interval_rad=[0.8, 1.1], target_cap_rad=0.85,
    rule="min(predicted, 0.85) inside inclusive interval; otherwise unchanged",
    scope="active gripper only; initialization bypassed; before native Dex1 driver")


def calibrate_gripper(command, *, skill, hand, initialization=False):
    """Return a copy; do not change measured feedback or inactive channels."""
    calibrated = np.asarray(command, dtype=float).copy()
    if skill == "pick_up_test_tube" and not initialization:
        index = {"left": 7, "right": 15}[hand]
        value = calibrated[index]
        if 0.8 <= value <= 1.1:
            calibrated[index] = min(value, 0.85)
    return calibrated
