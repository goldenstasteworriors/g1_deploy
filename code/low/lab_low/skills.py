"""Closed skill vocabulary and explicit actuator contract (all prompts English)."""
import numpy as np

SKILLS = {
    "take_dropper_and_aspirate": (0, "left", "Use the left gripper to take the indicated dropper from the source bottle and aspirate reagent once. Keep holding the dropper."),
    "pick_up_test_tube": (1, None, "Use the {hand} gripper to pick up the indicated test tube and keep holding it."),
    "dispense_into_held_test_tube": (2, "both", "Use the dropper held by the left gripper to dispense into the test tube held by the right gripper. Do not aspirate again. Keep holding both objects."),
    "return_dropper": (3, "left", "Use the left gripper to return the held dropper directly to the indicated source bottle and release it."),
    "mix_test_tube": (4, None, "Mix the test tube held by the {hand} gripper using the configured mixing procedure. Keep holding the tube."),
    "return_test_tube_to_rack": (5, None, "Use the {hand} gripper to place the held test tube into the indicated empty rack slot and release it once supported."),
    "place_test_tube_in_water_bath": (6, None, "Use the {hand} gripper to place the held test tube at the indicated supported position in the water bath and release it once stable."),
}
HANDS = {"left": 0, "right": 1, "both": 2}
LEFT = [15, 16, 17, 18, 19, 20, 21, 29]
RIGHT = [22, 23, 24, 25, 26, 27, 28, 30]
ARM_INDICES = LEFT + RIGHT
ACTION_DIM = 16  # left 7 joints+Dex1; right 7 joints+Dex1; no base output
STATE_DIM = 31
CONTRACT_VERSION = "spacetime_manip_v3"
# Both objects are already identified by their holding hands for this skill.
OPTIONAL_TARGET_SKILLS = frozenset({"dispense_into_held_test_tube"})


def validate_targets(skill, targets):
    """Normalize omitted marks only for explicitly supported object-in-hand skills."""
    if skill not in SKILLS:
        raise ValueError(f"Unsupported skill {skill!r}")
    targets = [] if targets is None else targets
    if not isinstance(targets, list) or len(targets) > 2 or any(not isinstance(m, dict) for m in targets):
        raise ValueError("targets must be a list of at most two initial target objects")
    if not targets and skill not in OPTIONAL_TARGET_SKILLS:
        raise ValueError(f"{skill} requires an initial target box or mask")
    return targets


def condition(skill, hand):
    if skill not in SKILLS:
        raise ValueError(f"Unsupported skill {skill!r}; choose from {tuple(SKILLS)}")
    sid, fixed_hand, template = SKILLS[skill]
    if hand not in HANDS or (fixed_hand is not None and hand != fixed_hand):
        raise ValueError(f"Invalid hand={hand!r} for skill={skill!r} (requires {fixed_hand})")
    if fixed_hand is None and hand not in ("left", "right"):
        raise ValueError(f"{skill} requires a single explicit hand")
    return sid, HANDS[hand], template.format(hand=hand)


def action_mask(skill, hand):
    condition(skill, hand)
    mask = np.zeros(ACTION_DIM, dtype=bool)
    if hand in ("left", "both"):
        mask[:8] = True
    if hand in ("right", "both"):
        mask[8:] = True
    return mask


def gate_action(raw_action, measured_state, skill, hand):
    """Apply after denormalization. Inactive joints HOLD, never command zero angles.

    This does not implement balance, collision avoidance, water-bath release,
    skill completion or emergency stopping; those belong to the controller.
    """
    raw_action = np.asarray(raw_action, dtype=np.float32)
    measured_state = np.asarray(measured_state, dtype=np.float32)
    if raw_action.shape[-1] != ACTION_DIM or measured_state.shape != (STATE_DIM,):
        raise ValueError(f"Expected action[...,{ACTION_DIM}] and measured_state[{STATE_DIM}]")
    if not np.isfinite(raw_action).all() or not np.isfinite(measured_state).all():
        raise ValueError("Non-finite command/state")
    hold = measured_state[ARM_INDICES]
    return np.where(action_mask(skill, hand), raw_action, hold)
