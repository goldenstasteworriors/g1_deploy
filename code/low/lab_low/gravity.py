"""Measured-state G1 arm gravity feedforward; no actuator or ROS/DDS I/O.

Replace the bare G1 URDF's rubber hands with nominal Dex1-1 wrist payloads.
An optional per-side total payload override includes the gripper, adapters and
objects, in the corresponding wrist-yaw joint frame. No Dex3 placeholder mass.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


# Unitree xr_teleoperate G1_29_ArmController: shoulder/elbow and wrist gains.
XR_KP = np.tile([80., 80., 80., 80., 40., 40., 40.], 2)
XR_KD = np.tile([3., 3., 3., 3., 1.5, 1.5, 1.5], 2)


def apply_xr_arm_gains(sender, *, enabled):
    """Only the 14 arm motors; lower-body gains belong to SONICMJ."""
    motors = np.asarray(sender.config["JOINT2MOTOR"])[15:29]
    sender.robot_kp[motors] = XR_KP if enabled else 0.
    sender.robot_kd[motors] = XR_KD if enabled else 0.


class ArmGravityCompensator:
    def __init__(self, robot, *, scale=1., payload=None):
        import pinocchio as pin
        import decoupled_wbc
        if not np.isfinite(scale):
            raise ValueError("Gravity scale must be finite")
        urdf = Path(decoupled_wbc.__file__).parent / "control/robot_model/model_data/g1/g1_29dof.urdf"
        tree = ET.parse(urdf).getroot()
        self.removed_rubber_hands = {}
        for side in ("left", "right"):
            link = tree.find(f"./link[@name='{side}_rubber_hand']")
            if link is None or link.find("inertial") is None:
                raise ValueError("Expected the SONICMJ G1 bare model with fixed rubber hands")
            inertial = link.find("inertial")
            self.removed_rubber_hands[side] = float(inertial.find("mass").get("value"))
            link.remove(inertial)
        self.pin, self.model = pin, pin.buildModelFromXML(ET.tostring(tree, encoding="unicode"))
        self.body = np.asarray(robot.get_body_actuated_joint_indices(), dtype=int)
        self.input_dofs = robot.num_dofs
        if len(self.body) != 29 or self.model.nq != 29 or self.model.nv != 29:
            raise ValueError("Gravity feedforward requires fixed-base G1 29-DOF model")
        names = [robot.joint_names[i] for i in self.body]
        if names[15:22] != ["left_"+s+"_joint" for s in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]:
            raise ValueError("Unexpected G1 arm actuator order")
        if names[22:29] != [n.replace("left_", "right_") for n in names[15:22]]:
            raise ValueError("Unexpected right arm actuator order")
        joints = [self.model.joints[self.model.getJointId(n)] for n in names]
        if any(j.nq != 1 or j.nv != 1 for j in joints):
            raise ValueError("Expected single-axis named G1 joints")
        self.q_indices = np.array([j.idx_q for j in joints])
        self.v_indices = np.array([j.idx_v for j in joints])
        self.dex1_reference = json.loads((Path(__file__).resolve().parents[1]/"configs/dex1_1_payload.json").read_text())
        self.payload = {} if payload is None else json.loads(Path(payload).read_text())
        if not isinstance(self.payload, dict) or set(self.payload)-{"left", "right"}:
            raise ValueError("Payload must map left/right to mass_kg and com_m")
        # Overrides are TOTAL external payloads, not additions to the Dex1.
        self.tool_payload = dict(self.dex1_reference["payload"], **self.payload)
        for side, item in self.tool_payload.items():
            if not isinstance(item, dict) or set(item) != {"mass_kg", "com_m"}:
                raise ValueError("Payload requires mass_kg and com_m in wrist-yaw joint coordinates")
            mass, com = float(item["mass_kg"]), np.asarray(item["com_m"], dtype=float)
            if com.shape != (3,) or not np.isfinite(com).all() or not np.isfinite(mass) or mass < 0:
                raise ValueError("Invalid tool payload mass/COM")
            jid = self.model.getJointId(f"{side}_wrist_yaw_joint")
            self.model.inertias[jid] += pin.Inertia(mass, com, np.zeros((3, 3)))
        self.data = self.model.createData()
        self.scale = float(scale)
        self.reset()

    def reset(self):
        self.last_raw = np.zeros(14)

    def raw(self, observation):
        q_obs = np.asarray(observation["q"], dtype=float)
        base = np.asarray(observation["floating_base_pose"], dtype=float)
        if q_obs.shape != (self.input_dofs,) or not np.isfinite(q_obs).all() or base.shape != (7,) or not np.isfinite(base).all():
            raise ValueError("Gravity compensation requires finite measured joints and base pose")
        quat = base[3:7]  # SONICMJ / Unitree state is w,x,y,z (pelvis in world).
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Invalid measured base quaternion for gravity compensation")
        w, x, y, z = quat/norm
        rotation = self.pin.Quaternion(w, x, y, z).matrix()
        self.model.gravity.linear = rotation.T @ np.array([0., 0., -9.81])
        q = np.zeros(self.model.nq)
        q[self.q_indices] = q_obs[self.body]
        tau = np.asarray(self.pin.computeGeneralizedGravity(self.model, self.data, q))
        result = np.zeros(29)
        result[15:29] = tau[self.v_indices[15:29]] * self.scale
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite gravity torque")
        return result

    def sample(self, observation, *, enabled=True, now=None):
        if not enabled:
            self.reset()
            return np.zeros(29)
        result = self.raw(observation)
        self.last_raw = result[15:29].copy()
        return result

    def target_observation(self, observation, body_target):
        """XR desired-arm static dynamics, retaining measured waist and IMU."""
        target = np.asarray(body_target, dtype=float)
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("Expected finite body29 target")
        obs = observation.copy()
        obs["q"] = np.asarray(observation["q"], dtype=float).copy()
        obs["q"][self.body[15:29]] = target[15:29]
        return obs

    def sample_target(self, observation, body_target, *, enabled=True):
        if not enabled:
            return self.sample(observation, enabled=False)
        return self.sample(self.target_observation(observation, body_target))

    def summary(self):
        return dict(enabled=True, model="G1 body with rubber hands replaced by Dex1-1; no Dex3 inertia",
                    scale=self.scale,
                    payload=self.payload, end_effector_payload=self.tool_payload,
                    removed_rubber_hand_mass_kg=self.removed_rubber_hands,
                    payload_assumption=self.dex1_reference["assumption"],
                    uncalibrated_extra_loads=["mount/camera adapters", "held objects"],
                    uncompensated_tools=[])


def build_arm_gravity(robot, config, options=None):
    if not config.get("enable_gravity_compensation", False):
        return None
    if config.get("gravity_compensation_joints") not in (None, ["arms"]):
        raise ValueError("LabWAM gravity feedforward supports arms only; legs/waist remain with WBC")
    return ArmGravityCompensator(robot, **(options or {}))


def add_gravity_arguments(parser):
    parser.add_argument("--arm-gravity-scale", type=float, default=1.)
    parser.add_argument("--arm-gravity-payload", type=Path, help="Per-side TOTAL gripper+tool payload override; default nominal Dex1-1")


def gravity_options(args):
    return dict(scale=args.arm_gravity_scale, payload=args.arm_gravity_payload)
