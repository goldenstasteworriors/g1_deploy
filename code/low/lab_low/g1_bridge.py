"""Arm-only adapter for the SONICMJ wired-PICO decoupled controller.

No DDS or ROS imports here. Leg/waist commands remain in the existing WBC.
Target ownership is explicit; elapsed time never expires a command.
"""
import time
import numpy as np

from lab_low.skills import ARM_INDICES, action_mask
from lab_low.gripper_calibration import calibrate_gripper, PICK_TUBE_GRIPPER_CALIBRATION


class ArmTargetBuffer:
    def __init__(self, robot_model, *, max_arm_speed):
        self.robot = robot_model
        self.body = np.asarray(robot_model.get_body_actuated_joint_indices(), dtype=int)
        if len(self.body) != 29:
            raise ValueError("Expected 29 G1 body actuators")
        self.arms = self.body[15:29]
        self.upper = np.asarray(robot_model.get_joint_group_indices("upper_body"), dtype=int)
        if set(self.upper) & set(self.body[:15]):
            raise ValueError("Configure waist_location=lower_body / enable_waist=False")
        self.speed = float(max_arm_speed)
        self.interpolator = None
        self.binding = None
        self.sequence = -1
        self.observation = None
        self.target = None
        self.native = None
        self.last_error = None
        self.hold_target = None
        self.trace_metadata = None
        self.skill = self.hand = None
        self.gripper_trace = None

    def observe(self, observation):
        q = np.asarray(observation["q"], dtype=float)
        left = np.asarray(observation["left_hand_dex1_q"], dtype=float).reshape(-1)
        right = np.asarray(observation["right_hand_dex1_q"], dtype=float).reshape(-1)
        if q.shape != (self.robot.num_dofs,) or left.shape != (1,) or right.shape != (1,):
            raise ValueError("Expected full model state and native one-axis Dex1 feedback")
        state = np.r_[q[self.body], left, right]
        if not np.isfinite(state).all():
            raise ValueError("Invalid measured state")
        self.observation = q.copy()
        self.state = state
        if self.hold_target is None:
            self.hold_target = state[ARM_INDICES].copy()

    def stop(self, *, release=False):
        self.binding, self.target = None, None
        self.sequence = -1
        self.trace_metadata = None
        self.skill = self.hand = None
        if release:
            self.hold_target = None

    def receive(self, goal, now=None):
        now = time.monotonic() if now is None else now
        if self.observation is None:
            raise RuntimeError("Measured state required before a command")
        if self.hold_target is None:
            self.hold_target = self.state[ARM_INDICES].copy()
        if goal.get("lab_stop"):
            if goal.get("lab_binding_id") == self.binding:
                self.stop()
            return
        if "lab_target" not in goal:
            return
        binding = goal["lab_binding_id"]
        sequence = goal["lab_sequence"]
        if not isinstance(binding, str) or not binding or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("Invalid binding/sequence")
        if binding == self.binding and sequence <= self.sequence:
            raise ValueError("Repeated or reordered command")
        active = action_mask(goal["lab_skill"], goal["lab_hand"])
        target = np.asarray(goal["lab_target"], dtype=float)
        if target.shape != (16,) or not np.isfinite(target).all():
            raise ValueError("Expected finite 16D manipulation command")
        if binding != self.binding:
            if not goal.get("lab_reset", False):
                raise ValueError("A new target binding requires lab_reset")
            from decoupled_wbc.control.policy.interpolation_policy import InterpolationPolicy
            self.interpolator = InterpolationPolicy(
                init_time=now,
                init_values={"arms": self.hold_target[np.r_[0:7, 8:15]].copy()},
                max_change_rate=self.speed)
        arm_target = np.r_[target[:7], target[8:15]]
        arm_active = np.r_[active[:7], active[8:15]]
        self.interpolator.set_goal({
            "arms": np.where(arm_active, arm_target, self.hold_target[np.r_[0:7, 8:15]]),
            "target_time": now + 1 / 50,
            "interpolation_garbage_collection_time": now})
        self.binding, self.sequence = binding, sequence
        self.target, self.active = target.copy(), active
        self.skill, self.hand = goal["lab_skill"], goal["lab_hand"]
        self.trace_metadata = goal.get("lab_trace")
        self.last_error = None

    def sample(self, now=None):
        now = time.monotonic() if now is None else now
        if self.observation is None:
            raise RuntimeError("No measured state")
        if self.hold_target is None:
            self.hold_target = self.state[ARM_INDICES].copy()
        if self.target is None:
            command = self.hold_target.copy()
        else:
            command = np.where(self.active, self.target, self.hold_target)
            command[np.r_[0:7, 8:15]] = self.interpolator.get_action(now)["arms"]
            command = np.where(self.active, command, self.hold_target)
        raw_grippers = command[[7, 15]].copy()
        if self.target is not None:
            command = calibrate_gripper(command, skill=self.skill, hand=self.hand,
                                        initialization=self.binding.startswith("init:"))
        self.gripper_trace = dict(
            raw_gated_targets_rad=raw_grippers,
            calibrated_targets_rad=command[[7, 15]].copy(),
            adjusted=(raw_grippers != command[[7, 15]]),
            skill=self.skill, hand=self.hand,
            stage="before native Dex1 command_position validation/feedback limit")
        # Hold the last command, not the next measured position. Tracking a
        # sagging measurement as the setpoint removes the position error.
        self.hold_target = command.copy()
        # Dex1 applies its own native target validation and feedback step limit.
        q = self.observation.copy()
        q[self.arms] = np.r_[command[:7], command[8:15]]
        self.native = command[[7, 15]].copy()
        return {"target_upper_body_pose": q[self.upper], "navigate_cmd": np.zeros(3)}


def install_decoupled_adapter(loop_module, *, gravity_options=None, trace_dir=None):
    """Select LabWAM adapter in a dedicated control process, not the source tree."""
    print("[夹爪标定] 取试管活动夹爪：0.8～1.1 rad 内取 min(预测值, 0.85)；初始化不应用。", flush=True)
    original_factory, original_env = loop_module.get_wbc_policy, loop_module.G1Env
    holder = {}

    class LabEnvironment(original_env):
        def __init__(self, *args, **kwargs):
            from lab_low.gravity import build_arm_gravity
            # Validate dynamics before initializing any robot transport.
            self.lab_gravity = build_arm_gravity(kwargs["robot_model"], kwargs["config"], gravity_options)
            super().__init__(*args, **kwargs)
            if self.lab_gravity is not None:
                from lab_low.gravity import apply_xr_arm_gains
                apply_xr_arm_gains(self.body().body_command_sender, enabled=True)
            self.lab_trace = None
            self.lab_trace_context = {}
            self.lab_observed_ns = None
            if trace_dir is not None:
                from lab_low.control_trace import ControlTrace, TracedPublisher
                sender = self.body().body_command_sender
                body = self.robot_model.get_body_actuated_joint_indices()
                motors = np.asarray(sender.config["JOINT2MOTOR"])[15:29]
                self.lab_trace = ControlTrace(trace_dir, dict(
                    gravity=None if self.lab_gravity is None else self.lab_gravity.summary(),
                    gripper_calibration=PICK_TUBE_GRIPPER_CALIBRATION,
                    arm_joint_names=[self.robot_model.joint_names[i] for i in body[15:29]],
                    arm_motor_indices=motors, target16_order="left_arm7,left_Dex1,right_arm7,right_Dex1",
                    write_scope="local DDS Write; not acknowledgement of motor execution"))
                sender.lowcmd_publisher_ = TracedPublisher(sender.lowcmd_publisher_, self.lab_trace,
                    lambda: dict(self.lab_trace_context, damping=bool(self._damping_stopped),
                                 phase="damped" if self._damping_stopped else getattr(self, "operator_phase", None)), motors)

        def damping_stop(self):
            try:
                return super().damping_stop()
            finally:
                if self.lab_trace is not None:
                    self.lab_trace.close()

        def enter_damping(self):
            with self._action_lock:
                if self.lab_gravity is not None:
                    self.lab_gravity.reset()
                return super().enter_damping()

        def restart_initialization(self):
            with self._action_lock:
                if self.lab_gravity is not None:
                    self.lab_gravity.reset()
                return super().restart_initialization()

        def observe(self):
            observation = super().observe()
            self.lab_observed_ns = time.monotonic_ns()
            buffer = holder.get("buffer")
            observation["lab_binding_id"] = None if buffer is None else buffer.binding
            observation["lab_last_sequence"] = -1 if buffer is None else buffer.sequence
            observation["lab_error"] = None if buffer is None else buffer.last_error
            observation["lab_hold_target"] = (None if buffer is None or buffer.hold_target is None
                                               else buffer.hold_target.copy())
            return observation

        def _queue_action(self, action):
            if self.lab_trace is not None:
                buffer = holder.get("buffer")
                body = self.robot_model.get_body_actuated_joint_indices()
                self.lab_trace_context = dict(
                    phase=getattr(self, "operator_phase", None),
                    epoch=getattr(self, "lab_operator_epoch", None),
                    observation_monotonic_ns=self.lab_observed_ns,
                    binding_id=None if buffer is None else buffer.binding,
                    command_sequence=None if buffer is None else buffer.sequence,
                    requested_target16=None if buffer is None or buffer.target is None else buffer.target.copy(),
                    gripper_calibration=None if buffer is None else buffer.gripper_trace,
                    model=None if buffer is None else buffer.trace_metadata,
                    measured_arm_q=None if self.last_obs is None else self.last_obs["q"][body[15:29]].copy(),
                    measured_arm_dq=None if self.last_obs is None or "dq" not in self.last_obs else self.last_obs["dq"][body[15:29]].copy(),
                    measured_gripper_q=None if self.last_obs is None else [self.last_obs["left_hand_dex1_q"][0], self.last_obs["right_hand_dex1_q"][0]])
            if self.last_obs is not None:
                action = self.safety_monitor.handle_violations(self.last_obs, action)["action"]
            if self.config.get("diagnostic_read_only", False):
                return
            body_q = self.robot_model.get_body_actuated_joints(action["q"])
            body_tau = np.zeros_like(body_q)
            if self.lab_gravity is not None:
                from lab_low.gravity import apply_xr_arm_gains
                enabled = action.get("lab_control_arms", True)
                apply_xr_arm_gains(self.body().body_command_sender, enabled=enabled)
                body_tau = self.lab_gravity.sample_target(self.last_obs, body_q, enabled=enabled)
            self.body().queue_action({"body_q": body_q, "body_dq": np.zeros_like(body_q),
                                      "body_tau": body_tau})
            link = self.body().body_command_sender.onboard_link
            if link is not None and link.disconnected:
                return
            if not action.get("lab_control_hands", True):
                return
            # This process owns the sole Dex1 writer. Never convert to binary grasp modes.
            targets = action.get("lab_dex1_targets")
            if targets is None:
                targets = [float(self.last_obs[f"{side}_hand_dex1_q"][0]) for side in ("left", "right")]
            for side, value in zip(("left", "right"), targets):
                getattr(self.hands(), side).command_position(float(value))

    def factory(robot_type, robot_model, wbc_config, *args, **kwargs):
        if wbc_config.get("HAND_TYPE") != "dex1" or not wbc_config.get("with_hands") or wbc_config.get("enable_waist"):
            raise ValueError("Lab deployment requires Dex1, with_hands=True, enable_waist=False")
        policy = original_factory(robot_type, robot_model, wbc_config, *args, **kwargs)
        buffer = ArmTargetBuffer(robot_model, max_arm_speed=wbc_config["upper_body_max_joint_speed"])
        holder["buffer"] = buffer
        original_observe, original_action = policy.set_observation, policy.get_action

        class Upper:
            def set_goal(self, goal):
                buffer.receive(goal)

            def get_action(self, time=None):
                return buffer.sample(time)

        lab_upper = Upper()

        def observe(observation):
            buffer.observe(observation)
            original_observe(observation)

        def goal(message):
            try:
                if "target_upper_body_pose" in message:
                    raise ValueError("Do not run the PICO arm publisher alongside LabWAM")
                buffer.receive(message)
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                # An invalid upper-body command must not take down balance.
                buffer.stop()
                buffer.last_error = str(error)
                policy.upper_body_policy = lab_upper
                return
            if "lab_target" in message:
                policy.upper_body_policy = lab_upper
            policy.lower_body_policy.set_use_teleop_policy_cmd(True)
            policy.lower_body_policy.set_goal({k: message[k] for k in ("set_policy_action", "toggle_policy_action") if k in message})

        def action(time=None):
            if policy.upper_body_policy is not lab_upper:
                # CollectionMode restores its own startup policy after damping.
                buffer.stop(release=True)
            result = original_action(time)
            result["lab_dex1_targets"] = (buffer.native.copy() if policy.upper_body_policy is lab_upper
                                           else buffer.state[[29, 30]].copy())
            return result

        policy.set_observation, policy.set_goal, policy.get_action = observe, goal, action
        policy.lab_buffer, policy.lab_upper_policy = buffer, lab_upper
        return policy

    loop_module.get_wbc_policy, loop_module.G1Env = factory, LabEnvironment
