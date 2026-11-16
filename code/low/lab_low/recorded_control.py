"""Live-state command pump for execution-indexed recorded vision.

No camera, model RPC, recorded joints, or DDS writer in this module. The original
operator owns d/r and rejects old epochs. A step means a published target held
for one control period, not proof of physical task completion.
"""
from collections import deque
import queue
import threading
import time

import numpy as np

from lab_low.skills import ARM_INDICES, action_mask


class LiveCommandPump:
    def __init__(self, read_state, publish, *, skill, hand, binding, epoch=None):
        self.read_state, self.publish = read_state, publish
        self.skill, self.hand, self.binding, self.epoch = skill, hand, binding, epoch
        self.active = action_mask(skill, hand)
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.commands, self.results = deque(), queue.Queue()
        self.latest = self.error = None
        self.hold_target = None
        self.prediction = None
        self.trace = None
        self.stream = None
        self.stream_started = False
        self.allow = self.in_flight = False
        self.sequence, self.armed = 0, False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.condition:
            while self.latest is None and self.error is None and not self.stopped.is_set():
                self.condition.wait()
            if self.error is not None:
                raise RuntimeError(self.error)
            if self.stopped.is_set():
                raise RuntimeError("State pump stopped")
            return self.latest.copy()

    def target_snapshot(self):
        with self.condition:
            return self.hold_target.copy()

    def enable(self):
        with self.condition:
            self.allow = True

    def execute(self, actions, *, traces=None):
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
            raise ValueError("Expected finite [N,16] action prefix")
        with self.condition:
            if self.commands or self.in_flight:
                raise RuntimeError("Previous action prefix is still executing")
            if self.error is not None:
                raise RuntimeError(self.error)
            if traces is not None and len(traces) != len(actions):
                raise ValueError("One trace per action required")
            self.commands.extend((a.copy(), traces[i].copy() if traces is not None else
                                 dict(self.prediction or {}, action_index=i, model_target16=a.tolist()))
                                 for i, a in enumerate(actions))
        states = []
        for _ in actions:
            state = self.results.get()
            if isinstance(state, Exception):
                raise state
            states.append(state)
        return states

    def set_prediction(self, metadata):
        with self.condition:
            self.prediction = metadata.copy()

    def execute_stream(self, stream):
        """Choose the next target at the feedback tick, without an idle tick.

        The generator only selects actions and reads sequential recorded RGB;
        model RPC and image encoding stay on the prefetch worker.
        """
        done = threading.Event()
        with self.condition:
            if self.stream is not None or self.commands or self.in_flight:
                raise RuntimeError("Another command sequence is running")
            self.stream, self.stream_done = stream, done
            self.stream_started = False
        while not done.wait(.05):
            if self.stopped.is_set():
                raise RuntimeError(self.error or "State pump stopped")
        if self.error is not None:
            raise RuntimeError(self.error)

    def _tick(self):
        message = self.read_state()
        if message is None:
            return
        state = np.asarray(message["state"], dtype=float)
        if state.shape != (31,) or not np.isfinite(state).all():
            raise ValueError("Live state must contain finite body29 + left/right Dex1 angles")
        with self.condition:
            self.latest = state.copy()
            if self.hold_target is None:
                held = message.get("lab_hold_target")
                self.hold_target = np.asarray(state[ARM_INDICES] if held is None else held, dtype=float).copy()
                if self.hold_target.shape != (16,) or not np.isfinite(self.hold_target).all():
                    raise ValueError("Invalid initial hold target")
            self.condition.notify_all()
            if self.publish is not None and self.allow:
                if message.get("control_mode") != "running" or message.get("lab_operator_epoch") != self.epoch:
                    raise RuntimeError("Operator stopped or changed epoch; discard recorded session")
                if message.get("lab_binding_id") == self.binding:
                    self.armed = True
                elif self.armed:
                    raise RuntimeError("Operator revoked this target binding")
            # Feedback at this tick belongs to the target sent one period ago.
            if self.in_flight:
                self.results.put(state.copy())
                self.in_flight = False
            target = self.hold_target.copy()
            if self.stream is not None:
                try:
                    item = self.stream.send(state.copy()) if self.stream_started else next(self.stream)
                    self.stream_started = True
                    model_target, self.trace = item
                    target = np.where(self.active, model_target, target)
                except StopIteration:
                    self.stream = None
                    self.stream_done.set()
            elif self.commands:
                model_target, self.trace = self.commands.popleft()
                target = np.where(self.active, model_target, target)
                self.in_flight = True
            self.hold_target = target.copy()
            if self.publish is not None and self.allow:
                self.publish(dict(lab_target=target.tolist(), lab_skill=self.skill, lab_hand=self.hand,
                    lab_binding_id=self.binding, lab_sequence=self.sequence, lab_reset=not self.armed,
                    lab_trace=self.trace,
                    lab_operator_epoch=self.epoch, lab_sent_monotonic=time.monotonic()))
                self.sequence += 1

    def _run(self):
        deadline = time.monotonic()
        try:
            while not self.stopped.is_set():
                self._tick()
                deadline = max(deadline + 1 / 30, time.monotonic())
                self.stopped.wait(max(0., deadline - time.monotonic()))
        except Exception as error:
            with self.condition:
                self.error = str(error)
                self.results.put(RuntimeError(self.error))
                self.condition.notify_all()
            self.stopped.set()

    def close(self):
        with self.condition:
            self.allow = False
            self.commands.clear()
            self.stream = None
            self.stopped.set()
            self.condition.notify_all()
            self.results.put(RuntimeError("State pump stopped"))
        self.thread.join()
        if self.publish is not None:
            self.publish(dict(lab_stop=True, lab_binding_id=self.binding, lab_operator_epoch=self.epoch))


class ROSStateReader:
    def __init__(self):
        from decoupled_wbc.control.main.constants import STATE_TOPIC_NAME
        from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgSubscriber
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
        self.manager = ROSManager(node_name="labwam_recorded_vision")
        self.subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
        robot = instantiate_g1_robot_model(waist_location="lower_body")
        self.body = robot.get_body_actuated_joint_indices()
        self.latest = None

    def read(self):
        message = self.subscriber.get_msg()
        if message is not None:
            self.latest = dict(message, state=np.r_[np.asarray(message["q"])[self.body],
                message["left_hand_dex1_q"], message["right_hand_dex1_q"]])
        return self.latest

    def close(self):
        from decoupled_wbc.control.utils.ros_utils import ROSManager
        ROSManager.shutdown()
        self.manager.thread.join()


class ReadOnlyDDSStateReader:
    """Subscriber-only input check without launching/releasing a robot controller."""
    def __init__(self, interface):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
        cfg = ControlLoopConfig(interface=interface, diagnostic_read_only=True,
                                with_hands=True, hand_type="dex1", enable_waist=False).load_wbc_yaml()
        self.mapping = cfg["JOINT2MOTOR"]
        ChannelFactoryInitialize(0, interface)
        self.messages, self.subscribers = {}, []
        for name, topic, typ in (("body", "rt/lowstate", LowState_),
                                 ("left", "rt/dex1/left/state", MotorStates_),
                                 ("right", "rt/dex1/right/state", MotorStates_)):
            subscriber = ChannelSubscriber(topic, typ)
            subscriber.Init(lambda msg, key=name: self.messages.__setitem__(key, msg), 10)
            self.subscribers.append(subscriber)

    def read(self):
        messages = self.messages.copy()
        if set(messages) != {"body", "left", "right"}:
            return None
        return dict(state=np.array([messages["body"].motor_state[i].q for i in self.mapping] +
                                   [messages["left"].states[0].q, messages["right"].states[0].q]))

    def close(self):
        for subscriber in self.subscribers:
            subscriber.Close()
