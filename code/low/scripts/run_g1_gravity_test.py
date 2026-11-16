"""Independent G1 arm gravity test using the native SONICMJ balance/damping loop."""
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from lab_low.gravity import XR_KP, XR_KD, apply_xr_arm_gains


def install(loop, *, execute, options, arm_control="pure"):
    from lab_low.gravity import ArmGravityCompensator
    original_env = loop.G1Env

    class GravityEnv(original_env):
        def __init__(self, *args, **kwargs):
            kwargs["config"] = dict(kwargs["config"], diagnostic_read_only=True)
            self.compensator = ArmGravityCompensator(kwargs["robot_model"], **options)
            print("[重力载荷] " + json.dumps(self.compensator.summary(), ensure_ascii=False), flush=True)
            self.test_phase = "idle"
            super().__init__(*args, **kwargs)
            sender = self.body().body_command_sender
            self.arm_motors = np.asarray(sender.config["JOINT2MOTOR"])[15:29]
            # Pure mode disables both P and D; d still uses native full damping.
            sender.robot_kp[self.arm_motors] = 0.
            self.arm_kd = sender.robot_kd[self.arm_motors].copy()
            self.arm_hold_q = None
            self.body_indices = np.asarray(self.robot_model.get_body_actuated_joint_indices())

        def gravity_observation(self):
            if arm_control == "xr-hold" and self.arm_hold_q is not None:
                target = self.robot_model.get_body_actuated_joints(self.last_obs["q"]).copy()
                target[15:29] = self.arm_hold_q
                return self.compensator.target_observation(self.last_obs, target)
            return self.last_obs

        def enter_damping(self):
            if hasattr(self, "test_cancel"):
                self.test_cancel.set()
            return super().enter_damping()

        def _queue_action(self, action):
            if self.test_phase in ("idle", "taking_control", "damped"):
                return
            action = self.safety_monitor.handle_violations(self.last_obs, action)["action"]
            if self.config["diagnostic_read_only"]:
                return
            q = self.robot_model.get_body_actuated_joints(action["q"])
            sender = self.body().body_command_sender
            active = self.test_phase == "running"
            xr_hold = active and arm_control == "xr-hold"
            sender.robot_kp[self.arm_motors] = XR_KP if xr_hold else 0.
            sender.robot_kd[self.arm_motors] = XR_KD if xr_hold else 0.
            if xr_hold:
                apply_xr_arm_gains(sender, enabled=True)
                q[15:29] = self.arm_hold_q
            # XR computes RNEA at the desired arm pose with zero dq/ddq.
            # computeGeneralizedGravity is the same static dynamics term.
            tau = (self.compensator.sample_target(self.last_obs, q) if xr_hold
                   else self.compensator.sample(self.last_obs, enabled=active))
            self.body().queue_action(dict(body_q=q, body_dq=np.zeros_like(q), body_tau=tau))
            # No hand writer: config.with_hands=False.

    class Mode:
        def __init__(self, env, policy, robot_model, config, shutdown):
            self.env, self.policy, self.shutdown = env, policy, shutdown
            self.state = "idle"
            self.events = queue.SimpleQueue()
            self.results = queue.SimpleQueue()
            self.cancel = threading.Event()
            env.test_cancel = self.cancel
            self.thread = None
            self.last_report = float("-inf")
            self.owns_control = False
            upper_indices = robot_model.get_joint_group_indices("upper_body")

            class MeasuredUpper:
                def get_action(inner, time=None):
                    return dict(target_upper_body_pose=env.last_obs["q"][upper_indices], navigate_cmd=np.zeros(3))

            policy.upper_body_policy = MeasuredUpper()
            self.notice()

        def notice(self):
            hints = {"idle": "待机，无电机写入。第一次 Enter 启用下半身站立。",
                     "taking_control": "正在接管厂商模式；d 可立即阻尼。",
                     "standing": "下半身启动渐变中。",
                     "standing_ready": "请确认站稳，第二次 Enter 开启双臂重力补偿。",
                     "running": ("XR 双臂姿态保持＋重力前馈：肩肘 Kp/Kd=80/3，腕部40/1.5。d 全身阻尼，Ctrl+C 退出。"
                                 if arm_control == "xr-hold" else "纯重力补偿已开启：Kp=0、Kd=0；含名义 Dex1-1 载荷，不锁定姿态。d 全身阻尼，Ctrl+C 退出。"),
                     "damped": "全身阻尼。r 返回待机，随后重新按 Enter。"}
            print(f"[GravityTest {self.state}] {hints[self.state]}", flush=True)

        def handle_keyboard_button(self, key):
            if key == "d":
                self.shutdown.request_estop("keyboard:d")
            else:
                self.events.put(key)
            return True

        def enter_damping(self, reason):
            self.cancel.set()
            self.env.enter_damping()
            self.env.test_phase = self.state = "damped"
            self.policy.lower_body_policy.use_policy_action = False
            while not self.events.empty():
                self.events.get()
            self.shutdown.estop_requested.clear()
            self.notice()

        def takeover(self):
            try:
                if execute and not self.owns_control:
                    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
                    client = MotionSwitcherClient()
                    client.SetTimeout(5.)
                    client.Init()
                    while not self.cancel.is_set():
                        status, result = client.CheckMode()
                        if status != 0:
                            raise RuntimeError(f"CheckMode failed: {status}")
                        if not result.get("name"):
                            self.owns_control = True
                            break
                        client.ReleaseMode()
                        self.cancel.wait(1.)
                self.results.put(None)
            except Exception as error:
                self.results.put(str(error))
            finally:
                if self.cancel.is_set():
                    self.env.send_damping()

        def update(self, goal, now):
            goal.clear()  # No external arm or navigation goals in this test.
            if self.shutdown.estop_requested.is_set():
                self.enter_damping("keyboard")
            while not self.events.empty():
                key = self.events.get()
                if key == "r" and self.state == "damped" and (self.thread is None or not self.thread.is_alive()):
                    self.state = self.env.test_phase = "idle"
                    self.notice()
                elif key == "enter" and self.state == "idle":
                    self.cancel.clear()
                    while not self.results.empty():
                        self.results.get()
                    self.state = self.env.test_phase = "taking_control"
                    self.env.config["diagnostic_read_only"] = not execute
                    self.thread = threading.Thread(target=self.takeover, daemon=True)
                    self.thread.start()
                    self.notice()
                elif key == "enter" and self.state == "standing_ready":
                    # Capture exactly once; feedback must not move this target.
                    self.env.arm_hold_q = self.env.robot_model.get_body_actuated_joints(self.env.last_obs["q"])[15:29].copy()
                    self.state = self.env.test_phase = "running"
                    self.notice()
            if self.state == "taking_control" and not self.results.empty():
                error = self.results.get()
                if error:
                    print(error, flush=True)
                    self.enter_damping(error)
                else:
                    self.env.restart_initialization()
                    self.state = self.env.test_phase = "standing"
                    self.notice()
            if self.state == "standing" and self.env.safety_monitor.startup_complete:
                self.state = self.env.test_phase = "standing_ready"
                self.notice()
            lower = self.policy.lower_body_policy
            lower.use_policy_action = self.state in ("standing", "standing_ready", "running")
            lower.cmd[:] = 0.
            if now-self.last_report >= 1.:
                tau = self.env.compensator.raw(self.env.gravity_observation())[15:29]
                sender = self.env.body().body_command_sender
                print(json.dumps(dict(phase=self.state, arm_control=arm_control,
                                     actuator_output=execute and self.state == "running",
                                     calculated_arm_tau_nm=tau.round(4).tolist(),
                                     arm_kp=sender.robot_kp[self.env.arm_motors].tolist(),
                                     arm_kd=sender.robot_kd[self.env.arm_motors].tolist(),
                                     arm_target=None if self.env.arm_hold_q is None else self.env.arm_hold_q.tolist(),
                                     arm_feedback=self.env.robot_model.get_body_actuated_joints(self.env.last_obs["q"])[15:29].tolist())), flush=True)
                self.last_report = now

        def status(self):
            return dict(control_mode=self.state, gravity_test=True)

    loop.G1Env, loop.CollectionMode = GravityEnv, Mode


def main():
    from lab_low.gravity import add_gravity_arguments, gravity_options, ArmGravityCompensator
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sonic-root", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--arm-control", choices=["pure", "xr-hold"], default="pure")
    add_gravity_arguments(p)
    args, rest = p.parse_known_args()
    sys.path.insert(0, str(args.sonic_root))
    sys.path.insert(0, str(args.sonic_root/"external_dependencies/unitree_sdk2_python"))
    import tyro
    from decoupled_wbc.control.main.teleop import run_g1_control_loop as loop
    cfg = tyro.cli(loop.ControlLoopConfig, args=rest)
    cfg.with_hands, cfg.enable_waist, cfg.startup_t_pose = False, False, False
    cfg.diagnostic_read_only = True
    options = gravity_options(args)
    if args.check_config:
        robot = loop.instantiate_g1_robot_model(waist_location="lower_body", with_hands=False)
        gravity = ArmGravityCompensator(robot, **options)
        loop.get_wbc_policy("g1", robot, cfg.load_wbc_yaml())
        print(json.dumps(dict(check="gravity-only configuration", robot_io_opened=False,
                              arm_control=args.arm_control,
                              arm_kp=XR_KP.tolist() if args.arm_control == "xr-hold" else 0,
                              arm_kd=XR_KD.tolist() if args.arm_control == "xr-hold" else 0, hands_controlled=False,
                              gravity=gravity.summary())))
        return
    install(loop, execute=args.execute, options=options, arm_control=args.arm_control)
    print("请先退出旧部署控制程序。此测试使用唯一身体控制进程；下半身使用原站立策略。", flush=True)
    loop.main(cfg)


if __name__ == "__main__":
    main()
