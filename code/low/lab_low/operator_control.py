"""Three-Enter operator lifecycle around the original SONICMJ control loop.

The keyboard/damping path never waits for model HTTP. Recorded legs are never
commanded. Initialization uses both arms and native Dex1 achieved positions.
"""
import json
import queue
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
from lab_low.initial_poses import SKILL_NAMES_ZH


class OperatorState:
    def __init__(self):
        self.phase, self.epoch, self.recovering = "idle", 0, False

    def key(self, key):
        before, previous_epoch = self.phase, self.epoch
        if key == "d":
            if self.phase != "damped":
                self.epoch += 1
            self.phase = "damped"
        elif key == "r" and self.phase == "damped":
            self.phase, self.recovering = "standing", True
            self.epoch += 1
        elif key == "r" and self.phase in ("running", "ready", "arm_initializing", "standing_ready", "fault", "paused"):
            self.phase, self.recovering = "arm_initializing", False
            self.epoch += 1
        elif key == "enter":
            self.phase = {"idle": "standing", "standing_ready": "arm_initializing",
                          "ready": "running"}.get(self.phase, self.phase)
            if self.phase != before:
                self.epoch += 1
        return before != self.phase or previous_epoch != self.epoch

    def tick(self, *, stand_ready=False, pose_ready=False):
        if self.phase == "standing" and stand_ready:
            self.phase = "arm_initializing" if self.recovering else "standing_ready"
        elif self.phase == "arm_initializing" and pose_ready:
            self.phase, self.recovering = "ready", False


def validate_pose(robot, pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (16,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite 16D achieved pose")


def install_operator(loop, *, catalog, selection_path, marks, sonic_root, policy_url,
                     log_dir, execute, chunks, policy_ready, on_close=lambda: None,
                     camera_host="127.0.0.1", camera_port=5555, gravity_options=None, recorded_dataset=None,
                     inference_mode="sync", inference_horizon=32, inference_delay_steps=16, denoise_steps=10, chunk_smoothing_steps=0,
                     live_grounding_config=None):
    """Install only in this launcher process; leave SONICMJ source untouched."""
    from lab_low.g1_bridge import install_decoupled_adapter
    install_decoupled_adapter(loop, gravity_options=gravity_options, trace_dir=log_dir)
    base_env, base_shutdown = loop.G1Env, loop.DampingShutdown
    holder = {}
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    class SessionEnv(base_env):
        def __init__(self, *args, **kwargs):
            # Do not release the vendor's standing mode while merely opening
            # the program. The first Enter owns the explicit takeover.
            kwargs["config"] = dict(kwargs["config"], diagnostic_read_only=True)
            super().__init__(*args, **kwargs)
            self.operator_phase = "idle"
            self.operator_arms_enabled = False
            sender = self.body().body_command_sender
            self.arm_motors = np.asarray(sender.config["JOINT2MOTOR"])[15:29]
            self.arm_kp = sender.robot_kp[self.arm_motors].copy()
            self.arm_kd = sender.robot_kd[self.arm_motors].copy()

        def _queue_action(self, action):
            if self.operator_phase in ("idle", "taking_control"):
                return
            arms_enabled = self.operator_arms_enabled and self.operator_phase not in ("standing", "standing_ready", "damped")
            sender = self.body().body_command_sender
            sender.robot_kp[self.arm_motors] = self.arm_kp if arms_enabled else 0.
            sender.robot_kd[self.arm_motors] = self.arm_kd if arms_enabled else 0.
            super()._queue_action({**action, "lab_control_hands": arms_enabled, "lab_control_arms": arms_enabled})

    class SessionMode:
        def __init__(self, env, policy, robot_model, config, shutdown):
            self.env, self.policy, self.robot, self.shutdown = env, policy, robot_model, shutdown
            self.machine = OperatorState()
            self.events = queue.SimpleQueue()
            self.selection = catalog.load_selection(selection_path)
            if recorded_dataset is not None:
                recorded_dataset.task(self.selection)
            self.target = catalog.pose(self.selection)
            validate_pose(self.robot, self.target)
            self.client, self.retiring = None, []
            self.client_log = None
            self.menu_input = None
            self.pose_start = self.pose_origin = None
            self.stand_started = None
            self.takeover_thread = None
            self.takeover_result = queue.SimpleQueue()
            self.takeover_cancel = threading.Event()
            self.owns_control = False
            self.sequence = 0
            self.last_notice = None
            self.last_fault = None
            self.arms_engaged = False
            self.last_progress = float("-inf")
            self.grounding_process = None
            self.grounding_ready = live_grounding_config is None
            self.grounding_log = None
            holder["mode"] = self
            # Feed measured arms to the lower-body policy while arms are free.
            buffer = policy.lab_buffer
            class HoldUpper:
                def get_action(self, time=None):
                    return {"target_upper_body_pose": buffer.observation[buffer.upper], "navigate_cmd": np.zeros(3)}
            self.hold_upper = HoldUpper()
            policy.upper_body_policy = self.hold_upper
            self.notice()

        @property
        def state(self):
            return self.machine.phase

        def notice(self):
            model_ready = bool(policy_ready())
            text = (self.state, json.dumps(self.selection), self.menu_input is not None, model_ready, self.last_fault,
                    self.grounding_ready, self.grounding_process is not None)
            if text != self.last_notice:
                stages = {
                    "idle": ("0/3 待机", "尚未接管身体和夹爪。按第一次 Enter 开始下半身站立。"),
                    "standing": ("1/3 正在启动下半身", "正在接管并启用站立策略，请等待，不要连续按 Enter。"),
                    "standing_ready": ("1/3 下半身初始化等待已结束", "站立策略已启用；请目视确认站稳，再按第二次 Enter 初始化双臂和夹爪。"),
                    "arm_initializing": ("2/3 正在初始化双臂和夹爪", "正在缓慢进入所选轨迹姿态，请等待插值完成提示。尚未进行模型推理。"),
                    "ready": ("2/3 初始化轨迹已发送", "请目视确认双臂和夹爪到位；检查物体、推理任务和初始框；模型就绪后按第三次 Enter 开始推理。"),
                    "running": ("3/3 推理执行中", "模型正在循环预测和执行；r 停止推理并直接回初始姿态，保持站立；d 全身阻尼急停。"),
                    "damped": ("已急停：全身阻尼", "不再主动站立。确认支撑和姿态后按 r，恢复站立并返回初始姿态。"),
                    "fault": ("故障：推理未运行", "下半身控制仍可能运行！d 是全身阻尼急停；查清原因后再用 r 恢复。"),
                    "paused": ("推理已停止", "下半身继续站立。按 r 返回初始姿态，或在有支撑时按 d。"),
                }
                title, hint = stages[self.state]
                if live_grounding_config is not None and self.state == "ready":
                    if self.grounding_process is not None:
                        hint = "双臂初始化轨迹已发送；正在取图/等待 Qwen 标记，请等待目标就绪。d 仍可立即急停。"
                    elif not self.grounding_ready:
                        hint = "请目视确认双臂和夹爪到位，然后按 b 捕获实时图像并标记目标；目标和模型都就绪后再按第三次 Enter 推理。"
                    else:
                        hint = "实时初始目标已就绪；确认预览中的目标正确且模型就绪后，按第三次 Enter 推理。需要更新目标时按 b。"
                s = self.selection
                print(f"\n[LabOperator 阶段 {title}] ({self.state})\n  {hint}\n"
                      f"  初始姿态来源：{SKILL_NAMES_ZH.get(s['skill'], s['skill'])} / 轨迹 {s['episode']} / 帧 {s['frame']}\n"
                      f"  模型：{'已就绪' if model_ready else '尚未就绪'}；"
                      + ("数据集模式：p 同时选择姿态、视觉起始帧和推理技能。" if recorded_dataset is not None
                         else "初始姿态选择不等于推理技能选择。"), flush=True)
                if self.state in ("idle", "ready"):
                    try:
                        task = (recorded_dataset.task(self.selection) if recorded_dataset is not None
                                else json.loads(Path(marks).read_text()))
                        print(f"  推理任务：{SKILL_NAMES_ZH.get(task['skill'], task['skill'])} / {task['hand']}；初始框 {len(task.get('targets', []))} 个。", flush=True)
                        if recorded_dataset is not None:
                            print(f"  图像：数据集轨迹 {self.selection['episode']}，从帧 {self.selection['frame']} 开始；"
                                  f"框保留在原标注帧 {task['reference_frame']}。\n  Prompt: {task['prompt']}", flush=True)
                    except (OSError, ValueError, KeyError, TypeError):
                        if live_grounding_config is not None:
                            print("  尚无本次实时标记：双臂到位后按 b 自动获取，无需手填 marks.json。", flush=True)
                        else:
                            print(f"  推理任务尚未正确配置，推理前请填写：{marks}", flush=True)
                    print(("  b：实时取图/自动打框（加液无框，仅取图）；" if live_grounding_config is not None else "  ")
                          + "p：选择初始轨迹帧；r：直接回初始姿态；d：阻尼急停；Ctrl+C：阻尼退出。", flush=True)
                if live_grounding_config is not None:
                    status = "正在生成，请等待" if self.grounding_process is not None else ("已就绪" if self.grounding_ready else "未就绪，请在双臂到位后按 b")
                    print(f"  实时目标：{status}。b 仅在待机、站立就绪、双臂就绪或阻尼状态可用。", flush=True)
                if self.state == "fault":
                    print(f"  双臂：{'保持上一条有效目标（含已启用的补偿）' if self.arms_engaged else '尚未接管'}\n"
                          f"  原因：{self.last_fault}", flush=True)
                self.last_notice = text

        def fail(self, error):
            self.last_fault = f"{self.state}: {error}"
            self.policy.lab_buffer.stop()
            self.machine.phase = "fault"
            self.notice()

        def handle_keyboard_button(self, key):
            # The original KeyboardEStop handles d first and sends damping
            # immediately, independent of the 50 Hz loop and HTTP inference.
            if key == "d":
                self.shutdown.request_estop("keyboard:d")
            else:
                self.events.put(key)
            return True  # never pass navigation/legacy restart keys downstream

        def stop_client(self):
            if self.client is not None:
                if self.client.poll() is None:
                    self.client.terminate()
                    self.retiring.append((self.client, time.monotonic()))
                self.client = None
            self.policy.lab_buffer.stop()

        def enter_damping(self, reason):
            self.takeover_cancel.set()
            while not self.events.empty():
                self.events.get()
            self.stop_client()
            self.policy.lab_buffer.stop(release=True)
            self.machine.key("d")
            self.arms_engaged = self.env.operator_arms_enabled = False
            self.menu_input = None
            self.policy.lower_body_policy.use_policy_action = False
            self.env.enter_damping()
            self.env.operator_phase = "damped"
            self.notice()

        def takeover(self, epoch):
            """Native vendor RPC timing, on a cancellable thread independent of d."""
            try:
                if execute and not self.owns_control:
                    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
                    client = MotionSwitcherClient()
                    client.SetTimeout(5.)
                    client.Init()
                    while not self.takeover_cancel.is_set():
                        if self.takeover_cancel.is_set():
                            return
                        status, result = client.CheckMode()
                        if status != 0:
                            raise RuntimeError(f"CheckMode failed: {status}")
                        if not result.get("name"):
                            self.owns_control = True
                            break
                        if self.takeover_cancel.is_set():
                            return
                        status = client.ReleaseMode()
                        if self.takeover_cancel.wait(1.):
                            return
                self.takeover_result.put((epoch, None))
            except Exception as error:
                self.takeover_result.put((epoch, str(error)))
            finally:
                if self.takeover_cancel.is_set():
                    self.env.send_damping()  # also cover an RPC that returned after d

        def begin_stand(self, now):
            self.stop_client()
            self.policy.lab_buffer.stop(release=True)
            self.arms_engaged = self.env.operator_arms_enabled = False
            self.last_fault = None
            self.policy.upper_body_policy = self.hold_upper
            lower = self.policy.lower_body_policy
            lower.use_policy_action = False
            lower._waist_initial_time = None
            lower.action[:] = 0.
            lower.obs_history.clear()
            lower.cmd = lower.config["cmd_init"].copy()
            lower.cmd[:] = 0.
            lower.height_cmd = lower.config["height_cmd"]
            lower.freq_cmd = lower.config["freq_cmd"]
            lower.roll_cmd, lower.pitch_cmd, lower.yaw_cmd = lower.config["rpy_cmd"]
            self.stand_started = None
            self.env.config["diagnostic_read_only"] = not execute
            self.env.operator_phase = "taking_control"
            self.takeover_cancel.clear()
            self.takeover_thread = threading.Thread(target=self.takeover, args=(self.machine.epoch,), daemon=True)
            self.takeover_thread.start()

        def begin_pose(self, now):
            self.notice()
            self.policy.lab_buffer.stop()
            self.pose_origin = self.policy.lab_buffer.state[np.r_[15:22, 29, 22:29, 30]].copy()
            self.pose_start = now
            self.pose_binding = f"init:{self.machine.epoch}:{self.pose_start}"
            self.arms_engaged = True
            self.last_progress = float("-inf")
            # Reuse SONICMJ's configured startup ramp duration for the selected
            # pose transition; no extra speed, arrival tolerance or timeout.
            monitor = self.env.safety_monitor
            self.pose_duration = monitor.ramp_duration_steps / monitor.control_frequency
            self.sequence = 0

        def pose_tick(self, now):
            fraction = min(1., max(0., (now-self.pose_start)/self.pose_duration))
            target = self.pose_origin + fraction * (self.target-self.pose_origin)
            self.policy.set_goal(dict(lab_target=target, lab_binding_id=self.pose_binding,
                lab_sequence=self.sequence, lab_sent_monotonic=time.monotonic(), lab_skill="dispense_into_held_test_tube",
                lab_hand="both", lab_reset=True))
            self.sequence += 1
            if self.policy.lab_buffer.last_error:
                raise ValueError(self.policy.lab_buffer.last_error)
            measured = self.policy.lab_buffer.state[np.r_[15:22, 29, 22:29, 30]]
            if self.state == "arm_initializing" and now-self.last_progress >= 1.:
                arm_error = np.max(np.abs((measured-self.target)[np.r_[0:7, 8:15]]))
                hand_error = np.max(np.abs((measured-self.target)[[7, 15]]))
                print(f"[阶段 2/3] 轨迹进度 {fraction*100:.0f}% | 双臂最大偏差 {arm_error:.3f} rad "
                      f"| 夹爪最大偏差 {hand_error:.3f} rad | 请目视确认实际姿态", flush=True)
                self.last_progress = now
            return fraction == 1.

        def start_client(self):
            if not self.grounding_ready:
                raise ValueError("实时初始目标尚未就绪：双臂到位后按 b 获取当前画面的框；等待成功后再按 Enter")
            if self.retiring:
                raise ValueError("Previous inference client is still stopping; press Enter again after it exits")
            if not policy_ready():
                raise ValueError("Policy service is not ready; initialization remains active")
            if recorded_dataset is None and not Path(marks).is_file():
                raise ValueError(f"Provide fresh skill/hand/initial marks at {marks}")
            # Validate before any inference transition; the worker also checks
            # checkpoint capability and target binding on every fresh reset.
            from lab_low.skills import condition, validate_targets
            from lab_low.live_marks import load_live_marks
            data = (recorded_dataset.task(self.selection) if recorded_dataset is not None
                    else load_live_marks(marks))
            condition(data["skill"], data["hand"])
            validate_targets(data["skill"], data.get("targets"))
            scripts = Path(__file__).resolve().parents[1] / "scripts"
            if recorded_dataset is None:
                print(f"[推理目标] 读取 {marks}\n  技能={data['skill']} 手别={data['hand']} 初始框={data['targets']}\n"
                      f"  绑定原始参考图={'是' if data.get('reference_images') else '否（无框操作）'}；后续观测使用实时三路相机。", flush=True)
                command = [sys.executable, str(scripts / "run_g1_lab_client.py"),
                           "--marks", str(marks), "--camera-host", camera_host, "--camera-port", str(camera_port),
                           "--denoise-steps",str(denoise_steps)]
            else:
                command = [sys.executable, str(scripts / "run_g1_recorded_client.py"),
                           "--dataset", str(recorded_dataset.root), "--episode", str(self.selection["episode"]),
                           "--start-frame", str(self.selection["frame"])]
            command.extend(["--sonic-root", str(sonic_root), "--policy-url", policy_url,
                            "--operator-epoch", str(self.machine.epoch)])
            if recorded_dataset is not None:
                command.extend(["--inference-mode",inference_mode,"--inference-horizon",str(inference_horizon),
                    "--inference-delay-steps",str(inference_delay_steps),"--denoise-steps",str(denoise_steps),
                    "--chunk-smoothing-steps",str(chunk_smoothing_steps)])
            if chunks is not None:
                command.extend(["--chunks", str(chunks)])
            if execute:
                command.append("--execute")
            self.client_log = log_dir/f"inference_{self.machine.epoch}_{time.time_ns()}.log"
            print(f"[推理记录] {self.client_log}", flush=True)
            with self.client_log.open("x") as log:
                self.client = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)

        def choose(self):
            selection = catalog.parse(self.menu_input)
            if recorded_dataset is not None:
                recorded_dataset.task(selection)
            target = catalog.pose(selection)
            validate_pose(self.robot, target)
            catalog.save_selection(selection_path, selection)
            self.selection, self.target = selection, target
            self.menu_input = None
            if self.state == "ready":
                self.machine.epoch += 1
                self.machine.phase = "arm_initializing"
                self.begin_pose(time.monotonic())

        def begin_grounding(self):
            if live_grounding_config is None:
                return
            if self.state not in ("idle", "standing_ready", "ready", "damped"):
                raise ValueError("请先用 r 回到初始化状态，再按 b 重新标记")
            if self.grounding_process is not None:
                raise ValueError("正在读取相机/等待 Qwen；d 仍可立即急停")
            self.grounding_ready = False
            self.grounding_log = log_dir / f"grounding_{time.time_ns()}.log"
            script = Path(__file__).resolve().parents[1] / "scripts/prepare_live_marks.py"
            with self.grounding_log.open("x") as log:
                self.grounding_process = subprocess.Popen([sys.executable, str(script),
                    "--config", str(live_grounding_config), "--output", str(marks)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            print(f"[实时目标] 正在捕获三路图像并标记；等待期间 d 仍可急停。日志：{self.grounding_log}", flush=True)

        def key(self, key, now):
            if key == "b" and self.menu_input is None and live_grounding_config is not None:
                self.begin_grounding()
                return
            if key == "p" and self.state in ("idle", "standing_ready", "ready", "damped"):
                self.menu_input = ""
                if recorded_dataset is None:
                    print(catalog.menu(), flush=True)
                else:
                    print(recorded_dataset.menu(catalog.skills) + "\n输入：动作编号 轨迹编号 起始帧；Esc 取消。", flush=True)
                return
            if key == "r":
                self.menu_input = None
            if self.menu_input is not None:
                if key == "enter":
                    self.choose()
                elif key in ("esc", "escape"):
                    self.menu_input = None
                elif key == "backspace":
                    self.menu_input = self.menu_input[:-1]
                elif key == "space" or key.isdecimal():
                    self.menu_input += " " if key == "space" else key
                if self.menu_input is not None:
                    print(f"\r[Pose selection] {self.menu_input}   ", end="", flush=True)
                return
            if key == "r" and self.takeover_thread is not None and self.takeover_thread.is_alive():
                raise ValueError("Takeover RPC is still exiting; wait before r")
            before = self.state
            if self.machine.key(key):
                if self.state == "standing":
                    self.begin_stand(now)
                elif self.state == "arm_initializing":
                    if key == "r":
                        # Revoke the inference epoch before returning the arms;
                        # do not reset the active lower-body controller.
                        self.stop_client()
                        self.last_fault = None
                        if self.stand_started is None:
                            self.machine.phase, self.machine.recovering = "standing", True
                            self.begin_stand(now)
                            return
                    self.begin_pose(now)
                elif self.state == "running":
                    try:
                        self.start_client()
                        self.policy.lab_buffer.stop()
                    except Exception:
                        self.machine.phase = before
                        raise

        def update(self, goal, now):
            link = self.env.body().body_command_sender.onboard_link
            if link is not None and link.disconnected:
                self.enter_damping("connection lost")
                goal.clear()
                return
            if self.shutdown.estop_requested.is_set() or goal.get("emergency_stop"):
                self.enter_damping("operator emergency stop")
                self.shutdown.estop_requested.clear()
                goal.clear()
                return
            for child, started in list(self.retiring):
                if child.poll() is not None:
                    self.retiring.remove((child, started))
                elif now-started > 5:
                    child.kill()
            if self.grounding_process is not None and self.grounding_process.poll() is not None:
                self.grounding_ready = self.grounding_process.returncode == 0
                self.grounding_process = None
                if self.grounding_ready:
                    print(f"[实时目标已就绪] 初始框已绑定原始图；到 ready 后可按 Enter 推理。日志：{self.grounding_log}", flush=True)
                    try:
                        summary = json.loads(self.grounding_log.read_text().splitlines()[-1])
                        print(f"  目标：{summary['targets']}\n  预览：{summary['preview']}", flush=True)
                    except (OSError, ValueError, IndexError, KeyError):
                        pass
                else:
                    print(f"[实时目标失败] 未采用旧框；修正原因后按 b 重试。日志：{self.grounding_log}", flush=True)
            while not self.events.empty():
                try:
                    self.key(self.events.get(), now)
                except (ValueError, KeyError, IndexError, OSError) as error:
                    if self.state == "arm_initializing":
                        self.fail(error)
                    else:
                        print(f"[操作未执行] {error}", flush=True)
            if not self.takeover_result.empty():
                epoch, error = self.takeover_result.get()
                if epoch == self.machine.epoch and self.state == "standing":
                    if error:
                        self.enter_damping(error)
                    else:
                        self.env.restart_initialization()
                        self.stand_started = now
            try:
                if self.state == "standing" and self.stand_started is not None:
                    self.machine.tick(stand_ready=self.env.safety_monitor.startup_complete)
                    if self.state == "arm_initializing":
                        self.begin_pose(now)
                if self.state in ("arm_initializing", "ready"):
                    ready = self.pose_tick(now)
                    self.machine.tick(pose_ready=ready)
            except (ValueError, RuntimeError) as error:
                self.fail(error)
            if self.client is not None and self.client.poll() is not None:
                result = self.client.returncode
                self.client = None
                self.policy.lab_buffer.stop()
                self.machine.phase = "paused" if result == 0 else "fault"
                self.last_fault = None if result == 0 else f"推理客户端退出 {result}；日志：{self.client_log}"
                print(f"[推理已停止] 客户端退出码 {result}；r 返回初始姿态。", flush=True)
            standing = self.state not in ("idle", "damped") and self.stand_started is not None
            self.policy.lower_body_policy.use_policy_action = standing
            self.policy.lower_body_policy.cmd[:] = 0.
            self.env.operator_phase = self.state if standing or self.state in ("idle", "damped") else "taking_control"
            self.env.operator_arms_enabled = self.arms_engaged
            self.env.lab_operator_epoch = self.machine.epoch
            # An old client must never re-arm after d/r, even if its delayed
            # packet arrives after a new inference epoch has started.
            if self.state != "running" or goal.get("lab_operator_epoch") != self.machine.epoch:
                goal.clear()
            self.notice()

        def status(self):
            return dict(control_mode=self.state, lab_operator_epoch=self.machine.epoch,
                        selected_initial_pose=self.selection, initialization_epoch=self.machine.epoch,
                        connection_latched=False, initialization_only=False)

        def close(self):
            self.takeover_cancel.set()
            self.stop_client()
            if self.grounding_process is not None:
                self.grounding_process.terminate()
                try:
                    self.grounding_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.grounding_process.kill()
                    self.grounding_process.wait(timeout=2)
            for child, _ in self.retiring:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)

    class SessionShutdown(base_shutdown):
        def finish(self):
            self.env.enter_damping()
            if "mode" in holder:
                holder["mode"].close()
            try:
                return super().finish()
            finally:
                on_close()

    loop.G1Env, loop.CollectionMode, loop.DampingShutdown = SessionEnv, SessionMode, SessionShutdown
    return holder
