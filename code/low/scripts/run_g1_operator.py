"""Interactive three-Enter G1 deployment, with optional managed local policy."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from lab_low.gravity import add_gravity_arguments, gravity_options, build_arm_gravity
    p = argparse.ArgumentParser(description=__doc__)
    add_gravity_arguments(p)
    p.add_argument("--sonic-root", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--marks", type=Path, required=True)
    p.add_argument("--live-grounding-config", type=Path, help="Enable b: capture live cameras and obtain an initial Qwen box")
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--policy-url")
    p.add_argument("--inference-mode",choices=["sync","async"],default="sync")
    p.add_argument("--inference-horizon",type=int,default=32)
    p.add_argument("--inference-delay-steps",type=int,default=16)
    p.add_argument("--denoise-steps",type=int,default=10)
    p.add_argument("--chunk-smoothing-steps",type=int,default=0)
    p.add_argument("--camera-host", default="127.0.0.1")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--recorded-dataset", type=Path)
    p.add_argument("--recorded-capabilities", type=Path)
    p.add_argument("--model", choices=["general", "take_tube", "dispense", "take_dropper", "return_dropper", "return_tube", "water_bath"], default="general")
    p.add_argument("--start-policy", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--check-config", action="store_true")
    p.add_argument("--chunks", type=int, default=None, help="Optional explicit chunk count; otherwise run until stopped")
    args, rest = p.parse_known_args()
    args.policy_url = args.policy_url or "http://127.0.0.1:8849"
    if not 1<=args.denoise_steps<=100 or not 1<=args.inference_horizon<=32 or not 0<=args.inference_delay_steps<args.inference_horizon:
        p.error("Invalid denoising/horizon/prefetch configuration")
    if args.inference_mode=="async" and args.recorded_dataset is None:
        p.error("The async adapter currently requires the recorded-image entry point")
    if not 0 <= args.chunk_smoothing_steps <= 32 or (args.chunk_smoothing_steps and args.inference_mode != "async"):
        p.error("chunk-smoothing-steps must be 0..32 and requires async mode")
    if args.chunks is not None and args.chunks < 1:
        p.error("chunks must be positive")
    if not 1 <= args.camera_port <= 65535:
        p.error("camera-port must be 1..65535")
    if os.environ.get("SONIC_ONBOARD_BODY_HOST"):
        p.error("This staged launcher uses direct DDS; unset SONIC_ONBOARD_BODY_HOST before starting")
    sys.path.insert(0, str(args.sonic_root.resolve()))
    sys.path.insert(0, str(args.sonic_root.resolve()/"external_dependencies/unitree_sdk2_python"))
    import tyro
    from lab_low.initial_poses import PoseCatalog
    from lab_low.operator_control import install_operator, validate_pose
    from decoupled_wbc.control.main.teleop import run_g1_control_loop as loop
    catalog = PoseCatalog(args.catalog)
    recorded = None
    if args.recorded_dataset is not None:
        from lab_low.recorded_vision import RecordedDataset
        if args.recorded_capabilities is None:
            p.error("--recorded-capabilities is required with --recorded-dataset")
        recorded = RecordedDataset(args.recorded_dataset, json.loads(args.recorded_capabilities.read_text()))
    config = tyro.cli(loop.ControlLoopConfig,
                      default=loop.ControlLoopConfig(enable_gravity_compensation=True), args=rest)
    config.with_hands, config.hand_type, config.enable_waist = True, "dex1", False
    if config.gravity_compensation_joints is None:
        config.gravity_compensation_joints = ["arms"]
    config.startup_t_pose = False
    config.diagnostic_read_only = not args.execute
    robot = loop.instantiate_g1_robot_model(waist_location="lower_body")
    actual_names = [robot.joint_names[i] for i in robot.get_body_actuated_joint_indices()]
    if actual_names != catalog.meta["body_joint_names"]:
        raise ValueError("Dataset body joint order differs from this robot")
    selection = catalog.load_selection(args.selection)
    if recorded is not None:
        recorded.task(selection)
    validate_pose(robot, catalog.pose(selection))
    gravity = build_arm_gravity(robot, config.load_wbc_yaml(), gravity_options(args))
    gravity_status = gravity.summary() if gravity is not None else {"enabled": False}
    if gravity is not None:
        from lab_low.gravity import XR_KP, XR_KD
        gravity_status.update(evaluation="desired_arm_pose; measured_waist_and_pelvis_IMU",
                              arm_pd="Unitree XR G1_29", arm_kp=XR_KP.tolist(), arm_kd=XR_KD.tolist())
    print("[启动] 双臂控制：" + ("XR 位置PD＋目标角重力前馈；肩肘80/3，腕40/1.5。待机和仅站立阶段关闭。" if gravity is not None else "重力前馈关闭，沿用原位置增益。"), flush=True)
    if gravity is not None:
        print("[启动] 已计入每侧名义 Dex1-1 载荷（默认550g及官方标准安装质心）；额外转接件/相机/持物仍需标定。", flush=True)
    service = None
    ready, stopped = threading.Event(), threading.Event()
    args.logs.mkdir(parents=True, exist_ok=True)
    (args.logs/"gravity_config.json").write_text(json.dumps(gravity_status, indent=2)+"\n")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def monitor():
        last_ready = None
        incompatible_reported = False
        while not stopped.is_set():
            try:
                with opener.open(args.policy_url.rstrip("/")+"/", timeout=2) as response:
                    status = json.load(response)
                compatible = recorded is None or "recorded_reference_v1" in status.get("protocol_features", [])
                if not compatible and not incompatible_reported:
                    print("[模型] 当前服务仍是旧版协议；请正常退出旧部署后重新启动，加载数据集参考图协议。", flush=True)
                    incompatible_reported = True
                settings_match = ('per_request_denoise_steps' in status.get('protocol_features',[]) or
                                  status.get('denoise_steps') == args.denoise_steps)
                if compatible and settings_match and status.get("ready") and status.get("alive") and f"/models/{args.model}/" in status.get("checkpoint", ""):
                    ready.set()
                else:
                    ready.clear()
                    if compatible and not settings_match and not incompatible_reported:
                        print("[模型] 旧服务不支持当前去噪配置，请重启模型服务后使用异步入口。",flush=True)
                        incompatible_reported=True
            except (OSError, ValueError):
                ready.clear()
            if ready.is_set() != last_ready:
                print("[模型] " + (f"{args.model} 已就绪；仍需手臂到位后按 Enter 才开始推理。" if ready.is_set()
                      else f"尚未就绪或连接不可用；不能开始推理。加载日志：{args.logs/'policy.log'}"), flush=True)
                last_ready = ready.is_set()
            stopped.wait(1)

    def close_service():
        stopped.set()
        if service is not None and service.poll() is None:
            service.send_signal(signal.SIGINT)
            try:
                service.wait(timeout=15)
            except subprocess.TimeoutExpired:
                service.terminate()
                try:
                    service.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    service.kill()
                    service.wait(timeout=5)

    install_operator(loop, catalog=catalog, selection_path=args.selection, marks=args.marks,
        sonic_root=args.sonic_root, policy_url=args.policy_url, log_dir=args.logs,
        execute=args.execute, chunks=args.chunks, policy_ready=ready.is_set, on_close=close_service,
        camera_host=args.camera_host, camera_port=args.camera_port, gravity_options=gravity_options(args),
        recorded_dataset=recorded,inference_mode=args.inference_mode,
        inference_horizon=args.inference_horizon,inference_delay_steps=args.inference_delay_steps,
        denoise_steps=args.denoise_steps,chunk_smoothing_steps=args.chunk_smoothing_steps,
        live_grounding_config=args.live_grounding_config)
    if args.check_config:
        policy = loop.get_wbc_policy("g1", robot, config.load_wbc_yaml())
        print(json.dumps(dict(check="operator configuration, catalog and ONNX", selection=selection,
                              control="three Enter / d / r / p", robot_io_opened=False,
                              camera_host=args.camera_host, camera_port=args.camera_port,
                              recorded_task=None if recorded is None else recorded.task(selection),
                              inference_mode=args.inference_mode,denoise_steps=args.denoise_steps,
                              chunk_smoothing_steps=args.chunk_smoothing_steps,
                              policy_url=args.policy_url,inference_horizon=args.inference_horizon,
                              inference_delay_steps=args.inference_delay_steps,arm_gravity=gravity_status)))
        return
    if args.start_policy:
        from urllib.parse import urlparse
        address = urlparse(args.policy_url)
        if address.scheme!='http' or address.hostname!='127.0.0.1' or not address.port or address.path not in ('','/'):
            p.error("--start-policy requires a local 127.0.0.1 HTTP URL with an explicit port")
        import socket
        with socket.socket() as probe:
            occupied = probe.connect_ex(("127.0.0.1", address.port)) == 0
        if not occupied:
            with (args.logs/"policy.log").open("w") as log:
                service = subprocess.Popen(["bash", str(ROOT/"scripts/start_pku_policy.sh"), args.model, str(args.denoise_steps)],
                    env=dict(os.environ, LABWAM_BIND="127.0.0.1", LABWAM_PORT=str(address.port)),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    threading.Thread(target=monitor, daemon=True).start()
    key_sequence = ("Enter 站立 → 等待 → Enter 双臂初始化 → ready 后按 b 实时取图/标记 → 目标就绪后 Enter 推理。"
                    if args.live_grounding_config is not None else
                    "Enter 站立 → 等待 → Enter 双臂初始化 → 等待 → Enter 推理。")
    print("[启动中] 正在连接机器人、等待夹爪反馈并加载下半身 ONNX。看到“阶段 0/3 待机”后再操作。\n"
          "  接下来的 calibrating hand 仅等待夹爪反馈，不执行电机校零。\n"
          f"  按键顺序：{key_sequence}d 为全身阻尼急停。", flush=True)
    try:
        loop.main(config)
    finally:
        close_service()


if __name__ == "__main__":
    main()
