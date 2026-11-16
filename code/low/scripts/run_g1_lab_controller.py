"""Run SONICMJ's controller with an arm-only LabWAM adapter.

Use the SONICMJ decoupled_vla_collection environment, on the robot controller
host. This launcher preserves the existing startup, mode control and e-stop.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from lab_low.gravity import add_gravity_arguments, gravity_options, build_arm_gravity
    p = argparse.ArgumentParser(description=__doc__)
    add_gravity_arguments(p)
    p.add_argument("--sonic-root", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="Allow the existing controller to write actuators")
    p.add_argument("--trace-dir", type=Path, help="Defaults to deployment logs/controller")
    p.add_argument("--check-imports", action="store_true", help="Import and configure the adapter without opening robot I/O")
    p.add_argument("--check-config", action="store_true", help="Load kinematics and WBC ONNX with the requested settings; no robot I/O")
    args, remaining = p.parse_known_args()
    root = args.sonic_root.resolve()
    required = root / "decoupled_wbc/control/envs/g1/g1_dex1_hand.py"
    if not required.is_file():
        raise FileNotFoundError("Use the wired-PICO Dex1 SONICMJ source snapshot")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "external_dependencies/unitree_sdk2_python"))
    import tyro
    from decoupled_wbc.control.main.teleop import run_g1_control_loop as loop
    from lab_low.g1_bridge import install_decoupled_adapter
    if args.check_imports:
        install_decoupled_adapter(loop)
        print("CONTROLLER_IMPORTS_OK; no ROS/DDS initialized")
        return
    config = tyro.cli(loop.ControlLoopConfig,
                      default=loop.ControlLoopConfig(enable_gravity_compensation=True), args=remaining)
    config.with_hands, config.hand_type, config.enable_waist = True, "dex1", False
    if config.gravity_compensation_joints is None:
        config.gravity_compensation_joints = ["arms"]
    config.diagnostic_read_only = not args.execute
    install_decoupled_adapter(loop, gravity_options=gravity_options(args),
                              trace_dir=args.trace_dir or root.parent/"logs/controller")
    if args.check_config:
        robot = loop.instantiate_g1_robot_model(waist_location="lower_body")
        wbc_config = config.load_wbc_yaml()
        loop.get_wbc_policy("g1", robot, wbc_config)
        gravity = build_arm_gravity(robot, wbc_config, gravity_options(args))
        print(f"ARM_GRAVITY: {gravity.summary() if gravity is not None else {'enabled': False}}")
        print(f"CONTROLLER_CONFIG_OK; interface={config.interface}, arm_speed={config.upper_body_joint_speed}; no ROS/DDS initialized")
        return
    loop.main(config)


if __name__ == "__main__":
    main()
