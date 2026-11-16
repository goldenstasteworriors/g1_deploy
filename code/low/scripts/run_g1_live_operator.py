"""Real RGB/state operator with b-triggered initial Qwen grounding."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_low.initial_poses import PoseCatalog
from lab_low.live_grounding import OPERATIONS, prepare
from lab_low.skills import condition


def pickup_plan(skill, description):
    """Task-scoped single pickup; no fabricated completion receipts."""
    if not description or not description.isascii():
        raise ValueError("Use an English target description for Qwen")
    tube = skill == "pick_up_test_tube"
    if not tube and skill != "take_dropper_and_aspirate":
        raise ValueError("This skill needs --grounding-plan or --grounding-session; water bath is not supported by the current Qwen service")
    hand = "right" if tube else "left"
    target = "selected_tube" if tube else "selected_dropper"
    objects = {target: {"kind": "test_tube" if tube else "dropper"}}
    if tube:
        groups = {"rack": {"label": "the test tube rack", "items": [target]}}
        locations = {target: {"container": "rack", "position": 0}}
    else:
        objects[target]["source_id"] = "source_bottle"
        objects["source_bottle"] = {"kind": "reagent_bottle"}
        groups = {"table": {"label": "the laboratory table", "items": ["source_bottle"]}}
        locations = {target: {"container": "source_bottle", "position": 0},
                     "source_bottle": {"container": "table", "position": 0}}
    return dict(plan=dict(execution_allowed=False, protocol_id="operator_declared_single_pickup",
        state_spec=dict(authority="single_authoritative_ledger", fields=["hands", "locations", "reagent_stages", "gates"],
                        commit_rule="Only actual completed operation receipts may change task state."),
        initial_state=dict(hands=dict(left=None, right=None), locations=locations, reagent_stages={}, gates={}),
        steps=[dict(step_id=1, operation="TAKE_TUBE" if tube else "TAKE_DROPPER_AND_ASPIRATE", hand=hand,
                    instruction=condition(skill, hand)[2], target=dict(id=target, description=description), tools={}, destination=None)]),
        layout=dict(groups=groups, objects=objects, identity_anchors=dict(operator_selection=description,
                    task_scope="Only the selected object is enumerated. Its task-list index is not its physical rank. Other hand occupancy is not visually verified.")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deploy-root", type=Path, required=True)
    p.add_argument("--model", default="take_tube")
    p.add_argument("--skill")
    p.add_argument("--hand")
    p.add_argument("--episode", type=int, help="Initial pose only; inference images always come from live cameras")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--target-description", help="English identity/spatial description for a standalone pickup")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--grounding-plan", type=Path, help="English plan/layout describing the actual current task state")
    group.add_argument("--grounding-session", help="Existing high-level receipt-ledger session ID")
    p.add_argument("--grounding-url", default="http://127.0.0.1:19113")
    p.add_argument("--camera-host", default=os.environ.get("LABWAM_CAMERA_HOST", "192.168.123.164"))
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--check-inputs", action="store_true", help="Capture/ground only; do not open robot control or infer actions")
    args, rest = p.parse_known_args()
    root = args.deploy_root.resolve()
    pairs = json.loads((root / "runtime/models" / args.model / "supported_skill_hands.json").read_text())
    if args.skill is None:
        if len(pairs) != 1:
            p.error("Use --skill with a multi-skill checkpoint")
        args.skill = pairs[0][0]
    matching = [hand for skill, hand in pairs if skill == args.skill]
    hand = args.hand or (matching[0] if len(matching) == 1 else None)
    if [args.skill, hand] not in pairs:
        p.error("The checkpoint was not trained for this skill/hand")
    condition(args.skill, hand)
    config = dict(skill=args.skill, hand=hand, grounding_url=args.grounding_url,
                  camera_host=args.camera_host, camera_port=args.camera_port, sonic_root=str(root / "sonicmj"))
    if args.skill != "dispense_into_held_test_tube":
        if args.skill not in OPERATIONS.values():
            p.error("The current online Qwen service does not support this skill (including water bath)")
        if args.grounding_session:
            if not args.grounding_session.replace("_", "").replace("-", "").isalnum():
                p.error("Invalid grounding session ID")
            config["session_id"] = args.grounding_session
        elif args.grounding_plan:
            config["plan_packet"] = json.loads(args.grounding_plan.read_text())
        else:
            if args.target_description is None:
                p.error("Specify the actual target in English with --target-description, or provide a grounding plan/session")
            config["plan_packet"] = pickup_plan(args.skill, args.target_description)
        if "plan_packet" in config:
            packet = config["plan_packet"]
            if not json.dumps(packet, ensure_ascii=False).isascii() or packet["plan"].get("execution_allowed") is not False:
                p.error("Grounding plan/layout must be English and execution_allowed:false")
    catalog_path = root / "assets/initial_poses/catalog.json"
    catalog = PoseCatalog(catalog_path)
    entries = [e for e in catalog.meta["episodes"] if e["skill"] == args.skill]
    episode = args.episode if args.episode is not None else entries[0]["episode"]
    entry = next(e for e in catalog.meta["episodes"] if e["episode"] == episode)
    selection = dict(skill=entry["skill"], episode=episode, frame=args.frame)
    catalog.pose(selection)
    folder = root / "runtime/live_operator" / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    config_path, marks = folder / "grounding_config.json", folder / "marks.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    selection_path = folder / "selected_initial_pose.json"
    catalog.save_selection(selection_path, selection)
    print(f"[实时部署] 模型={args.model} 技能={args.skill} 手别={hand}\n"
          f"[初始姿态] episode={episode} frame={args.frame}；仅用于双臂/夹爪初始化。\n"
          f"[真实输入] ego＋双腕图像、实机状态；目标描述={args.target_description or args.grounding_session or args.grounding_plan or '无需框'}\n"
          f"[流程] Enter站立 → Enter双臂初始化 → 到ready后按b取图/标记 → 等目标就绪后Enter推理。\n"
          f"[按键] d阻尼；r恢复；p修改初始化姿态；b重新取图标记。\n[本次文件] {folder}", flush=True)
    if args.check_inputs:
        print(json.dumps(prepare(config, marks), ensure_ascii=False), flush=True)
        return
    if "--policy-url" not in rest:
        rest.extend(["--policy-url", "http://127.0.0.1:" + ("8849" if args.model == "take_tube" else "8850")])
    command = [sys.executable, str(ROOT / "scripts/run_g1_operator.py"),
        "--sonic-root", str(root / "sonicmj"), "--catalog", str(catalog_path),
        "--selection", str(selection_path), "--marks", str(marks),
        "--live-grounding-config", str(config_path), "--logs", str(root / "logs/live_operator"),
        "--model", args.model, "--camera-host", args.camera_host, "--camera-port", str(args.camera_port),
        "--interface", os.environ.get("LABWAM_G1_INTERFACE", "enp132s0"), "--start-policy", *rest]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
