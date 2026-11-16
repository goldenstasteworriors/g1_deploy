"""Select a dataset episode/frame before the existing three-Enter G1 operator."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_low.initial_poses import PoseCatalog
from lab_low.recorded_vision import RecordedDataset, RecordedEpisode


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Other options pass to run_g1_operator.py, including --execute and --check-config.")
    parser.add_argument("--deploy-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", default="general")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--list", action="store_true", help="List supported episodes without robot I/O")
    args, rest = parser.parse_known_args()
    root = args.deploy_root.resolve()
    dataset = args.dataset or root / "assets/datasets/lab_primitives_train_20261110_boxed_v3"
    capabilities = root / "runtime/models" / args.model / "supported_skill_hands.json"
    source = RecordedDataset(dataset, json.loads(capabilities.read_text()))
    if args.list:
        print(source.menu())
        return
    episode, frame = args.episode, args.frame
    if episode is None:
        print(source.menu())
        values = input("输入轨迹编号和起始帧（例如 0 0），然后按 Enter：").split()
        if len(values) != 2:
            parser.error("Enter exactly two integers: episode frame")
        episode, frame = map(int, values)
    selection = source.selection(episode, frame)
    catalog_path = root / "assets/initial_poses/catalog.json"
    catalog = PoseCatalog(catalog_path)
    if catalog.meta["source_dataset"] != Path(dataset).name:
        raise ValueError("Pose catalog belongs to a different dataset")
    catalog.pose(selection)
    if catalog.entries[(selection["skill"], episode)]["frames"] != source.entries[episode]["frames"]:
        raise ValueError("Pose catalog and recorded video lengths differ")
    video = RecordedEpisode(source, episode, frame)
    try:
        task = video.task
        print(f"[数据集视觉] 轨迹 {episode} / 起始帧 {frame} / 标注参考帧 {task['reference_frame']}", flush=True)
        print(f"[任务] {task['skill']} / {task['hand']}\n[Prompt] {task['prompt']}", flush=True)
        asynchronous = '--inference-mode' in rest and rest[rest.index('--inference-mode')+1] == 'async'
        print("[输入] 状态来自实机；三路图像和初始框来自数据集；" +
              ("异步预取，首次推理后视频按控制时间推进（含缓冲耗尽时的保持步）。" if asynchronous else "同步推理等待时视频暂停。"), flush=True)
    finally:
        video.close()
    selection_path = root / "runtime/selected_recorded_pose.json"
    catalog.save_selection(selection_path, selection)
    command = [sys.executable, str(ROOT / "scripts/run_g1_operator.py"),
        "--sonic-root", str(root / "sonicmj"), "--catalog", str(catalog_path),
        "--selection", str(selection_path), "--marks", str(root / "runtime/marks.json"),
        "--logs", str(root / "logs/recorded_operator"), "--model", args.model,
        "--recorded-dataset", str(dataset), "--recorded-capabilities", str(capabilities),
        "--interface", os.environ.get("LABWAM_G1_INTERFACE", "enp132s0"), "--start-policy", *rest]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
