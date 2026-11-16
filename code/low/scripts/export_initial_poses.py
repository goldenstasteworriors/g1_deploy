"""Export a compact pose selector from LeRobot observation.state only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_low.skills import ARM_INDICES, SKILLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("Use a new output filename")
    info = json.loads((args.dataset / "meta/info.json").read_text())
    task_map = {}
    for line in (args.dataset / "meta/tasks.jsonl").read_text().splitlines():
        task = json.loads(line)
        for skill, (_, fixed, template) in SKILLS.items():
            if any(task["task"] == template.format(hand=hand) for hand in (fixed,) if fixed) or any(
                    task["task"] == template.format(hand=hand) for hand in ("left", "right") if fixed is None):
                task_map[task["task_index"]] = skill
    arrays, entries = {}, []
    for line in (args.dataset / "meta/episodes.jsonl").read_text().splitlines():
        episode = json.loads(line)["episode_index"]
        path = args.dataset / info["data_path"].format(episode_index=episode, episode_chunk=episode // info["chunks_size"])
        table = pq.read_table(path, columns=["observation.state", "task_index"])
        tasks = set(table["task_index"].to_pylist())
        if len(tasks) != 1 or next(iter(tasks)) not in task_map:
            continue
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 31 or not np.isfinite(states).all():
            raise ValueError(f"Invalid achieved states in episode {episode}")
        key = f"episode_{episode:06d}"
        arrays[key] = states[:, ARM_INDICES]
        entries.append(dict(key=key, episode=episode, skill=task_map[next(iter(tasks))], frames=len(states)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    array_path = args.output.with_suffix(".npz")
    np.savez_compressed(array_path, **arrays)
    metadata = dict(source_dataset=args.dataset.name, source_column="observation.state", fps=info["fps"],
                    body_joint_names=info["features"]["observation.state"]["names"][:29],
                    action_indices=ARM_INDICES, array_file=array_path.name,
                    sha256=hashlib.sha256(array_path.read_bytes()).hexdigest(), episodes=entries)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(episodes=len(entries), frames=sum(e["frames"] for e in entries), bytes=array_path.stat().st_size)))


if __name__ == "__main__":
    main()
