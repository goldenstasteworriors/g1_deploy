"""Convert high/examples/.../prompts_v1.json to one-initial-box skill segments."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_low.skills import SKILLS, HANDS, condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skill", choices=tuple(SKILLS), default="pick_up_test_tube")
    parser.add_argument("--hand", choices=tuple(HANDS), default="right")
    args = parser.parse_args()
    condition(args.skill, args.hand)
    source = json.loads(Path(args.prompts).read_text())
    rows = []
    for episode in source["episodes"]:
        if episode.get("original_discarded", False):
            continue
        rows.append(dict(dataset=args.dataset, episode_index=episode["episode_index"],
            start_frame=episode["start_frame"], skill=args.skill, hand=args.hand,
            binding_id=f"test_tube_{episode['episode_index']}",
            targets=[dict(role=0, camera="observation.images.ego_view", box=episode["box"], confidence=.5)],
            annotation_source=episode.get("prompt_source", "unspecified")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(format_version=1, target_mode="initial_reference_only",
        points_used=False, source_prompts=str(Path(args.prompts).resolve()), segments=rows), indent=2)+"\n")
    print(f"Wrote {len(rows)} segments to {output}. Initial boxes only; no points or future tracking labels.")


if __name__ == "__main__":
    main()
