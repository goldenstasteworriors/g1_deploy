"""Four comparable manipulation experiments with source-trajectory-held-out splits."""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_low.skills import condition, validate_targets

TASKS = {
    "TAKE_TUBE": ("pick_up_test_tube", "right"),
    "TAKE_DROPPER_AND_ASPIRATE": ("take_dropper_and_aspirate", "left"),
    "DISPENSE_TO_HELD_TUBE": ("dispense_into_held_test_tube", "both"),
    "RETURN_DROPPER": ("return_dropper", "left"),
    "PLACE_TUBE_IN_BATH": ("place_test_tube_in_water_bath", "right"),
}
EXPERIMENTS = {"general": None, "take_tube": "TAKE_TUBE", "dispense": "DISPENSE_TO_HELD_TUBE",
               "take_dropper": "TAKE_DROPPER_AND_ASPIRATE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--runtime-dataset", help="Optional remote root; source annotations are still validated locally")
    args = ap.parse_args()
    root = args.dataset.resolve()
    meta = root / "meta"
    source = json.loads((meta / "segment_provenance.json").read_text())
    boxes = json.loads((meta / "initial_ego_boxes.json").read_text())["episodes"]
    marks = {b["episode_index"]: b for b in boxes}
    assert len(marks) == len(boxes), "Duplicate annotation episode"
    records = []
    for r in source:
        if r["task"] not in TASKS:
            continue
        ep = r["output_episode"]
        b = marks[ep]
        assert b["review_id"] == r["review_id"] and b["task"] == r["task"]
        assert b["frame_index"] == 0
        if "source_episode" in b:
            assert b["source_episode"] == r["episode"]
            assert b["source_dataset"] == r["dataset"]
            assert b["source_frame_index"] == r["start_frame"]
        skill, hand = TASKS[r["task"]]
        condition(skill, hand)
        targets = [] if r["task"] == "DISPENSE_TO_HELD_TUBE" else [dict(t, confidence=t.get("confidence", b.get("prompt", {}).get("confidence", .5))) for t in b["targets"]]
        validate_targets(skill, targets)
        for target in targets:
            x1,y1,x2,y2 = target["box"]
            w,h = b["image_size"]
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        records.append(dict(dataset=args.runtime_dataset or str(root), episode_index=ep,
            start_frame=0, skill=skill, hand=hand, targets=targets, task=r["task"],
            binding_id=r["review_id"], source_group=json.dumps([r["dataset"],r["episode"]]),
            annotation_status=b.get("status"), annotation_user_approved=b.get("user_approved",False)))
    # Select a reproducible approximately 15% held-out set while retaining every
    # skill in both splits. Related skill cuts from one raw episode stay together.
    groups = sorted({r["source_group"] for r in records})
    total = Counter(r["task"] for r in records)
    rng = random.Random(42)
    best = None
    for _ in range(4000):
        selected = set(rng.sample(groups, max(1,round(len(groups)*.15))))
        counts = Counter(r["task"] for r in records if r["source_group"] in selected)
        if any(not 0 < counts[t] < total[t] for t in total):
            continue
        score = sum((counts[t]/total[t]-.15)**2 for t in total)
        if best is None or score < best[0]: best=(score,selected)
    if best is None: raise ValueError("Cannot build source-disjoint validation with all skills")
    for r in records: r["split"]="val" if r["source_group"] in best[1] else "train"
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(seed=42,steps_per_experiment=30000,source=str(root),
                annotation_sha256=hashlib.sha256((meta/'initial_ego_boxes.json').read_bytes()).hexdigest(),
                excluded=dict(Counter(r['task'] for r in source if r['task'] not in TASKS)), experiments={})
    for name,task in EXPERIMENTS.items():
        chosen=[r for r in records if task is None or r["task"]==task]
        (args.output/(name+'.json')).write_text(json.dumps(dict(segments=chosen),indent=2)+'\n')
        report['experiments'][name]={s:dict(Counter(r['task'] for r in chosen if r['split']==s)) for s in ('train','val')}
    (args.output/'suite_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
