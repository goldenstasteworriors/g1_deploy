"""Preserve the original held-out source split; add three single skills and lighting ablation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_low.skills import condition, validate_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--suite', type=Path, default=ROOT/'data/boxed_v3_suite')
    args = ap.parse_args()
    original = json.loads((args.suite/'general.json').read_text())['segments']
    split = {r['source_group']: r['split'] for r in original}
    runtime_root = original[0]['dataset']
    selected = {
        'return_dropper': [r for r in original if r['skill'] == 'return_dropper'],
        'water_bath': [r for r in original if r['skill'] == 'place_test_tube_in_water_bath'],
        'take_tube_light': json.loads((args.suite/'take_tube.json').read_text())['segments'],
    }
    meta = args.dataset/'meta'
    source = json.loads((meta/'segment_provenance.json').read_text())
    marks = {b['episode_index']: b for b in json.loads((meta/'initial_ego_boxes.json').read_text())['episodes']}
    rack = []
    for r in source:
        if r['task'] != 'RETURN_TUBE_TO_RACK':
            continue
        ep = r['output_episode']; b = marks[ep]
        assert b['review_id'] == r['review_id'] and b['frame_index'] == 0 and b['task'] == r['task']
        assert b['source_dataset'] == r['dataset'] and b['source_episode'] == r['episode'] and b['source_frame_index'] == r['start_frame']
        skill, hand = 'return_test_tube_to_rack', 'right'
        condition(skill, hand)
        targets = [dict(t, confidence=t.get('confidence', b.get('prompt', {}).get('confidence', .5))) for t in b['targets']]
        validate_targets(skill, targets)
        for mark in targets:
            x1,y1,x2,y2 = mark['box']; w,h = b['image_size']
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        group = json.dumps([r['dataset'],r['episode']])
        # Existing source groups MUST retain their old assignment.
        assignment = split.get(group, 'val' if int(hashlib.sha256(('42:'+group).encode()).hexdigest()[:8],16)%100 < 15 else 'train')
        rack.append(dict(dataset=runtime_root,episode_index=ep,start_frame=0,skill=skill,hand=hand,
            targets=targets,task=r['task'],binding_id=r['review_id'],source_group=group,split=assignment,
            annotation_status=b.get('status'),annotation_user_approved=b.get('user_approved',False)))
    selected['return_tube'] = rack
    report = {}
    for name, rows in selected.items():
        counts = Counter(r['split'] for r in rows)
        assert counts['train'] and counts['val'], (name, counts)
        assert all(r['source_group'] not in split or r['split'] == split[r['source_group']] for r in rows)
        (args.suite/(name+'.json')).write_text(json.dumps(dict(segments=rows),indent=2)+'\n')
        report[name] = dict(counts)
    (args.suite/'additional_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
