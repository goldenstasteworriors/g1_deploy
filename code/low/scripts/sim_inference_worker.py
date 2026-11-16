"""JSONL model worker: recorded three-camera RGB and caller's achieved sim state."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
import torch
from lab_low.runtime import OnlineConditionBuffer
from lab_low.dataset import CAMERAS
from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
from openwam.deploy.denoise_schedule import schedule_sync


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--steps', type=int, default=20)
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint)
    with contextlib.redirect_stdout(sys.stderr):
        _, model, cfg = build_architecture_from_ckpt_dir(str(checkpoint), weights_required=True)
        model.set_dtype_device(torch.bfloat16, torch.device('cuda:0'))
        model.eval().requires_grad_(False)
        model.normalizer = None
        model.init_training_schedulers(1000)
        schedule = schedule_sync(model.video_scheduler, model.action_scheduler, args.steps, 5.)
    stats = checkpoint.parent / 'normalization_stats.npy'
    rows = json.loads(Path(args.manifest).read_text())['segments']
    captures, buffer, previous = [], None, -1
    prompt_cache = {}
    print(json.dumps(dict(ready=True, checkpoint=str(checkpoint))), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request['op'] == 'reset':
                for cap in captures:
                    cap.release()
                row = next(r for r in rows if r['episode_index'] == request['episode'] and r['split'] == 'val')
                root = Path(row['dataset'])
                info = json.loads((root / 'meta/info.json').read_text())
                fields = dict(episode_index=row['episode_index'], episode_chunk=row['episode_index']//info['chunks_size'])
                captures = [cv2.VideoCapture(str(root/info['video_path'].format(video_key=c, **fields))) for c in CAMERAS]
                buffer = OnlineConditionBuffer(stats)
                previous = row['start_frame']
                buffer.reset(skill=row['skill'], hand=row['hand'], binding_id=row['binding_id'],
                    frame_index=previous, state=request['state'], images=read_images(captures, previous),
                    targets=row['targets'], annotation_dir=str(Path(args.manifest).parent))
                prompt_cache.clear()
                print(json.dumps(dict(reset=True, episode=row['episode_index'], skill=row['skill'])), flush=True)
                continue
            if request['op'] != 'predict' or buffer is None:
                raise ValueError('reset required before predict')
            for packet in request['observations']:
                t = packet['frame']
                if t != previous+1:
                    raise ValueError('Every 30 Hz simulated state must be supplied in order')
                buffer.observe(frame_index=t, state=packet['state'], images=read_images(captures, t))
                previous = t
            sample = buffer.sample()
            started = time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):
                condition = model.prepare_condition([sample])
                result = model.generate(schedule, sample['prompt'], first_frame_image=sample['first_frame_image'],
                    num_frames=9, action_num_frames=33, width=320, height=384, num_inference_steps=args.steps,
                    shift=5., seed=42+previous, decode_video=False,
                    proprio=sample['proprio'].to(model.device, model.dtype).unsqueeze(0),
                    active_action_mask=torch.as_tensor(buffer.active, device=model.device),
                    lab_condition=condition, prompt_embed_cache=prompt_cache)
            actions = result['actions']
            if hasattr(actions, 'detach'):
                actions = actions.detach().float().cpu().numpy()
            actions = buffer.decode_and_gate(actions)
            if actions.shape != (32,16) or not np.isfinite(actions).all():
                raise ValueError('Invalid predicted action chunk')
            print(json.dumps(dict(frame=previous, actions=actions.tolist(),
                inference_seconds=time.monotonic()-started)), flush=True)
    finally:
        for cap in captures:
            cap.release()


def read_images(captures, frame):
    result = []
    for cap in captures:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f'Failed to decode recorded camera frame {frame}')
        result.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    return result


if __name__ == '__main__':
    main()
