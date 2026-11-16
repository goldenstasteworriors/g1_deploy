"""Fresh single-GPU experiments using only project assets below /share.

Run with the /share environment's Python. --check validates imports, selected
data files, and the exact Hydra configuration without constructing a model.
"""
import argparse
from datetime import datetime
import fcntl
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT.parent
EXPERIMENTS = ('general', 'take_tube', 'dispense', 'take_dropper',
               'return_dropper', 'water_bath', 'take_tube_light', 'return_tube')
AUGMENTATION_EXPERIMENTS = {
    'take_tube_local_light': 'take_tube',
    'take_tube_box_robust': 'take_tube',
    'general_light_balanced': 'general',
    'take_tube_camera': 'take_tube',
}
EXPERIMENTS += tuple(AUGMENTATION_EXPERIMENTS)


def shared(path):
    path = Path(path).resolve()
    if not path.is_relative_to('/share'):
        raise ValueError(f'Project dependency must resolve below /share: {path}')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment', choices=EXPERIMENTS)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.gpu < 0:
        ap.error('--gpu must be nonnegative')
    env_root = shared(DEPLOY / 'envs/openwam')
    python = shared(env_root / 'bin/python')
    if Path(sys.executable).resolve() != python:
        raise RuntimeError(f'Invoke this script with {env_root}/bin/python')
    shared(ROOT)
    foundation = shared(DEPLOY / 'assets/OpenWAM-Alpha-Pretrain-Foundation-Model')
    manifest_name = AUGMENTATION_EXPERIMENTS.get(args.experiment, args.experiment)
    manifest = shared(ROOT / f'data/boxed_v3_suite/{manifest_name}.json')
    for p in (foundation / 'checkpoint_step_154000.safetensors', foundation / 'config.yaml',
              foundation / 'tokenizer', manifest):
        if not shared(p).exists():
            raise FileNotFoundError(p)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'_{os.getpid()}'
    run = shared(ROOT / 'outputs/share_runs' / args.experiment / stamp)
    run.mkdir(parents=True, exist_ok=False)
    cache = run / 'cache'
    environment = dict(os.environ)
    environment.pop('PYTHONHOME', None)
    environment.update(PATH=f'{env_root}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
                       PYTHONPATH=str(ROOT), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
                       CUDA_VISIBLE_DEVICES='' if args.check else str(args.gpu),
                       TOKENIZERS_PARALLELISM='false', HYDRA_FULL_ERROR='1', OMP_NUM_THREADS='4',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
    for key, subdir in dict(TMPDIR='tmp', XDG_CACHE_HOME='xdg', HF_HOME='huggingface',
                           TORCH_HOME='torch', TORCH_EXTENSIONS_DIR='extensions',
                           TRITON_CACHE_DIR='triton', TORCHINDUCTOR_CACHE_DIR='inductor',
                           CUDA_CACHE_PATH='cuda', MPLCONFIGDIR='matplotlib',
                           NUMBA_CACHE_DIR='numba', WANDB_DIR='wandb').items():
        path = cache / subdir
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    overrides = [
        f'model.architecture.foundation_checkpoint={foundation}/checkpoint_step_154000.safetensors',
        f'training.finetune_ckpt_path={foundation}',
        f'model.video_backbone.model_path={foundation}',
        f'dataloader.manifest={manifest}', f'dataloader.artifact_dir={cache}/data',
        'training.max_steps=30000', 'training.save_steps=1000',
        'validation_every=1000', 'validation_windows=10',
        f'training.output_path={run}/weights', f'hydra.run.dir={run}/hydra',
        f'project.wandb.run_name={stamp}_{args.experiment}',
        f'dataloader.photometric.enabled={str(args.experiment == "take_tube_light" or args.experiment in AUGMENTATION_EXPERIMENTS).lower()}',
    ]
    if args.experiment in AUGMENTATION_EXPERIMENTS:
        overrides.append(f'+experiment={args.experiment}')
    command = [str(python), '-m', 'torch.distributed.run', '--nproc_per_node=1', '--nnodes=1',
               '--master_addr=127.0.0.1', f'--master_port={29700 + args.gpu}',
               'scripts/train_lab.py', *overrides]
    (run / 'launch.json').write_text(json.dumps(dict(command=command, experiment=args.experiment,
        gpu=args.gpu, check_only=args.check, environment={k: environment[k] for k in environment
        if k in ('PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TMPDIR',
                 'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH')}), indent=2) + '\n')
    print(f'Run directory: {run}', flush=True)
    print(f'Log: {run}/train.log', flush=True)
    if args.check:
        # Child imports use the same isolated path/cache settings as training.
        check_code = (
            'import importlib,json; from pathlib import Path; '
            'names=("torch","hydra","accelerate","deepspeed","cv2","pyarrow","transformers","safetensors"); '
            'paths={n:str(Path(importlib.import_module(n).__file__).resolve()) for n in names}; '
            'assert all(Path(p).is_relative_to("/share") for p in paths.values()), paths; print(json.dumps(paths))'
        )
        with (run / 'imports.log').open('w') as log:
            subprocess.run([str(python), '-c', check_code], env=environment, cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = json.loads(manifest.read_text())['segments']
        for row in rows:
            dataset = shared(row['dataset'])
            info = json.loads((dataset / 'meta/info.json').read_text())
            ep = row['episode_index']
            fields = dict(episode_index=ep, episode_chunk=ep // info['chunks_size'])
            paths = [dataset / info['data_path'].format(**fields)]
            paths += [dataset / info['video_path'].format(video_key=c, **fields) for c in
                      ('observation.images.ego_view', 'observation.images.left_wrist', 'observation.images.right_wrist')]
            for path in paths:
                if not shared(path).is_file():
                    raise FileNotFoundError(path)
        with (run / 'resolved_config.yaml').open('w') as log:
            subprocess.run([str(python), 'scripts/train_lab.py', *overrides, '--cfg', 'job', '--resolve'],
                           env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'CHECK PASSED: {args.experiment}; {len(rows)} segments; no GPU model/training started', flush=True)
        return
    locks = ROOT / '.cache/share_gpu_locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f'{socket.gethostname()}_gpu{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free = int(subprocess.check_output(['nvidia-smi', f'--id={args.gpu}',
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < 76000:
            raise RuntimeError(f'GPU {args.gpu}: only {free} MiB free; need >=76000 MiB')
        with (run / 'train.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        readbacks = list((run / 'weights').glob('*/checkpoint_readback.json'))
        if len(readbacks) != 1:
            raise RuntimeError('Expected exactly one final checkpoint readback')
        row = json.loads(readbacks[0].read_text())
        if not row['exact_match'] or not row['checkpoint'].endswith('checkpoint_step_30000.safetensors'):
            raise RuntimeError('Missing verified 30000-step checkpoint')
        (run / 'completed.json').write_text(json.dumps(row, indent=2) + '\n')
        print(f'COMPLETED: {run}', flush=True)


if __name__ == '__main__':
    main()
