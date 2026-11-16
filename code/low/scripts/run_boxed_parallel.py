"""One supervised single-GPU run; the old sequential queue only waits for it."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/boxed_v3_30k'
DEVICES = {'take_tube': (0, 29650), 'dispense': (1, 29651), 'take_dropper': (2, 29652)}


def verify(name):
    paths = sorted((OUT / name).glob('*/checkpoint_readback.json'))
    if not paths:
        raise RuntimeError('Missing final checkpoint readback')
    row = json.loads(paths[-1].read_text())
    if not row['exact_match'] or not row['checkpoint'].endswith('checkpoint_step_30000.safetensors'):
        raise RuntimeError('No verified 30000-step checkpoint')
    return row['checkpoint']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment', choices=DEVICES)
    ap.add_argument('--wait', action='store_true')
    args = ap.parse_args()
    name = args.experiment
    folder = OUT / 'parallel'
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / (name + '.json')
    if args.wait:
        while True:
            row = json.loads(status.read_text())
            if row['state'] == 'completed':
                verify(name)
                return
            if row['state'] == 'failed':
                raise RuntimeError(row)
            time.sleep(30)
    with (folder / (name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.exists() or (OUT / name).exists():
            raise RuntimeError('Existing run: inspect before restarting')
        gpu, port = DEVICES[name]
        def save(state, **extra):
            row = dict(experiment=name, gpu=gpu, port=port, steps=30000,
                       state=state, time=time.time(), pid=os.getpid(), **extra)
            tmp = status.with_suffix('.pending')
            tmp.write_text(json.dumps(row, indent=2) + '\n')
            tmp.replace(status)
            print(json.dumps(row), flush=True)
        save('running')
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MASTER_PORT=str(port), LAB_PARALLEL_OWNER='1')
            with (folder / (name + '.log')).open('x') as log:
                subprocess.run(['bash', 'scripts/train_boxed_experiment.sh', name], cwd=ROOT,
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            save('completed', checkpoint=verify(name))
        except BaseException as error:
            save('failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
