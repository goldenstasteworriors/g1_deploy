"""Foreground, observable four-GPU launcher; failures are never silently restarted."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
JOBS = ('return_dropper', 'water_bath', 'take_tube_light', 'return_tube')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    out = ROOT/'outputs/additional_launches'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out.mkdir(parents=True, exist_ok=False)
    children, logs, state = {}, [], {}
    print('LAUNCH_DIRECTORY='+str(out), flush=True)
    try:
        for gpu, name in enumerate(JOBS):
            command = [sys.executable, str(ROOT/'scripts/train_share.py'), name, '--gpu', str(gpu)]
            if args.check:
                command.append('--check')
            log = (out/(name+'.log')).open('x'); logs.append(log)
            child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children[name] = child
            state[name] = dict(gpu=gpu,pid=child.pid,command=command,log=str(out/(name+'.log')))
        while True:
            for name, child in children.items():
                state[name]['returncode'] = child.poll()
            pending = out/'status.pending'
            pending.write_text(json.dumps(state,indent=2)+'\n');pending.replace(out/'status.json')
            if all(child.poll() is not None for child in children.values()):
                break
            time.sleep(10)
    except BaseException:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        raise
    finally:
        for log in logs:
            log.close()
    print(json.dumps(state),flush=True)
    return int(any(c.returncode for c in children.values()))


if __name__ == '__main__':
    sys.exit(main())
