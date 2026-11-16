"""Observable, foreground single-GPU queue; failures halt subsequent experiments."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/boxed_v3_30k"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status_path = OUT / "queue_status.json"
    if status_path.exists():
        raise RuntimeError("Queue already has a status file; inspect it before starting another run")
    status = dict(started=datetime.now(timezone.utc).isoformat(), state="running", completed=[],
                  pending=["general", "take_tube", "dispense", "take_dropper"], steps_each=30000)
    def save():
        tmp=status_path.with_suffix('.json.pending')
        tmp.write_text(json.dumps(status,indent=2)+'\n'); tmp.replace(status_path)
    save()
    try:
        while status['pending']:
            name=status['pending'][0]; status['active']=name; save()
            with (OUT/(name+'.log')).open('x') as log:
                result=subprocess.run(['bash','scripts/train_boxed_experiment.sh',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'{name} failed with exit {result.returncode}; see log')
            runs=sorted((OUT/name).glob('*/checkpoint_readback.json'))
            if not runs: raise RuntimeError(f'{name}: missing verified final checkpoint')
            readback=json.loads(runs[-1].read_text())
            if not readback['exact_match'] or not readback['checkpoint'].endswith('checkpoint_step_30000.safetensors'):
                raise RuntimeError(f'{name}: incomplete 30000-step run')
            status['completed'].append(name); status['pending'].pop(0); save()
        status.update(state='completed',active=None); save()
    except BaseException as error:
        status.update(state='failed',error=str(error)); save(); raise


if __name__=='__main__': main()
