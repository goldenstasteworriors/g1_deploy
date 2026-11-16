"""Supervisor-managed readiness, real GPU smoke checks, then four 30k runs."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEPLOY=ROOT.parent
DATA=DEPLOY/'data/lab_primitives_train_20261110_boxed_v3'
OUT=ROOT/'outputs/boxed_v3_readiness'
FOUNDATION_SHA='180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    def state(stage,**extra):
        row=dict(stage=stage,time=time.time(),**extra)
        tmp=OUT/'status.pending'; tmp.write_text(json.dumps(row,indent=2)+'\n'); tmp.replace(OUT/'status.json')
        print(json.dumps(row),flush=True)
    try:
        expected=json.loads((ROOT/'configs/boxed_suite_integrity.json').read_text())
        state('waiting_for_verified_data')
        marker=DATA/'.boxed_suite_ready.json'
        while not marker.exists(): time.sleep(30)
        ready=json.loads(marker.read_text())
        if ready['manifest_digest']!=expected['manifest_digest']: raise ValueError('Wrong dataset readiness marker')
        state('checking_foundation')
        h=hashlib.sha256()
        checkpoint=DEPLOY/'assets/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors'
        with checkpoint.open('rb') as f:
            while chunk:=f.read(16*1024*1024): h.update(chunk)
        if h.hexdigest()!=FOUNDATION_SHA: raise ValueError('Foundation checksum mismatch')
        subprocess.run([sys.executable,'scripts/prepare_boxed_suite.py','--dataset',str(DATA),
                        '--output','data/boxed_v3_suite'],cwd=ROOT,check=True)
        for name in ('general','dispense'):
            state('waiting_for_gpu',experiment=name)
            while True:
                free=int(subprocess.check_output(['nvidia-smi','--id='+os.environ.get('CUDA_VISIBLE_DEVICES','3'),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=76000: break
                time.sleep(30)
            state('gpu_smoke',experiment=name)
            with (OUT/(name+'_smoke.log')).open('w') as log:
                subprocess.run(['bash','scripts/train_boxed_experiment.sh',name,
                    'training.max_steps=3','training.save_steps=3','validation_every=3',
                    'validation_windows=5','training.output_path=outputs/boxed_v3_smoke/'+name],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('training_suite')
        subprocess.run([sys.executable,'scripts/run_boxed_suite.py'],cwd=ROOT,check=True)
        state('completed')
    except BaseException as error:
        state('failed',error=str(error)); raise


if __name__=='__main__': main()
