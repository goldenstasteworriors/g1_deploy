"""Package completed boxed-v3 deltas; publish privately using a token on stdin.

Never reads Zotero or stores credentials. Only explicitly selected completed runs
are included. Large files are hard-linked on /share without chmod or mutation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('general', 'take_tube', 'dispense', 'take_dropper')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def prepare(bundle):
    from safetensors import safe_open
    bundle.mkdir(parents=True, exist_ok=False)
    entries = []
    for name in NAMES:
        checks = list((ROOT/'outputs/boxed_v3_30k'/name).glob('*/checkpoint_readback.json'))
        if len(checks) != 1:
            raise ValueError(f'Expected one completed {name} run')
        run = checks[0].parent
        readback = json.loads(checks[0].read_text())
        assert readback['exact_match'] and readback['checkpoint'].endswith('checkpoint_step_30000.safetensors')
        target = bundle/'models'/name; target.mkdir(parents=True)
        weights = [run/'checkpoint_step_30000.safetensors']
        if (run/'checkpoint_best.safetensors').exists():
            weights.append(run/'checkpoint_best.safetensors')
        for src in weights:
            with safe_open(str(src), framework='pt', device='cpu') as f:
                meta = f.metadata()
                assert meta['lab_checkpoint'] == 'delta_v1'
                assert meta['conditioner_version'] == 'spacetime_manip_v3'
                assert len(f.keys()) == readback['tensors']
                assert set(f.keys()) == set(json.loads(meta['tensor_names']))
            dst = target/src.name
            try:
                os.link(src,dst)
            except OSError:
                shutil.copyfile(src,dst)
            print('PACKED '+name+'/'+src.name,flush=True)
        for filename in ('config.yaml','normalization_stats.npy','best_metric.json',
                         'checkpoint_readback.json','data_contract.json','transfer_report.json',
                         'gradient_checks.json','validation.jsonl'):
            shutil.copyfile(run/filename,target/filename)
        shutil.copytree(run/'tokenizer',target/'tokenizer')
        entries.append(dict(name=name,source_run=str(run.relative_to(ROOT)),
            final_step=30000,best=json.loads((run/'best_metric.json').read_text())))
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(bundle/'lab-low-code.tar.gz'),'HEAD'],cwd=ROOT,check=True)
    shutil.copyfile(ROOT/'scripts/configure_downloaded_foundation.py',bundle/'configure_foundation.py')
    shutil.copyfile(ROOT/'LICENSE',bundle/'LICENSE')
    (bundle/'configuration.json').write_text(json.dumps(dict(framework='pytorch'),indent=2)+'\n')
    (bundle/'README.md').write_text('''---
license: apache-2.0
tags:
- robotics
- openwam
- labwam
---
# LabWAM Boxed-v3: completed 30k manipulation models

Private backup of four independently trained OpenWAM-based spacetime_manip_v3 models:
general, take_tube, dispense, take_dropper. All final checkpoints completed 30,000
steps and were read back tensor-by-tensor against live trained weights. Each model
also includes its best held-out flow-loss checkpoint if distinct from the final.
Best validation loss is not a manipulation success rate.

These are **delta checkpoints, not standalone Foundation models**. They require:
https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model
including checkpoint_step_154000.safetensors. This dependency follows the official
OpenWAM finetuning setup:
https://github.com/OpenWAM-Official/OpenWAM/blob/main/assets/openwam_usage_docs/openwam-alpha-finetuning.md

## Contents and behavior

- models/<name>/: final/best delta, config, normalization statistics, tokenizer,
  training contract, transfer/readback reports, validation losses.
- lab-low-code.tar.gz: matching project source snapshot; LICENSE retained.
- MANIFEST.json: source code revision, checkpoint selection and SHA-256 checksums.
- No dataset images, raw trajectories, API credentials or unfinished training runs.
- 31D observed state; 16D arm/gripper targets; no locomotion.
- Skill/hand conditioning, one initial target mark and causal visual/state history.
- Dispense uses no target box. Supervision is future observation.state.
- These four runs have no photometric augmentation. New augmented training is separate.

## Restore

Download the repository with ModelScope authentication, unpack lab-low-code.tar.gz
into your project, and use the project's OpenWAM environment/pyproject dependencies.
Download the Foundation separately. Then run the following single-line command,
replacing both paths with your local absolute directories:

```bash
python /path/to/bundle/configure_foundation.py --bundle /path/to/bundle --foundation /path/to/OpenWAM-Alpha-Pretrain-Foundation-Model
```

This updates downloaded configs only, retaining config.original.yaml backups.
Use an explicit models/<name>/checkpoint_step_30000.safetensors path for final
evaluation, or checkpoint_best.safetensors when present. Load with the bundled
build_architecture_from_ckpt_dir utility and OnlineConditionBuffer using that
model's normalization_stats.npy. Frozen video weights are restored from Foundation.
Keep model.normalizer=None when using OnlineConditionBuffer to avoid double normalization.

Training configs and diagnostic reports retain original provenance paths; these
paths are records, not a requirement to recreate the original filesystem. The
included configure_foundation.py adjusts the inference Foundation dependency.
This backup does not claim successful closed-loop robot operation.
''')
    files = {}
    for path in sorted(bundle.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(bundle))] = dict(bytes=path.stat().st_size,sha256=sha256(path))
    manifest = dict(code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    models=entries,foundation_repo='OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model',files=files)
    (bundle/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(files=len(files)+1,bytes=sum(x['bytes'] for x in files.values()),bundle=str(bundle))),flush=True)


def add_foundation(bundle):
    """Extend our existing export without touching original model assets."""
    manifest_path = bundle/'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    source = ROOT.parent/'assets/OpenWAM-Alpha-Pretrain-Foundation-Model'
    target = bundle/'foundation'
    paths = [source/'checkpoint_step_154000.safetensors', source/'config.yaml', source/'README.md']
    paths += sorted((source/'tokenizer').rglob('*'))
    for src in paths:
        if src.is_dir():
            continue
        dst = target/src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            if src.suffix == '.safetensors':
                os.link(src,dst)
            else:
                shutil.copyfile(src,dst)
        print('HASHING '+str(dst.relative_to(bundle)),flush=True)
        manifest['files'][str(dst.relative_to(bundle))] = dict(bytes=dst.stat().st_size,sha256=sha256(dst))
    required = target/'checkpoint_step_154000.safetensors'
    from safetensors import safe_open
    with safe_open(str(required),framework='pt',device='cpu') as f:
        names = list(f.keys())
        assert any(k.startswith('video_backbone.dit.') for k in names)
        assert any(k.startswith('video_backbone.vae.') for k in names)
        assert any(k.startswith('video_backbone.text_encoder.') for k in names)
        manifest['foundation_tensor_count'] = len(names)
    manifest['foundation_directory'] = 'foundation'
    manifest['foundation_included'] = True
    manifest['previous_code_revision'] = manifest['code_revision']
    manifest['code_revision'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    shutil.copyfile(ROOT/'scripts/configure_downloaded_foundation.py',bundle/'configure_foundation.py')
    subprocess.run(['git','archive','--format=tar.gz','--output='+str(bundle/'lab-low-code.tar.gz'),'HEAD'],cwd=ROOT,check=True)
    readme = (bundle/'README.md').read_text()
    readme = readme.replace('These are **delta checkpoints, not standalone Foundation models**. They require:',
        'The trained checkpoints are deltas. Their complete Foundation dependency is now included under `foundation/`; no separate weight download is needed. Original upstream source:')
    readme = readme.replace('Download the Foundation separately. Then run the following single-line command,\nreplacing both paths with your local absolute directories:',
        'The Foundation is included in this repository. After authenticating ModelScope (or setting MODELSCOPE_API_TOKEN), download everything with this single-line command:\n\n```bash\npython -c "from modelscope import snapshot_download; snapshot_download(\'ykjykj/LabWAM-BoxedV3-30k-20261111\', local_dir=\'./LabWAM\')"\n```\n\nThen configure the downloaded bundle with this single-line command; replace the bundle path with your local absolute directory:')
    readme = readme.replace(' --foundation /path/to/OpenWAM-Alpha-Pretrain-Foundation-Model','')
    if '- foundation/:' not in readme:
        readme = readme.replace('## Contents and behavior\n',
            '## Contents and behavior\n\n- foundation/: original 154000-step checkpoint, reconstruction config, tokenizer and upstream model card.\n')
    (bundle/'README.md').write_text(readme)
    for name in ('configure_foundation.py','lab-low-code.tar.gz','README.md'):
        path=bundle/name
        manifest['files'][name]=dict(bytes=path.stat().st_size,sha256=sha256(path))
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('FOUNDATION_PACKAGE_READY '+json.dumps(dict(files=len(manifest['files'])+1,
          bytes=sum(v['bytes'] for v in manifest['files'].values()))),flush=True)


def publish(bundle, repo_id, receipt_name=None):
    from modelscope_hub import HubApi
    from modelscope_hub.constants import Visibility
    from modelscope_hub.errors import NotExistError
    token = sys.stdin.read().strip()
    if not token:
        raise ValueError('Token required on stdin')
    api = HubApi(token=token)
    try:
        assert api.whoami().username == repo_id.split('/')[0]
        try:
            repo = api.get_repo(repo_id,'model')
        except NotExistError:
            repo = api.create_repo(repo_id,'model',visibility='private',license='apache-2.0',
                                   description='Completed 30k LabWAM manipulation delta checkpoints')
        repo = api.get_repo(repo_id,'model')
        if repo.visibility != Visibility.PRIVATE and repo.private is not True:
            raise RuntimeError('Refusing upload to a non-private repository')
        manifest = json.loads((bundle/'MANIFEST.json').read_text())
        expected = dict(manifest['files'])
        expected['MANIFEST.json'] = dict(bytes=(bundle/'MANIFEST.json').stat().st_size,sha256=sha256(bundle/'MANIFEST.json'))
        existing = {f.path for f in api.list_repo_files(repo_id,'model') if not f.is_dir}
        if existing-set(expected)-{'.gitattributes'}:
            raise RuntimeError('Repository contains unexpected files; refusing to overwrite')
        last = [0.]
        def progress(event):
            if time.monotonic()-last[0] > 30:
                # SDK callbacks can contain signed URLs; report only counters.
                print('UPLOAD_IN_PROGRESS '+time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
                last[0] = time.monotonic()
        print('UPLOADING '+repo_id,flush=True)
        tracker = ROOT/'.cache/modelscope_upload'/repo_id.split('/')[-1]/'tracker.json'
        tracker.parent.mkdir(parents=True, exist_ok=True)
        api.upload_folder(repo_id,'model',bundle,max_workers=2,disable_tqdm=True,
            commit_message='Back up LabWAM weights and included Foundation' if manifest.get('foundation_included') else f'Back up {len(manifest["models"])} completed 30000-step LabWAM models',
            sync_remote_repo=False,tracker_path=tracker,progress_callback=progress)
        remote = {f.path:f for f in api.list_repo_files(repo_id,'model') if not f.is_dir}
        verified = []
        for name, spec in expected.items():
            f = remote[name]
            if f.size != spec['bytes']:
                raise RuntimeError('Remote size mismatch: '+name)
            digest = f.sha256 or (f.lfs or {}).get('sha256') or (f.lfs or {}).get('oid')
            if digest:
                if digest.removeprefix('sha256:') != spec['sha256']:
                    raise RuntimeError('Remote SHA256 mismatch: '+name)
            else:
                # For ordinary Git blobs, verify the content-addressed Git ID.
                local = bundle/name
                h = hashlib.sha1(b'blob '+str(spec['bytes']).encode()+b'\0')
                h.update(local.read_bytes())
                if f.blob_id != h.hexdigest():
                    raise RuntimeError('Remote Git blob hash unavailable/mismatch: '+name)
            verified.append(name)
        report = dict(repo_id=repo_id,url='https://modelscope.cn/models/'+repo_id,private=True,
            verified_files=len(verified),verified_bytes=sum(x['bytes'] for x in expected.values()),
            verification='remote SHA256/LFS object ID, or Git blob SHA1, and exact byte sizes',
            bundle=str(bundle),models=manifest['models'],foundation_included=manifest.get('foundation_included',False))
        receipt = receipt_name or ('modelscope_foundation_upload.json' if manifest.get('foundation_included') else 'modelscope_upload.json')
        if Path(receipt).name != receipt or not receipt.endswith('.json'):
            raise ValueError('Receipt must be a JSON filename in validation/')
        (ROOT/'validation'/receipt).write_text(json.dumps(report,indent=2)+'\n')
        print('VERIFIED '+json.dumps(report),flush=True)
    except Exception as error:
        print(type(error).__name__+': '+str(error).replace(token,'[REDACTED]'),file=sys.stderr,flush=True)
        raise SystemExit(1) from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode',choices=('prepare','add-foundation','publish'))
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--repo-id',default='ykjykj/LabWAM-BoxedV3-30k-20261111')
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    if not bundle.is_relative_to('/share/personal/yekangjie/lab-low'):
        raise ValueError('Export must remain in the authorized shared project')
    if args.mode == 'prepare':
        prepare(bundle)
    elif args.mode == 'add-foundation':
        add_foundation(bundle)
    else:
        publish(bundle,args.repo_id)


if __name__ == '__main__':
    main()
