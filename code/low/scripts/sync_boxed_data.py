"""Foreground resumable SSH transfer for the explicitly selected boxed suite.

Uses only standard tools/Python on the recipient; no rsync installation or
global SSH changes. A failed transfer is retried against remote SHA-256 hashes.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_CHECK = r'''
import hashlib,json,sys
from pathlib import Path
request=json.load(sys.stdin)
root=Path(request['root']); root.mkdir(parents=True,exist_ok=True)
missing=[]
for name,spec in request['files'].items():
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in ('data','meta','videos'):
        raise ValueError('Unsafe transfer path')
    path=root/rel
    if not path.is_file() or path.stat().st_size!=spec['bytes']:
        missing.append(name); continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk=f.read(4*1024*1024)
            if not chunk: break
            h.update(chunk)
    if h.hexdigest()!=spec['sha256']: missing.append(name)
if not missing:
    ready=dict(manifest_digest=request['digest'],files=len(request['files']),
               bytes=sum(s['bytes'] for s in request['files'].values()))
    marker=request.get('ready_name','.boxed_suite_ready.json')
    if Path(marker).name != marker: raise ValueError('Ready marker must be a filename')
    tmp=root/(marker+'.pending')
    tmp.write_text(json.dumps(ready,indent=2)+'\n'); tmp.replace(root/marker)
print(json.dumps(dict(missing=missing)),flush=True)
'''


def inventory(dataset, manifest):
    info=json.loads((dataset/'meta/info.json').read_text())
    paths=set((dataset/'meta').glob('*'))
    for row in json.loads(manifest.read_text())['segments']:
        ep=row['episode_index']; fields=dict(episode_index=ep,episode_chunk=ep//info['chunks_size'])
        paths.add(dataset/info['data_path'].format(**fields))
        for camera in ('observation.images.ego_view','observation.images.left_wrist','observation.images.right_wrist'):
            paths.add(dataset/info['video_path'].format(video_key=camera,**fields))
    result={}
    for path in sorted(paths):
        if not path.is_file(): raise FileNotFoundError(path)
        h=hashlib.sha256()
        with path.open('rb') as f:
            while chunk:=f.read(4*1024*1024): h.update(chunk)
        result[str(path.relative_to(dataset))]=dict(bytes=path.stat().st_size,sha256=h.hexdigest())
    digest=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result,digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--remote-root',required=True)
    ap.add_argument('--inventory-only',action='store_true')
    ap.add_argument('--ready-name',default='.boxed_suite_ready.json')
    ap.add_argument('--cache-name',default='boxed_transfer')
    args=ap.parse_args()
    dataset=args.dataset.resolve()
    files,digest=inventory(dataset,args.manifest)
    if Path(args.cache_name).name != args.cache_name or Path(args.ready_name).name != args.ready_name:
        ap.error('cache-name and ready-name must be single names')
    cache=ROOT/'.cache'/args.cache_name; cache.mkdir(parents=True,exist_ok=True)
    request=dict(root=args.remote_root,files=files,digest=digest,ready_name=args.ready_name)
    (cache/'inventory.json').write_text(json.dumps(request,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),bytes=sum(s['bytes'] for s in files.values()),digest=digest)),flush=True)
    if args.inventory_only: return
    attempt=0
    while True:
        attempt+=1
        try:
            checked=subprocess.run(['bash',str(ROOT/'scripts/a800_ssh.sh'),'python3 -c '+shlex.quote(REMOTE_CHECK)],
                input=json.dumps(request),text=True,capture_output=True,check=True)
            missing=json.loads(checked.stdout)['missing']
            state=dict(attempt=attempt,missing=len(missing),state='complete' if not missing else 'transferring')
            (cache/'status.json').write_text(json.dumps(state,indent=2)+'\n'); print(json.dumps(state),flush=True)
            if not missing: return
            listing=cache/'missing.txt'; listing.write_text('\n'.join(missing)+'\n')
            tar=subprocess.Popen(['tar','-C',str(dataset),'-czf','-','-T',str(listing)],stdout=subprocess.PIPE)
            try:
                result=subprocess.run(['bash',str(ROOT/'scripts/a800_ssh.sh'),
                    'tar -C '+shlex.quote(args.remote_root)+' -xzf -'],stdin=tar.stdout)
                tar.stdout.close()
                code=tar.wait()
                if result.returncode or code: raise RuntimeError(f'SSH/tar transfer failed: {result.returncode}/{code}')
            finally:
                if tar.poll() is None: tar.terminate(); tar.wait()
        except (subprocess.CalledProcessError,RuntimeError,json.JSONDecodeError) as error:
            state=dict(attempt=attempt,state='retrying',error=str(error))
            (cache/'status.json').write_text(json.dumps(state,indent=2)+'\n'); print(json.dumps(state),flush=True)
            time.sleep(30)


if __name__=='__main__': main()
