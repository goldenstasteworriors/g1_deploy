"""Package the deployment source and selected SONICMJ runtime, then publish privately."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def prepare(bundle, sonic):
    bundle.mkdir(parents=True, exist_ok=False)
    selected = ["LICENSE", "decoupled_wbc/__init__.py", "decoupled_wbc/version.py",
        "decoupled_wbc/pyproject.toml", "decoupled_wbc/control", "decoupled_wbc/data",
        "decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml",
        "decoupled_wbc/sim2mujoco/resources/robots/g1/policy",
        "gear_sonic/__init__.py", "gear_sonic/camera", "gear_sonic/utils",
        "external_dependencies/unitree_sdk2_python", "install_scripts/install_decoupled_vla_collection.sh"]
    def filter_member(member):
        parts = Path(member.name).parts
        if ".git" in parts or "__pycache__" in parts or member.name.endswith(".pyc"):
            return None
        if member.issym() or member.islnk():
            raise ValueError("Source archive must contain regular files only: " + member.name)
        return member
    with tarfile.open(bundle / "sonicmj-dex1-runtime.tar.gz", "w:gz") as archive:
        for name in selected:
            path = sonic / name
            if path.exists():
                archive.add(path, arcname=name, filter=filter_member)
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(bundle / "lab-low-deployment.tar.gz"), "HEAD"], cwd=ROOT, check=True)
    shutil.copyfile(ROOT / "docs/G1_DEPLOYMENT_zh.md", bundle / "DEPLOYMENT_zh.md")
    (bundle / "configuration.json").write_text('{"framework":"pytorch"}\n')
    (bundle / "README.md").write_text("""---
license: other
---
# LabWAM G1 deployment assets

Private deployment source, native Dex1 adapter, and selected SONICMJ decoupled
runtime with its original Balance/Walk ONNX files and licenses. LabWAM model
weights including Foundation are in ykjykj/LabWAM-BoxedV3-30k-20261111.
Both repositories are required. See DEPLOYMENT_zh.md for tested scope and commands.
The included SONICMJ installer is preserved as source provenance; it is not an
instruction to run all data-collection dependency installation for inference.
No recorded camera frames, datasets, credentials or robot commands are uploaded.
LabWAM code retains Apache-2.0; SONICMJ LICENSE and the ONNX policy's NVIDIA Open
Model License remain inside the runtime archive. Hardware success is not claimed.
""")
    manifest = {"code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "sonic_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=sonic, text=True).strip(),
                "weights_repo": "ykjykj/LabWAM-BoxedV3-30k-20261111", "files": {}}
    for path in sorted(bundle.iterdir()):
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(manifest["files"]) + 1, "bytes": sum(x["bytes"] for x in manifest["files"].values())}))


def publish(bundle, repo):
    from modelscope_hub import HubApi
    from modelscope_hub.errors import NotExistError
    from modelscope_hub.constants import Visibility
    token = sys.stdin.read().strip()
    if not token:
        raise ValueError("Token required on stdin")
    api = HubApi(token=token)
    if api.whoami().username != repo.split("/")[0]:
        raise ValueError("Unexpected repository owner")
    try:
        info = api.get_repo(repo, "model")
    except NotExistError:
        info = api.create_repo(repo, "model", visibility="private", description="LabWAM G1 deployment source and decoupled controller")
    info = api.get_repo(repo, "model")
    if info.visibility != Visibility.PRIVATE and info.private is not True:
        raise ValueError("Repository must remain private")
    api.upload_folder(repo, "model", bundle, max_workers=2, disable_tqdm=True,
                      sync_remote_repo=False, commit_message="Package G1 dual-arm Dex1 deployment")
    print(json.dumps({"uploaded": repo, "private": True, "download_verification_required": True}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["prepare", "publish"])
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--sonic", type=Path)
    p.add_argument("--repo", default="ykjykj/LabWAM-G1-Deployment-20261112")
    args = p.parse_args()
    if args.operation == "prepare":
        if args.sonic is None:
            p.error("--sonic is required")
        prepare(args.bundle.resolve(), args.sonic.resolve())
    else:
        publish(args.bundle.resolve(), args.repo)


if __name__ == "__main__":
    main()
