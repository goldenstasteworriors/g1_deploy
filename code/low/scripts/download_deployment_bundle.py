"""Download and hash-check the complete ModelScope bundle into a new project.

Credentials are read from stdin only; never saved. Downloaded source is retained
as an archive. Configuration rewriting happens separately, after verification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def verify(bundle, kind="weights"):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    if kind == "weights" and not manifest.get("foundation_included"):
        raise ValueError("The complete Foundation must be included")
    if kind == "deployment" and manifest.get("weights_repo") != "ykjykj/LabWAM-BoxedV3-30k-20261111":
        raise ValueError("Unexpected deployment dependency")
    total = 0
    for name, expected in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file():
            raise ValueError(f"Invalid or missing bundle file: {name}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"SHA256 mismatch: {name}")
        total += expected["bytes"]
        print(f"VERIFIED {name}", flush=True)
    return {"verified_files": len(manifest["files"]), "verified_bytes": total,
            "foundation_included": bool(manifest.get("foundation_included")),
            "kind": kind, "code_revision": manifest["code_revision"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--kind", choices=["weights", "deployment"], default="weights")
    args = parser.parse_args()
    if args.repo is None:
        args.repo = ("ykjykj/LabWAM-BoxedV3-30k-20261111" if args.kind == "weights"
                     else "ykjykj/LabWAM-G1-Deployment-20261112")
    if not args.verify_only:
        from modelscope_hub import HubApi
        token = sys.stdin.read().strip()
        if not token:
            raise ValueError("ModelScope token required on stdin")
        args.bundle.mkdir(parents=True, exist_ok=True)
        # Dedicated unmodified download directory: do not rewrite configs here.
        api = HubApi(token=token)
        api.download_repo(args.repo, "model", revision=args.revision,
                          local_dir=args.bundle, max_workers=2)
    report = verify(args.bundle, args.kind)
    report.update(repo=args.repo, revision=args.revision)
    destination = args.bundle.parent / (args.bundle.name + ".verified.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
