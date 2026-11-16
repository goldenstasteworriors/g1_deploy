"""Publish explicitly selected, completed /share runs with Foundation provenance."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from publish_modelscope import ROOT, publish, sha256

REPO = "ykjykj/LabWAM-Additional-30k-20261112"
FOUNDATION_REPO = "ykjykj/LabWAM-BoxedV3-30k-20261111"
ALLOWED = {"return_dropper", "water_bath", "return_tube", "take_tube_light"}


def prepare(bundle, selections):
    from omegaconf import OmegaConf
    from safetensors import safe_open
    if not selections:
        raise ValueError("Select completed runs explicitly with --run name=path")
    runs = {}
    # Validate every completion marker before creating any export files.
    for selection in selections:
        name, value = selection.split("=", 1)
        source = Path(value).resolve()
        if name not in ALLOWED or name in runs or not source.is_relative_to(ROOT / "outputs/share_runs" / name):
            raise ValueError("Unexpected or duplicate run: " + name)
        completed = json.loads((source / "completed.json").read_text())
        checkpoint = Path(completed["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(source / "weights") or checkpoint.name != "checkpoint_step_30000.safetensors":
            raise ValueError("Expected this run's final 30k checkpoint")
        run = checkpoint.parent
        readback = json.loads((run / "checkpoint_readback.json").read_text())
        if not completed.get("exact_match") or not readback.get("exact_match") or completed["tensors"] != readback["tensors"]:
            raise ValueError("Completed checkpoint readback is missing or failed")
        if Path(readback["checkpoint"]).resolve() != checkpoint:
            raise ValueError("Readback refers to a different checkpoint")
        vals = [json.loads(x) for x in (run / "validation.jsonl").read_text().splitlines() if x.strip()]
        if not vals or vals[-1]["step"] != 30000:
            raise ValueError("Missing final 30k validation")
        contract = json.loads((run / "data_contract.json").read_text())
        if len(contract["skills"]) != 1 or len(contract["hands"]) != 1:
            raise ValueError("Expected an explicitly trained single skill/hand combination")
        runs[name] = (source, run, readback, contract)
    bundle.mkdir(parents=True, exist_ok=False)
    entries = []
    for name, (source, run, readback, contract) in runs.items():
        target = bundle / "models" / name
        target.mkdir(parents=True)
        weights = [run / "checkpoint_step_30000.safetensors"]
        if (run / "checkpoint_best.safetensors").exists():
            weights.append(run / "checkpoint_best.safetensors")
        for src in weights:
            with safe_open(str(src), framework="pt", device="cpu") as f:
                meta = f.metadata()
                if meta["lab_checkpoint"] != "delta_v1" or meta["conditioner_version"] != "spacetime_manip_v3":
                    raise ValueError("Unexpected model format")
                if len(f.keys()) != readback["tensors"] or set(f.keys()) != set(json.loads(meta["tensor_names"])):
                    raise ValueError("Checkpoint tensor inventory mismatch")
            try:
                os.link(src, target / src.name)
            except OSError:
                shutil.copyfile(src, target / src.name)
        for filename in ("config.yaml", "normalization_stats.npy", "best_metric.json", "checkpoint_readback.json",
                         "data_contract.json", "transfer_report.json", "gradient_checks.json", "validation.jsonl"):
            shutil.copyfile(run / filename, target / filename)
        shutil.copyfile(source / "completed.json", target / "completed.json")
        shutil.copytree(run / "tokenizer", target / "tokenizer")
        pairs = [[contract["skills"][0], contract["hands"][0]]]
        (target / "supported_skill_hands.json").write_text(json.dumps(pairs, indent=2) + "\n")
        cfg = OmegaConf.load(run / "config.yaml")
        entries.append({"name": name, "source_run": str(run.relative_to(ROOT)), "final_step": 30000,
                        "supported_skill_hands": pairs,
                        "photometric_augmentation": bool(OmegaConf.select(cfg, "dataloader.photometric.enabled", default=False)),
                        "best": json.loads((run / "best_metric.json").read_text())})
        print("PACKED " + name, flush=True)
    subprocess.run(["git", "archive", "--format=tar.gz", "--output=" + str(bundle / "lab-low-code.tar.gz"), "HEAD"], cwd=ROOT, check=True)
    shutil.copyfile(ROOT / "scripts/configure_downloaded_foundation.py", bundle / "configure_foundation.py")
    shutil.copyfile(ROOT / "LICENSE", bundle / "LICENSE")
    (bundle / "configuration.json").write_text('{"framework":"pytorch"}\n')
    names = ", ".join(runs)
    (bundle / "README.md").write_text(f"""---
license: apache-2.0
tags: [robotics, openwam, labwam]
---
# LabWAM additional completed 30k skills

Private completed checkpoints: **{names}**. Each final checkpoint reached 30,000
steps and passed exact tensor-by-tensor training readback. The best validation
checkpoint is included when distinct from the final checkpoint. Validation flow
loss is not a robot success rate. No unfinished runs or raw trajectory images.

These are spacetime_manip_v3 deltas. Download their full Foundation dependency
from https://modelscope.cn/models/{FOUNDATION_REPO} (the foundation/ directory).
The original four models remain in that repository; this package adds separate
single-skill models and does not retrain or replace the general model.

Each models/<name>/ includes config, tokenizer, normalization, trained skill/hand
pairs, final/best weights, completion and readback reports, and validation losses.
MANIFEST.json records exact sizes and SHA256 for every payload, plus augmentation
settings. lab-low-code.tar.gz is the matching source snapshot.

## Restore

Download both private repositories with ModelScope authentication. Extract this
package's lab-low-code.tar.gz into your project and use its OpenWAM environment.
Run this **single-line** command, replacing both paths with actual absolute paths:

```bash
python /path/to/additional/configure_foundation.py --bundle /path/to/additional --foundation /path/to/original/foundation
```

This keeps config.original.yaml backups and changes only configuration paths.
Then point scripts/deploy_lab_worker.py --checkpoint to an explicit
models/<name>/checkpoint_step_30000.safetensors (or checkpoint_best.safetensors
when present). The worker uses the accompanying supported_skill_hands.json and
normalization_stats.npy. It emits predictions only, not motor commands.

Inputs: three RGB views, observed 31D state, trained skill/hand, one initial box,
and causal history. Outputs: 32x16 arm/Dex1 targets, no locomotion. No claim of
closed-loop hardware success is made for these newly trained checkpoints.
""")
    files = {str(p.relative_to(bundle)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(bundle.rglob("*")) if p.is_file()}
    manifest = {"collection": "additional_completed", "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "models": entries, "foundation_included": False, "foundation_modelscope_repo": FOUNDATION_REPO,
                "foundation_sha256": "180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d", "files": files}
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"models": list(runs), "files": len(files) + 1, "bytes": sum(x["bytes"] for x in files.values())}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["prepare", "publish"])
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--run", action="append", default=[])
    p.add_argument("--repo", default=REPO)
    args = p.parse_args()
    bundle = args.bundle.resolve()
    if not bundle.is_relative_to("/share/personal/yekangjie/lab-low/exports"):
        raise ValueError("Export must remain on the authorized project share")
    if args.operation == "prepare":
        prepare(bundle, args.run)
    else:
        publish(bundle, args.repo, "modelscope_additional_upload_20261112.json")


if __name__ == "__main__":
    main()
