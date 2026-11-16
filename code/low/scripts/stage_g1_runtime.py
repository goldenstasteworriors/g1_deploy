"""Create relocatable runtime model configs without modifying downloaded files."""
import argparse
import json
from pathlib import Path
import shutil
from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    bundle, runtime = args.bundle.resolve(), args.runtime.resolve()
    capabilities = json.loads((Path(__file__).resolve().parents[1] / "configs/deploy_g1_capabilities.json").read_text())
    if runtime == bundle or runtime.is_relative_to(bundle):
        raise ValueError("Runtime must be outside the original bundle")
    foundation = bundle / "foundation/checkpoint_step_154000.safetensors"
    if not foundation.is_file():
        raise FileNotFoundError(foundation)
    names = []
    for source in sorted((bundle / "models").iterdir()):
        if not source.is_dir():
            continue
        pairs = capabilities["models"][source.name]
        contract = json.loads((source / "data_contract.json").read_text())
        if set(contract["skills"]) != {pair[0] for pair in pairs} or set(contract["hands"]) != {pair[1] for pair in pairs}:
            raise ValueError("Trained capabilities differ from this deployment manifest: " + source.name)
        destination = runtime / "models" / source.name
        destination.mkdir(parents=True, exist_ok=False)
        for path in source.iterdir():
            target = destination / path.name
            if path.suffix == ".safetensors":
                target.symlink_to(path)
            elif path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copyfile(path, target)
        config = destination / "config.yaml"
        cfg = OmegaConf.load(config)
        cfg.model.architecture.foundation_checkpoint = str(foundation)
        cfg.model.video_backbone.model_path = str(foundation.parent)
        cfg.model.video_backbone.encoder.model_path = str(foundation.parent)
        cfg.training.finetune_ckpt_path = str(foundation.parent)
        OmegaConf.save(cfg, config)
        (destination / "supported_skill_hands.json").write_text(json.dumps(pairs, indent=2) + "\n")
        names.append(source.name)
    report = {"bundle": str(bundle), "runtime": str(runtime), "models": names,
              "foundation": str(foundation), "downloaded_files_modified": False}
    (runtime / "staging.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
