"""Stage downloaded ModelScope models without changing weights or robot state."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--extra-bundle', type=Path, action='append', default=[])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    from omegaconf import OmegaConf
    root, bundle = args.root.resolve(), args.bundle.resolve()
    foundation = bundle / 'foundation/checkpoint_step_154000.safetensors'
    if not foundation.is_file():
        raise FileNotFoundError(foundation)
    defaults = json.loads((root / 'code/low/configs/deploy_g1_capabilities.json').read_text())['models']
    plans = []
    names = set()
    for package in [bundle, *[p.resolve() for p in args.extra_bundle]]:
        for source in sorted((package / 'models').iterdir()):
            if not source.is_dir():
                continue
            if source.name in names:
                raise ValueError('Duplicate model: ' + source.name)
            names.add(source.name)
            pairs_file = source / 'supported_skill_hands.json'
            pairs = json.loads(pairs_file.read_text()) if pairs_file.exists() else defaults[source.name]
            contract = json.loads((source / 'data_contract.json').read_text())
            if ({p[0] for p in pairs} != set(contract['skills']) or
                    {p[1] for p in pairs} != set(contract['hands'])):
                raise ValueError('Capability mismatch: ' + source.name)
            if not (source / 'checkpoint_step_30000.safetensors').is_file():
                raise FileNotFoundError(source / 'checkpoint_step_30000.safetensors')
            destination = root / 'runtime/models' / source.name
            if destination.exists():
                raise FileExistsError(destination)
            cfg = OmegaConf.load(source / 'config.yaml')
            cfg.model.architecture.foundation_checkpoint = str(foundation)
            cfg.model.video_backbone.model_path = str(foundation.parent)
            cfg.model.video_backbone.encoder.model_path = str(foundation.parent)
            cfg.training.finetune_ckpt_path = str(foundation.parent)
            plans.append((source, destination, pairs, cfg))
    if not plans:
        raise ValueError('No models found')
    for source, destination, pairs, cfg in plans:
        destination.mkdir(parents=True)
        for path in source.iterdir():
            target = destination / path.name
            if path.suffix == '.safetensors':
                target.symlink_to(path)
            elif path.is_dir():
                shutil.copytree(path, target)
            else:
                shutil.copyfile(path, target)
        OmegaConf.save(cfg, destination / 'config.yaml')
        (destination / 'supported_skill_hands.json').write_text(json.dumps(pairs, indent=2) + '\n')
    print(json.dumps({'models': sorted(names), 'runtime': str(root / 'runtime'),
                      'robot_io_opened': False, 'source_files_modified': False}))


if __name__ == '__main__':
    main()
