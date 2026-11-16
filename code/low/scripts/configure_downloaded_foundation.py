"""Point downloaded LabWAM delta configs to an existing Foundation directory."""
import argparse
from pathlib import Path
import shutil
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--foundation', type=Path, help='Defaults to the bundled foundation directory')
    args = ap.parse_args()
    foundation = (args.foundation or args.bundle/'foundation').resolve()
    weights = foundation/'checkpoint_step_154000.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    configs = sorted((args.bundle/'models').glob('*/config.yaml'))
    if not configs:
        raise ValueError('No downloaded model configs')
    for path in configs:
        backup = path.with_name('config.original.yaml')
        if not backup.exists():
            shutil.copyfile(path, backup)
        cfg = OmegaConf.load(path)
        cfg.model.architecture.foundation_checkpoint = str(weights)
        cfg.model.video_backbone.model_path = str(foundation)
        cfg.model.video_backbone.encoder.model_path = str(foundation)
        cfg.training.finetune_ckpt_path = str(foundation)
        OmegaConf.save(cfg, path)
        print(path)


if __name__ == '__main__':
    main()
