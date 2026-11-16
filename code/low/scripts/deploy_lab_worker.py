"""JSONL inference worker for a downloaded LabWAM checkpoint; no motor commands."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_low.inference_optimizations import add_optimization_arguments, options_from_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    add_optimization_arguments(parser)
    args = parser.parse_args()
    optimizations = options_from_arguments(args)
    if not args.checkpoint.is_file() or args.checkpoint.suffix != ".safetensors":
        raise ValueError("An explicit checkpoint safetensors file is required")
    if not 1 <= args.steps <= 100:
        raise ValueError("steps must be in [1,100]")
    capabilities = json.loads((args.checkpoint.parent / "supported_skill_hands.json").read_text())
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from lab_low.deployment import InferenceSession
        from lab_low.inference_sessions import InferenceSessions
        from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
        from openwam.deploy.denoise_schedule import schedule_sync
        _, model, _ = build_architecture_from_ckpt_dir(str(args.checkpoint), weights_required=True)
        model.set_dtype_device(torch.bfloat16, torch.device("cuda:0"))
        model.eval().requires_grad_(False)
        model.normalizer = None
        model.init_training_schedulers(1000)
        optimizations.configure_model(model)
        schedule = schedule_sync(model.video_scheduler, model.action_scheduler, args.steps, 5.)
        session = InferenceSessions(lambda: InferenceSession(
            model, args.checkpoint.parent / "normalization_stats.npy",
            schedule, steps=args.steps, allowed_pairs=capabilities, optimizations=optimizations))
        warmup_seconds = None
        if args.warmup_reset:
            import time
            packet = json.loads(Path(args.warmup_reset).read_text())
            packet.update(op="reset", binding_id="simulation-startup-warmup", client_scope="simulation")
            started = time.monotonic()
            session.request(packet)
            session.request(dict(op="predict", binding_id=packet["binding_id"], client_scope="simulation"))
            session.stop()
            warmup_seconds = time.monotonic() - started
    print(json.dumps({"ready": True, "checkpoint": str(args.checkpoint),
                      "optimizations": {**optimizations.metadata(), "warmup_seconds": warmup_seconds,
                          "warmup_required": optimizations.compile_mode != "none" and warmup_seconds is None,
                          "compiled_active": (getattr(model, "_compiled_mot_run_joint_loop", None) is not None)
                              if warmup_seconds is not None else None},
                      "denoise_steps": args.steps,
                      "protocol_features": ["recorded_reference_v1","per_request_denoise_steps",
                                            "isolated_client_scopes_v1"],
                      "execution_allowed": False, "supported_skill_hands": capabilities}), flush=True)
    for line in sys.stdin:
        try:
            packet = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = session.request(packet)
            print(json.dumps({"ok": True, **result}, allow_nan=False), flush=True)
        except Exception as error:
            # The router invalidates only the failing request's active owner.
            print(json.dumps({"ok": False, "error": str(error), "reset_required": True}), flush=True)
    session.stop()


if __name__ == "__main__":
    main()
