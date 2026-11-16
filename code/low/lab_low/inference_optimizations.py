"""Inference-only reuse and OpenWAM acceleration options; no robot controls."""
from collections import OrderedDict
from dataclasses import dataclass, asdict
import hashlib
import math


@dataclass(frozen=True)
class InferenceOptimizations:
    condition_cache: bool = True
    compile_mode: str = "none"
    dit_cache_threshold: float | None = None
    dit_cache_max_skips: int = 1

    def __post_init__(self):
        if self.compile_mode not in ("none", "default", "reduce-overhead"):
            raise ValueError("Unsupported compilation mode")
        if self.dit_cache_threshold is not None and (
                not math.isfinite(self.dit_cache_threshold) or not 0 < self.dit_cache_threshold <= 1):
            raise ValueError("DiT cache threshold must be finite and in (0,1]")
        if self.dit_cache_max_skips < 1:
            raise ValueError("DiT cache max skips must be positive")

    def configure_model(self, model):
        if self.compile_mode != "none":
            model.apply_compile_optimizations(dict(enabled=True, self_attn=dict(
                torch_mode=self.compile_mode, dynamic=False)))

    def new_dit_cache(self):
        if self.dit_cache_threshold is None:
            return None
        from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
        return DiTVelocityCache(self.dit_cache_threshold, self.dit_cache_max_skips)

    def metadata(self):
        return asdict(self)


class ImageLatentCache:
    """Bounded, per-session cache keyed by actual composed pixels, not frame IDs.

    Content keys also cover different reference/current images with equal frame
    numbers. Clear on reset/stop; never share between operator and simulation.
    """
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def encode(self, image, *, dtype, device, encoder):
        key = (image.mode, image.size, str(dtype), str(device),
               hashlib.blake2b(image.tobytes(), digest_size=16).digest())
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        value = encoder(image).detach()
        self.entries[key] = value
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return value


def add_optimization_arguments(parser):
    parser.add_argument("--compile-mode", choices=("none", "default", "reduce-overhead"), default="none")
    parser.add_argument("--no-condition-cache", action="store_true")
    parser.add_argument("--dit-cache-threshold", type=float, default=None)
    parser.add_argument("--dit-cache-max-skips", type=int, default=1)
    parser.add_argument("--warmup-reset", help="JSON reset observation for model-only startup warmup")


def options_from_arguments(args):
    return InferenceOptimizations(not args.no_condition_cache, args.compile_mode,
                                  args.dit_cache_threshold, args.dit_cache_max_skips)


def worker_optimization_arguments(args):
    result = ["--compile-mode", args.compile_mode, "--dit-cache-max-skips", str(args.dit_cache_max_skips)]
    if args.no_condition_cache:
        result.append("--no-condition-cache")
    if args.dit_cache_threshold is not None:
        result += ["--dit-cache-threshold", str(args.dit_cache_threshold)]
    if args.warmup_reset:
        result += ["--warmup-reset", args.warmup_reset]
    return result
