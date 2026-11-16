"""OpenWAM dual MoT: shared target/history context reaches BOTH branches."""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.video_backbone.wan import encode as wan_encode
from lab_low.conditioning import LabConditioner
from lab_low.jitter import jitter_weights, latent_jitter
from lab_low.skills import ACTION_DIM, STATE_DIM, CONTRACT_VERSION


@register_architecture("dual_system_lab", framework="dual_system", variant="lab")
class LabArchitecture(DualSystemSelfAttnArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        if self.action_dim != ACTION_DIM or self.proprio_dim != STATE_DIM:
            raise ValueError("Manipulation-only lab requires 16D actions (left8 + right8) and 31D observations")
        self.lab_conditioner = LabConditioner(context_dim=self.context_dim,
            width=int(cfg.get("condition_width", 256)),
            prior_strength=float(cfg.get("prior_strength", 2.)),
            memory_depth=int(cfg.get("memory_depth", 4)),
            temporal_every=int(cfg.get("memory_temporal_every", 4)))
        self.conditioner_version = CONTRACT_VERSION
        self.last_lab_token_count = 0
        self.jitter_scale = float(cfg.get("jitter_scale", .025))
        self.jitter_floor = float(cfg.get("jitter_floor", .1))
        self.transfer_from_foundation = bool(cfg.get("transfer_from_foundation", False))
        self.foundation_checkpoint = str(cfg.get("foundation_checkpoint", ""))
        self.delta_checkpoint = bool(cfg.get("delta_checkpoint", True))
        self.transfer_report = {}

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        if hasattr(self, "lab_conditioner"):
            self.lab_conditioner.to(device=device, dtype=dtype)

    def freeze_modules(self, names):
        """Freeze video PARAMETERS without severing trainable input gradients.

        Upstream freeze_modules recursively wraps every child in no_grad.
        That is appropriate for feature extraction but would detach the lab
        tokens at video DiT.text_embedding, making video jitter loss unable to
        train the conditioner. Frozen Wan layers still need an autograd graph.
        """
        frozen, ordinary = [], []
        for name in names:
            if name == "video_backbone" or name.startswith("video_backbone.dit"):
                module = self.get_submodule(name)
                module.requires_grad_(False)
                for child in module.modules():
                    if getattr(child, "_openwam_no_grad_wrapped", False):
                        raise RuntimeError(f"{name} already has a no_grad wrapper; rebuild before training")
                    child.training = False
                frozen.append(name)
            else:
                ordinary.append(name)
        return frozen + super().freeze_modules(ordinary)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.video_backbone.parameters()):
            for module in self.video_backbone.modules():
                module.training = False
        return self

    @torch.no_grad()
    def prepare_condition(self, samples, *, latent_cache=None):
        """Common train/inference path. Input includes no future target frames."""
        if latent_cache is not None and self.training:
            raise ValueError("Latent reuse is only available for inference")
        vb = self.video_backbone
        def encode(image):
            pixels = wan_encode.preprocess_video([image], dtype=self.dtype, device=self.device)
            return wan_encode.encode_video(pixels, vae=vb.vae, encoder=vb.video_encoder)[:, :, 0]
        all_latents = []
        for sample in samples:
            if sample["history_indices"][-1] != sample["start_frame"]:
                raise ValueError("Last history image must be current observation")
            if max(sample["history_indices"]) > sample["start_frame"]:
                raise ValueError("Future observation leaked into memory")
            cache, latents = {}, []
            for frame, image in zip(sample["history_indices"], sample["condition_images"]):
                if frame not in cache:
                    cache[frame] = encode(image) if latent_cache is None else latent_cache.encode(
                        image, dtype=self.dtype, device=self.device, encoder=encode)
                latents.append(cache[frame])
            all_latents.append(torch.stack(latents, dim=1))
        out = dict(latents=torch.cat(all_latents).to(self.dtype))
        mapping = dict(prior="target_prior", mark_valid="mark_valid", confidence="mark_confidence",
            states="history_state", previous_actions="previous_motion", previous_action_valid="previous_motion_valid",
            frame_valid="history_valid", relative_time="relative_time")
        for name, source in mapping.items():
            tensor = torch.stack([s[source] for s in samples]).to(self.device)
            out[name] = tensor.to(self.dtype) if tensor.is_floating_point() else tensor
        for name in ("skill_id", "hand_id"):
            out[name] = torch.tensor([s[name] for s in samples], device=self.device, dtype=torch.long)
        return out

    @torch.no_grad()
    def prepare_inference_tokens(self, condition):
        """Reuse fixed conditioning within ONE generate call, never across states."""
        if self.training:
            raise ValueError("Prepared tokens are inference-only")
        tokens, attention = self.lab_conditioner(**condition)
        return {**condition, "_prepared_tokens": (tokens, attention,
                self.lab_conditioner.context_mask(condition["frame_valid"]))}

    @torch.no_grad()
    def prepare_inputs(self, batch):
        samples = [batch] if isinstance(batch, dict) else batch
        inputs = super().prepare_inputs(samples)
        inputs["lab_condition"] = self.prepare_condition(samples)
        inputs["visual_jitter_score"] = torch.stack([s["visual_jitter_score"] for s in samples]).to(self.device)
        return inputs

    def forward(self, noisy_actions, action_timestep, *, proprio=None, **inputs):
        cond = inputs.pop("lab_condition", None)
        inputs.pop("visual_jitter_score", None)  # future IMU is LOSS-ONLY
        if cond is None:
            raise ValueError("Lab policy requires skill, hand, initial target reference and causal history")
        prepared = cond.get("_prepared_tokens")
        if prepared is None:
            tokens, attention = self.lab_conditioner(**cond)
            token_mask = self.lab_conditioner.context_mask(cond["frame_valid"])
        else:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Prepared tokens cannot bypass training gradients")
            tokens, attention, token_mask = prepared
        self.last_lab_token_count = tokens.shape[1]
        self.last_target_attention = attention.detach()
        context = inputs["context"]
        mask = inputs.get("context_mask")
        if mask is None:
            lens = inputs.get("seq_lens")
            mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device) if lens is None else (
                torch.arange(context.shape[1], device=context.device)[None] < lens[:, None])
        inputs["context"] = torch.cat([context, tokens.to(context.dtype)], dim=1)
        inputs["context_mask"] = torch.cat([mask, token_mask], dim=1)
        # Parent passes this SAME context to vb.prepare AND ab.prepare_state,
        # retaining mutual video/action attention at every MoT layer.
        return super().forward(noisy_actions, action_timestep, proprio=proprio, **inputs)

    def _compute_video_loss(self, prediction, target, timestep_ids, inputs, device):
        scores = inputs.get("visual_jitter_score")
        if scores is None:
            raise ValueError("Missing IMU/orientation jitter supervision")
        scores = latent_jitter(scores, prediction.shape[2], int(self.video_backbone.temporal_compression))
        weight = jitter_weights(scores, self.jitter_scale, self.jitter_floor)
        self.last_jitter_weight = float(weight[:, 1:].mean().detach())
        # Reuse upstream clean-prefix/padding/scheduler logic. Multiplying error
        # by sqrt(w) yields w*MSE; denominator is NEVER sum(jitter weights).
        weighted_prediction = target.float() + (prediction.float()-target.float())*weight[:, None, :, None, None].sqrt()
        return super()._compute_video_loss(weighted_prediction, target, timestep_ids, inputs, device)

    def load_checkpoint(self, path, strict=True):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        if metadata.get("lab_checkpoint") == "delta_v1":
            if metadata.get("conditioner_version") != self.conditioner_version:
                raise ValueError(f"Incompatible lab checkpoint: expected {self.conditioner_version}. "
                                 "Start from the original foundation or a matching manipulation-only delta; "
                                 "older 19D mobile policies cannot resume this 16D policy.")
            foundation = Path(self.foundation_checkpoint)
            if not foundation.is_file():
                raise FileNotFoundError(f"Delta checkpoint requires its foundation weights: {foundation}")
            self._load_foundation(foundation)
            source = load_file(str(path))
            current = self.state_dict()
            unexpected = set(source)-set(current)
            mismatched = [k for k in source if k in current and source[k].shape != current[k].shape]
            expected = set(json.loads(metadata["tensor_names"]))
            if unexpected or mismatched or set(source) != expected:
                raise RuntimeError(f"Invalid delta checkpoint: {unexpected}, {mismatched}")
            required = {k for k in current if k.startswith(("action_backbone.", "proprio_encoder.", "lab_conditioner."))}
            if not required.issubset(source):
                raise RuntimeError(f"Incomplete lab delta: {required-set(source)}")
            self.load_state_dict(source, strict=False)
            self.transfer_report["delta"] = str(path)
            return
        source = load_file(str(path))
        if any(k.startswith("lab_conditioner.") for k in source):
            self.load_state_dict(source, strict=True, assign=any(p.is_meta for p in self.parameters()))
        else:
            self._load_foundation(path, source)

    def _load_foundation(self, path, source=None):
        if not self.transfer_from_foundation:
            raise ValueError("Foundation transfer requires transfer_from_foundation=true")
        source = load_file(str(path)) if source is None else source
        current = self.state_dict()
        heads = ("action_backbone.action_encoder.", "action_backbone.action_decoder.", "proprio_encoder.")
        if source["action_backbone.action_encoder.weight"].shape[1] != 80:
            raise ValueError("Expected original 80D OpenWAM foundation")
        discarded = [k for k in source if k.startswith(heads)]
        for k in discarded:
            del source[k]
        unexpected = set(source)-set(current)
        mismatch = [k for k in source if k in current and source[k].shape != current[k].shape]
        missing = set(current)-set(source)
        allowed = {k for k in current if k.startswith(heads+("lab_conditioner.",))}
        if unexpected or mismatch or missing-allowed:
            raise RuntimeError(f"Unsafe foundation transfer: {unexpected}, {mismatch}, {missing-allowed}")
        self.load_state_dict(source, strict=False, assign=any(p.is_meta for p in self.parameters()))
        if any(p.is_meta for p in self.parameters()):
            raise RuntimeError("Unmaterialized parameters after foundation transfer")
        self.transfer_report = dict(source=str(path), loaded_tensors=len(source),
            reinitialized=sorted(missing), discarded=discarded, action_dim=ACTION_DIM, state_dim=STATE_DIM)

    def save_checkpoint(self, path, *, state_dict=None):
        if not self.delta_checkpoint:
            return super().save_checkpoint(path, state_dict=state_dict)
        if any(p.requires_grad for p in self.video_backbone.parameters()):
            raise ValueError("Delta format requires frozen foundation video backbone; use full checkpoint otherwise")
        current = self.state_dict() if state_dict is None else state_dict
        selected = {k: v.detach().contiguous() for k, v in current.items() if not k.startswith("video_backbone.")}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(selected, str(path), metadata=dict(lab_checkpoint="delta_v1",
            conditioner_version=self.conditioner_version,
            foundation=self.foundation_checkpoint, tensor_names=json.dumps(sorted(selected))))
