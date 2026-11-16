"""8-DoF joint transfer and a MEM-inspired short-history adapter for OpenWAM.

This is NOT the original PaliGemma/SigLIP MEM executor. It uses frozen Wan
image latents, four spatial layers, causal patchwise temporal attention at
layer four, and retains only current-time visual tokens. No language memory.
"""
import json
import math
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.model.video_backbone.wan import encode as wan_encode


class ShortVisualMemory(nn.Module):
    def __init__(self, context_dim=4096, width=256):
        super().__init__()
        self.input = nn.Linear(48, width)
        self.spatial = nn.ModuleList([
            nn.TransformerEncoderLayer(width, 8, width*4, dropout=0., batch_first=True, norm_first=True)
            for _ in range(4)
        ])
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal = nn.MultiheadAttention(width, 8, dropout=0., batch_first=True)
        self.output = nn.Linear(width, context_dim)
        self.state = nn.Linear(8, context_dim)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def position(values, width):
        frequencies = torch.exp(torch.arange(0,width,2,device=values.device).float()*(-math.log(10000.)/width))
        phase = values.float().unsqueeze(-1)*frequencies
        # Zero at the current timestep, matching MEM's relative-time convention.
        return torch.stack((phase.sin(),phase.cos()-1),dim=-1).flatten(-2)

    def forward(self, latents, state, valid):
        b,k,c,h,w = latents.shape
        if k != 4 or c != 48 or not valid[:,-1].all():
            raise ValueError("Memory requires four frames with a valid current observation")
        pooled = F.adaptive_avg_pool2d(latents.flatten(0,1), (6,5))
        x = self.input(pooled.flatten(2).transpose(1,2)).reshape(b,k,30,-1)
        width = x.shape[-1]
        spatial_pos = self.position(torch.arange(30,device=x.device),width).to(x.dtype)
        temporal_pos = self.position(torch.tensor([-2.,-4/3,-2/3,0.],device=x.device),width).to(x.dtype)
        x = x + spatial_pos[None,None] + temporal_pos[None,:,None]
        for layer in self.spatial:
            x = layer(x.flatten(0,1)).reshape(b,k,30,width)
        temporal = x.transpose(1,2).reshape(b*30,k,width)
        causal = torch.ones(k,k,device=x.device,dtype=torch.bool).tril()
        allowed = causal[None] & valid[:,None,:]
        # Padded queries may attend only themselves to avoid all-masked NaNs;
        # padded keys are still invisible to every real query.
        allowed = allowed | ((~valid)[:,:,None] & torch.eye(k,device=x.device,dtype=torch.bool)[None])
        mask = (~allowed)[:,None,None].expand(b,30,8,k,k).reshape(b*30*8,k,k)
        normalized = self.temporal_norm(temporal)
        y,_ = self.temporal(normalized,normalized,normalized,attn_mask=mask,need_weights=False)
        x = (temporal+y).reshape(b,30,k,width)[:,:,-1]
        visual = self.output(x)
        proprio = self.state(state) + self.position(
            torch.tensor([-2.,-4/3,-2/3,0.],device=x.device),visual.shape[-1]
        ).to(visual.dtype)[None]
        tokens = torch.cat([visual,proprio],dim=1)
        token_mask = torch.cat([torch.ones(b,30,device=x.device,dtype=torch.bool),valid],dim=1)
        return tokens,token_mask


@register_architecture("dual_system_right_state", framework="dual_system", variant="right_state")
class RightStateArchitecture(DualSystemSelfAttnArchitecture):
    def __init__(self,cfg=None):
        super().__init__(cfg)
        self.memory_enabled = bool(cfg.get("memory_enabled",False)) if cfg else False
        self.transfer_from_foundation = bool(cfg.get("transfer_from_foundation",False)) if cfg else False
        self.transfer_report = None
        self.short_memory = ShortVisualMemory(self.context_dim) if self.memory_enabled else None
        self.phase_loss_weight = float(cfg.get("gripper_phase_loss_weight", 0.0)) if cfg else 0.0
        self.phase_class_weights = list(cfg.get("gripper_phase_class_weights", [1., 1., 1.])) if cfg else [1., 1., 1.]
        self.gripper_phase_head = None
        self._phase_features = None
        self._capture_phase = False
        self.phase_diagnostics = {}
        if self.phase_loss_weight > 0:
            if self.action_dim != 8:
                raise ValueError("Gripper phase auxiliary head requires right8 state")
            self.gripper_phase_head = nn.Linear(self.action_backbone.action_decoder.in_features, 3)
            # Share the exact final action-token features; do not feed future
            # phase labels to forward(), memory, proprio, or inference.
            self.action_backbone.action_decoder.register_forward_pre_hook(self._capture_phase_features)

    def _capture_phase_features(self, module, args):
        if self._capture_phase:
            self._phase_features = args[0]

    def compute_loss(self, **inputs):
        labels = inputs.pop("gripper_phase", None)
        self.phase_diagnostics = {}
        if self.gripper_phase_head is None:
            return super().compute_loss(**inputs)
        if labels is None:
            raise ValueError("Phase auxiliary training requires achieved-state phase labels")
        self._capture_phase = True
        self._phase_features = None
        try:
            result = super().compute_loss(**inputs)
            if self._phase_features is None:
                raise RuntimeError("Action features did not reach the phase head")
            logits = self.gripper_phase_head(self._phase_features).float()
            labels = labels.to(device=logits.device, dtype=torch.long)
            if logits.shape[:-1] != labels.shape:
                raise ValueError(f"Phase shapes differ: {logits.shape}, {labels.shape}")
            weights = torch.tensor(self.phase_class_weights, device=logits.device, dtype=torch.float32)
            phase_loss = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), weight=weights)
            result["loss"] = result["loss"] + self.phase_loss_weight * phase_loss
            pred = logits.detach().argmax(-1)
            diagnostics = {"phase_ce": phase_loss.detach(),
                           "phase_weighted": (self.phase_loss_weight * phase_loss).detach(),
                           "phase_accuracy": (pred == labels).float().mean()}
            for k, name in enumerate(("open", "transition", "closed")):
                mask = labels == k
                diagnostics[f"phase_{name}_correct"] = ((pred == labels) & mask).float().sum()
                diagnostics[f"phase_{name}_count"] = mask.float().sum()
            self.phase_diagnostics = diagnostics
            return result
        finally:
            self._capture_phase = False
            self._phase_features = None

    def train(self,mode=True):
        super().train(mode)
        video = getattr(self,"video_backbone",None)
        if video is not None and not any(p.requires_grad for p in video.parameters()):
            # DeepSpeed and validation both call architecture.train(). Keep the
            # frozen T5/VAE/DiT deterministic rather than re-enabling dropout.
            for module in video.modules():
                module.training = False
        return self

    def set_dtype_device(self,dtype,device):
        super().set_dtype_device(dtype,device)
        if self.short_memory is not None:
            self.short_memory.to(device=device,dtype=dtype)
        if self.gripper_phase_head is not None:
            self.gripper_phase_head.to(device=device,dtype=dtype)

    def load_checkpoint(self,path,strict=True):
        source = load_file(path)
        current = self.state_dict()
        head_modules = ("action_backbone.action_encoder.","action_backbone.action_decoder.","proprio_encoder.")
        source_dim = source["action_backbone.action_encoder.weight"].shape[1]
        foundation = source_dim == 80 and self.action_dim == 8
        if foundation and not self.transfer_from_foundation:
            raise ValueError("80->8 joint transfer must be explicitly enabled")
        skipped = []
        if foundation:
            for key in list(source):
                if key.startswith(head_modules):
                    skipped.append(key)
                    del source[key]
        unexpected = set(source)-set(current)
        mismatched = [k for k in source if k in current and source[k].shape != current[k].shape]
        missing = set(current)-set(source)
        allowed = set(k for k in current if foundation and (k.startswith(head_modules) or k.startswith("short_memory.") or k.startswith("gripper_phase_head.")))
        if unexpected or mismatched or missing-allowed:
            raise RuntimeError(f"Unsafe transfer: unexpected={unexpected}, mismatched={mismatched}, missing={missing-allowed}")
        self.load_state_dict(source,strict=False,assign=any(p.is_meta for p in self.parameters()))
        if any(p.is_meta for p in self.parameters()):
            raise RuntimeError("Unmaterialized weights after transfer")
        self.transfer_report = dict(source=str(path),source_action_dim=int(source_dim),target_action_dim=self.action_dim,
                                    loaded_tensors=len(source),reinitialized=sorted(missing),discarded=sorted(skipped))
        print("[right-state transfer] "+json.dumps(self.transfer_report),flush=True)

    @torch.no_grad()
    def encode_memory(self,samples):
        vb = self.video_backbone
        all_latents = []
        for sample in samples:
            images = sample["memory_images"]
            latents = []
            # Encode each timestamp independently: no future or temporal VAE leakage.
            for image in images:
                pixels = wan_encode.preprocess_video([image],dtype=self.dtype,device=self.device)
                latent = wan_encode.encode_video(pixels,vae=vb.vae,encoder=vb.video_encoder)
                latents.append(latent[:,:,0])
            all_latents.append(torch.stack(latents,dim=1))
        return dict(memory_latents=torch.cat(all_latents).to(self.dtype),
                    memory_state=torch.stack([x["memory_state"] for x in samples]).to(self.device,self.dtype),
                    memory_valid=torch.stack([x["memory_valid"] for x in samples]).to(self.device))

    @torch.no_grad()
    def prepare_inputs(self,batch):
        samples = [batch] if isinstance(batch,dict) else batch
        inputs = super().prepare_inputs(samples)
        if self.memory_enabled:
            inputs.update(self.encode_memory(samples))
        if self.gripper_phase_head is not None:
            inputs["gripper_phase"] = torch.stack([x["gripper_phase"] for x in samples])
        return inputs

    def forward(self,noisy_actions,action_timestep,*,proprio=None,**inputs):
        latents = inputs.pop("memory_latents",None)
        states = inputs.pop("memory_state",None)
        valid = inputs.pop("memory_valid",None)
        if self.memory_enabled:
            if latents is None or states is None or valid is None:
                raise ValueError("MEM model requires historical images/state at BOTH train and inference")
            tokens, token_mask = self.short_memory(latents,states,valid)
            context = inputs["context"]
            mask = inputs.get("context_mask")
            if mask is None:
                mask = torch.arange(context.shape[1],device=context.device)[None] < inputs["seq_lens"][:,None]
            inputs["context"] = torch.cat([context,tokens],dim=1)
            inputs["context_mask"] = torch.cat([mask,token_mask],dim=1)
        return super().forward(noisy_actions,action_timestep,proprio=proprio,**inputs)
