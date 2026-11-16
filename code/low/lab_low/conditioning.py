"""Skill/hand FiLM, causal patch memory, then initial-reference target readout.

All queries are shared across skills. Marks bias attention, never erase pixels
or constrain attention to a hard ROI. Memory uses independently encoded past
images; no future video latent is accepted by this module.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from lab_low.visual_memory import SpatialTemporalMemory, temporal_position
from lab_low.skills import ACTION_DIM, STATE_DIM, SKILLS, HANDS


class LabConditioner(nn.Module):
    def __init__(self, latent_channels=48, context_dim=4096, width=256, roles=2, queries=2,
                 prior_strength=2., prior_floor=.05, memory_depth=4, temporal_every=4):
        super().__init__()
        self.roles, self.queries = roles, queries
        self.prior_strength, self.prior_floor = prior_strength, prior_floor
        self.visual = nn.Linear(latent_channels, width)
        self.xy = nn.Linear(2, width)
        self.skill = nn.Embedding(len(SKILLS), width)
        self.hand = nn.Embedding(len(HANDS), width)
        self.film = nn.Linear(width, 2 * width)
        self.norm = nn.LayerNorm(width)
        self.keys = nn.Linear(width, width)
        self.values = nn.Linear(width, width)
        self.query = nn.Parameter(torch.randn(roles, queries, width) / math.sqrt(width))
        self.role = nn.Embedding(roles, width)
        self.mark_state = nn.Linear(2, width)
        self.state_action = nn.Linear(STATE_DIM + ACTION_DIM + 1, width)
        self.visual_memory = SpatialTemporalMemory(width, depth=memory_depth, temporal_every=temporal_every)
        self.output = nn.Linear(width, context_dim)
        nn.init.normal_(self.output.weight, std=.005)
        nn.init.zeros_(self.output.bias)

    def context_mask(self, frame_valid):
        # skill/hand + target queries + global token + one state token per rolling frame
        fixed = torch.ones(frame_valid.shape[0], 2+self.roles*self.queries,
                           device=frame_valid.device, dtype=torch.bool)
        return torch.cat([fixed, frame_valid[:, 1:]], dim=1)

    def forward(self, latents, prior, mark_valid, confidence, states, previous_actions,
                previous_action_valid, frame_valid, relative_time, skill_id, hand_id):
        b, k, c, h, w = latents.shape
        if k < 2 or not frame_valid[:, 0].all() or not frame_valid[:, -1].all() or (relative_time > 0).any():
            raise ValueError("History must end at the valid current frame, never in the future")
        if mark_valid[:, 1:].any():
            raise ValueError("Only the persistent reference may carry target annotations")
        x = self.visual(latents.permute(0, 1, 3, 4, 2).reshape(b, k, h*w, c))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        xy = torch.stack([xx, yy], -1).reshape(h*w, 2).to(x.dtype)
        cond = self.skill(skill_id) + self.hand(hand_id)
        scale, shift = self.film(cond).chunk(2, -1)
        x = self.norm(x) * (1 + .1 * torch.tanh(scale[:, None, None])) + shift[:, None, None]
        # Reference and rolling window use the same weights but NEVER attend to
        # each other here. The reference remains a fixed identity anchor.
        reference = self.visual_memory(x[:, :1], frame_valid[:, :1],
                                       torch.zeros_like(relative_time[:, :1]))[:, 0]
        rolling = self.visual_memory(x[:, 1:], frame_valid[:, 1:], relative_time[:, 1:])
        current = rolling[:, -1]  # compress visual history BEFORE target pooling
        # Identity matching uses content; only the values carry current spatial
        # coordinates. An initial image location must not pin subsequent search
        # to that same coordinate when the humanoid/camera moves.
        reference_key = self.keys(reference)
        key = self.keys(current)
        spatial_x = current + self.xy(xy)[None]
        value = self.values(spatial_x)
        p = F.adaptive_avg_pool2d(prior.flatten(0, 2).unsqueeze(1).float(), (h, w))
        p = p.reshape(b, k, self.roles, h*w).clamp(0, 1)
        # Missing marks are a distinct mode, never a synthetic full-image box.
        p = torch.where(mark_valid[..., None], p, torch.zeros_like(p))
        confidence = torch.where(mark_valid, confidence, torch.zeros_like(confidence))
        reliability = mark_valid.float() * confidence.float().clamp(0, 1)
        # Frame zero is the persistent, initially marked reference image, not
        # the oldest rolling frame. Pool inside the soft mark to bind identity.
        # The identity query then searches ALL current patches as the camera moves.
        # A soft background floor makes imperfect boxes a prior, not a hard crop.
        anchor_weight = (self.prior_floor + (1-self.prior_floor)*p[:, 0]).pow(self.prior_strength)
        anchor_weight = anchor_weight / anchor_weight.sum(-1, keepdim=True).clamp_min(1e-6)
        anchor = torch.einsum("brn,bnd->brd", anchor_weight.to(x.dtype), reference_key)
        anchor = torch.where(mark_valid[:, 0, :, None],
                             anchor * reliability[:, 0, :, None].to(x.dtype), torch.zeros_like(anchor))
        query = self.query[None] + anchor[:, :, None]
        logits = torch.einsum("brqd,bnd->brqn", query, key) / math.sqrt(x.shape[-1])
        attention = logits.float().softmax(-1).to(x.dtype)
        targets = torch.einsum("brqn,bnd->brqd", attention, value)
        status = self.mark_state(torch.stack([mark_valid[:, 0], confidence[:, 0]], -1).to(x.dtype))
        targets = targets + self.role.weight[None, :, None] + status[:, :, None]
        targets = targets.flatten(1, 2)
        global_token = spatial_x.mean(1, keepdim=True)
        sa = torch.cat([states[:, 1:], previous_actions[:, 1:] * previous_action_valid[:, 1:, None],
                        previous_action_valid[:, 1:, None]], -1).to(x.dtype)
        state_tokens = self.state_action(sa) + temporal_position(relative_time[:, 1:], x.shape[-1], x.dtype)
        output = torch.cat([cond[:, None], targets, global_token, state_tokens], dim=1)
        output = self.output(output).masked_fill(~self.context_mask(frame_valid)[..., None], 0)
        return output, attention
