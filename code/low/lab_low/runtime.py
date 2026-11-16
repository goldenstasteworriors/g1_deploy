"""Online input buffer matching training: one initial mark, then observations.

No robot transport or actuator calls. The controller owns skill completion,
water-bath release, balance and command-rate limits.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lab_low.dataset import CAMERAS, compose, normalize, reference_prior
from lab_low.skills import ARM_INDICES, ACTION_DIM, STATE_DIM, CONTRACT_VERSION, action_mask, condition, gate_action, validate_targets


class OnlineConditionBuffer:
    """30 Hz frame-index contract. Reset on every new skill or target binding.

    `images` is three original PIL RGB views in CAMERAS order, or a dict using
    those names. Supply observations only; no action labels are required.
    """
    def __init__(self, normalization_stats, width=320, height=384, history_offsets=(-60,-40,-20)):
        stats = normalization_stats
        if isinstance(stats, (str, Path)):
            stats = np.load(stats, allow_pickle=True).item()
        self.state_stats, self.action_stats = stats["wbc_state"], stats["wbc"]
        if any(len(self.state_stats[key]) != STATE_DIM or len(self.action_stats[key]) != ACTION_DIM
               for key in ("min", "max")):
            raise ValueError("Expected lab 31D state / 16D manipulation-only action normalization")
        if stats.get("contract_version") != CONTRACT_VERSION:
            raise ValueError(f"Normalization must use the {CONTRACT_VERSION} contract")
        self.width, self.height = width, height
        self.offsets = list(history_offsets)
        if self.offsets != sorted(set(self.offsets)) or any(o >= 0 for o in self.offsets):
            raise ValueError("History offsets must be sorted unique negative frame offsets")
        self.reference = None
        self.frames = OrderedDict()

    @staticmethod
    def packet(frame_index, images, state):
        images = [images[c] for c in CAMERAS] if isinstance(images, dict) else list(images)
        if len(images) != 3 or any(not isinstance(im, Image.Image) for im in images):
            raise ValueError("Three original PIL camera images are required")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (31,) or not np.isfinite(state).all():
            raise ValueError("Expected finite observation.state[31]")
        return dict(index=int(frame_index), images=[im.convert("RGB").copy() for im in images],
                    state=state.copy())

    def reset(self, *, skill, hand, binding_id, frame_index, images, state, targets=None,
              annotation_dir=".", reference_images=None, prompt=None):
        # An invalid new binding must not leave the old target executable.
        self.reference = None
        self.frames.clear()
        self.sid, self.hid, self.prompt = condition(skill, hand)
        if prompt is not None and prompt != self.prompt:
            raise ValueError("Prompt must match the trained skill/hand instruction")
        self.skill, self.hand, self.binding = skill, hand, str(binding_id)
        self.active = action_mask(skill, hand)
        targets = validate_targets(skill, targets)
        current = self.packet(frame_index, images, state)
        # Hybrid deployment can retain the original annotated target image
        # while starting current vision later in the recording. Both states
        # are the live state at binding time; no recorded past state is used.
        reference = current if reference_images is None else self.packet(frame_index, reference_images, state)
        prior = np.zeros((2,self.height,self.width), dtype=np.float32)
        valid, confidence = np.zeros(2,bool), np.zeros(2,np.float32)
        for mark in targets:
            role = int(mark.get("role",0))
            if role not in (0,1) or valid[role]:
                raise ValueError("Invalid/duplicate target role")
            camera = mark.get("camera",CAMERAS[0])
            ci = CAMERAS.index(camera)
            score = float(mark.get("confidence",.5))
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Initial confidence must be in [0,1]")
            tiles = [Image.new("L",im.size,0) for im in reference["images"]]
            tiles[ci] = reference_prior(mark,reference["images"][ci].size,Path(annotation_dir))
            prior[role] = np.asarray(compose(tiles,self.width,self.height),dtype=np.float32)/255
            valid[role],confidence[role] = True,score
        self.prior,self.mark_valid,self.confidence = prior,valid,confidence
        self.reference = reference
        self.frames.clear()
        self.frames[current["index"]] = current

    def observe(self, *, frame_index, images, state, return_sample=True):
        if self.reference is None:
            raise RuntimeError("Call reset with the skill conditions before observe")
        if int(frame_index) <= next(reversed(self.frames)):
            raise ValueError("Observations must have strictly increasing frame indices")
        packet = self.packet(frame_index, images, state)
        self.frames[packet["index"]] = packet
        # Preserve enough history AND the preceding state for achieved deltas.
        oldest = packet["index"] + min(self.offsets, default=0) - 1
        while next(iter(self.frames)) < oldest:
            self.frames.popitem(last=False)
        return self.sample() if return_sample else None

    def sample(self):
        if self.reference is None:
            raise RuntimeError("No active skill binding; call reset first")
        current = self.frames[next(reversed(self.frames))]
        t,start = current["index"],self.reference["index"]
        packets,valid = [self.reference],[True]
        for offset in self.offsets:
            index = t+offset
            packet = self.frames.get(index)
            packets.append(self.reference if packet is None else packet)
            valid.append(packet is not None and index >= start)
        packets.append(current)
        valid.append(True)
        indices = [p["index"] for p in packets]
        k=len(packets)
        motion = np.zeros((k,ACTION_DIM),np.float32)
        motion_valid = np.zeros(k,bool)
        for i,packet in enumerate(packets):
            previous = self.frames.get(packet["index"]-1)
            if valid[i] and packet["index"] > start and previous is not None:
                motion_valid[i] = True
                motion[i] = packet["state"][ARM_INDICES]-previous["state"][ARM_INDICES]
        motion[:,~self.active] = 0
        prior=np.zeros((k,2,self.height,self.width),np.float32)
        marks=np.zeros((k,2),bool)
        confidence=np.zeros((k,2),np.float32)
        prior[0],marks[0],confidence[0]=self.prior,self.mark_valid,self.confidence
        return dict(skill_id=self.sid,hand_id=self.hid,prompt=self.prompt,binding_id=self.binding,
            reference_frame=start,start_frame=t,history_indices=indices,
            condition_images=[compose(p["images"],self.width,self.height) for p in packets],
            target_prior=torch.from_numpy(prior),mark_valid=torch.from_numpy(marks),
            mark_confidence=torch.from_numpy(confidence),
            history_state=torch.from_numpy(normalize(np.stack([p["state"] for p in packets]),self.state_stats)),
            previous_motion=torch.from_numpy(motion),previous_motion_valid=torch.from_numpy(motion_valid),
            history_valid=torch.tensor(valid,dtype=torch.bool),
            relative_time=torch.tensor([(i-t)/30 for i in indices],dtype=torch.float32),
            proprio=torch.from_numpy(normalize(current["state"],self.state_stats)),
            first_frame_image=[compose(current["images"],self.width,self.height)])

    def decode_and_gate(self, normalized_actions):
        """Use when architecture.normalizer is None; do not denormalize twice."""
        lo,hi=np.asarray(self.action_stats["min"]),np.asarray(self.action_stats["max"])
        raw=(np.asarray(normalized_actions)+1)*.5*(hi-lo)+lo
        current=self.frames[next(reversed(self.frames))]
        return gate_action(raw,current["state"],self.skill,self.hand)
