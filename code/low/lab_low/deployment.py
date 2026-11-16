"""Persistent manipulation inference session; no robot actuator transport.

RGB images and achieved state arrive together at the training's 30 Hz cadence.
New bindings clear all history. A failed reset invalidates the previous session.
"""
import base64
from io import BytesIO
import time

import numpy as np
from PIL import Image
import torch

from lab_low.dataset import CAMERAS
from lab_low.runtime import OnlineConditionBuffer
from lab_low.inference_optimizations import InferenceOptimizations, ImageLatentCache


def decode_images(packet):
    views = packet["images"]
    if not isinstance(views, dict) or set(views) != set(CAMERAS):
        raise ValueError(f"Expected RGB camera keys: {CAMERAS}")
    images = []
    for name in CAMERAS:
        data = base64.b64decode(views[name], validate=True)
        with Image.open(BytesIO(data)) as image:
            images.append(image.convert("RGB"))
    return images


class InferenceSession:
    def __init__(self, model, stats, schedule, steps=20, allowed_pairs=None, optimizations=None):
        if getattr(model, "conditioner_version", None) != "spacetime_manip_v3":
            raise ValueError("Expected spacetime_manip_v3 checkpoint")
        if model.normalizer is not None:
            raise ValueError("Disable the shared OpenWAM normalizer")
        self.model, self.schedule, self.steps = model, schedule, steps
        self.buffer = OnlineConditionBuffer(stats)
        self.prompt_cache = {}
        self.binding = None
        self.last_frame = None
        self.allowed_pairs = None if allowed_pairs is None else {tuple(pair) for pair in allowed_pairs}
        self.optimizations = optimizations or InferenceOptimizations()
        self.latent_cache = ImageLatentCache()
        self.dit_cache = self.optimizations.new_dit_cache()

    def stop(self):
        self.latent_cache.clear()
        if self.dit_cache is not None:
            self.dit_cache.reset()
        self.binding = None
        self.last_frame = None
        self.buffer.reference = None
        self.buffer.frames.clear()

    def _observation(self, packet):
        frame = packet["frame_index"]
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("frame_index must be a nonnegative 30 Hz integer")
        return dict(frame_index=frame, state=packet["state"], images=decode_images(packet))

    def reset(self, packet):
        self.stop()
        if self.allowed_pairs is not None and (packet["skill"], packet["hand"]) not in self.allowed_pairs:
            raise ValueError("This checkpoint was not trained for the requested skill/hand pair")
        binding = packet["binding_id"]
        if not isinstance(binding, str) or not binding.strip():
            raise ValueError("A nonempty binding_id is required")
        targets = packet.get("targets", [])
        if not isinstance(targets, list) or any("mask" in target for target in targets):
            raise ValueError("Deployment protocol accepts initial pixel boxes, not file paths")
        obs = self._observation(packet)
        reference_images = None if "reference_images" not in packet else decode_images({"images": packet["reference_images"]})
        self.buffer.reset(skill=packet["skill"], hand=packet["hand"],
                          binding_id=binding, targets=targets, reference_images=reference_images,
                          prompt=packet.get("prompt"), **obs)
        self.binding, self.last_frame = binding, obs["frame_index"]
        return {"reset": True, "binding_id": self.binding, "frame_index": self.last_frame,
                "reference_images_used": reference_images is not None, "prompt": self.buffer.prompt}

    def _require_binding(self, packet):
        if self.binding is None or packet.get("binding_id") != self.binding:
            raise ValueError("Reset required, or binding_id does not match")

    def observe(self, packet):
        self._require_binding(packet)
        if any(key in packet for key in ("targets", "skill", "hand", "prompt", "reference_images")):
            raise ValueError("Skill, hand and targets are reset-only fields")
        obs = self._observation(packet)
        if obs["frame_index"] != self.last_frame + 1:
            raise ValueError("Supply every 30 Hz observation in consecutive frame order")
        self.buffer.observe(**obs, return_sample=False)
        self.last_frame = obs["frame_index"]
        return {"observed": True, "frame_index": self.last_frame}

    @torch.inference_mode()
    def predict(self, packet):
        self._require_binding(packet)
        started = time.monotonic()
        steps = packet.get("denoise_steps",self.steps)
        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=100:
            raise ValueError("denoise_steps must be an integer in 1..100")
        # Both grids and scheduler internals must agree after a caller switches
        # step count; requests are serialized by the model worker.
        from openwam.deploy.denoise_schedule import schedule_sync
        schedule = schedule_sync(self.model.video_scheduler,self.model.action_scheduler,steps,5.)
        sample = self.buffer.sample()
        model = self.model
        if self.optimizations.condition_cache:
            condition = model.prepare_condition([sample], latent_cache=self.latent_cache)
            condition = model.prepare_inference_tokens(condition)
        else:
            condition = model.prepare_condition([sample])
        if self.dit_cache is not None:
            self.dit_cache.reset()  # velocity predictions are request-local
        result = model.generate(
            schedule, sample["prompt"], first_frame_image=sample["first_frame_image"],
            num_frames=9, action_num_frames=33, width=320, height=384,
            num_inference_steps=steps, shift=5., seed=42 + self.last_frame,
            decode_video=False, lab_condition=condition,
            proprio=sample["proprio"].to(model.device, model.dtype).unsqueeze(0),
            active_action_mask=torch.as_tensor(self.buffer.active, device=model.device),
            prompt_embed_cache=self.prompt_cache, dit_cache=self.dit_cache)
        actions = result["actions"]
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        actions = self.buffer.decode_and_gate(actions)
        if actions.shape != (32, 16) or not np.isfinite(actions).all():
            raise ValueError("Invalid action chunk")
        return {"binding_id": self.binding, "observation_frame": self.last_frame,
                "denoise_steps": steps,
                "actions": actions.tolist(), "action_hz": 30,
                "active_action_mask": self.buffer.active.tolist(),
                "inference_seconds": time.monotonic() - started,
                "optimizations": {**self.optimizations.metadata(),
                    "compiled_active": getattr(model, "_compiled_mot_run_joint_loop", None) is not None,
                    "latent_cache_hits": self.latent_cache.hits,
                    "latent_cache_misses": self.latent_cache.misses,
                    "dit_cache_stats": None if self.dit_cache is None else self.dit_cache.stats},
                "execution_allowed": False, "completion": None}

    def request(self, packet):
        op = packet.get("op")
        if op == "reset":
            return self.reset(packet)
        if op == "observe":
            return self.observe(packet)
        if op == "observe_batch":
            self._require_binding(packet)
            observations = packet.get("observations", [])
            if not isinstance(observations, list) or not observations:
                raise ValueError("Expected a nonempty list of sequential observations")
            for observation in observations:
                self.observe({**observation, "binding_id": self.binding})
            return {"observed": len(observations), "frame_index": self.last_frame}
        if op == "predict":
            return self.predict(packet)
        if op == "stop":
            self._require_binding(packet)
            self.stop()
            return {"stopped": True}
        raise ValueError(f"Unknown operation: {op}")
