"""Right-only achieved-state supervision; two real views, episode-held-out split."""
from pathlib import Path
import json
import random

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageEnhance, ImageOps

from openwam.dataloader.bases import BaseDataset

RIGHT = [22, 23, 24, 25, 26, 27, 28, 30]
CAMERAS = ["observation.images.ego_view", "observation.images.right_wrist"]


class SonicRightStateDataset(BaseDataset):
    def __init__(self, config, split="train"):
        self.config, self.split = config, split
        self.root = Path(config.dataset_dir)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        assert self.info["features"]["observation.state"]["shape"] == [31]
        names = self.info["features"]["observation.state"]["names"]
        expected = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint", "right_hand_dex1_motor"]
        if [names[i] for i in RIGHT] != expected:
            raise ValueError("Right joint ordering does not match this dataset contract")
        assert self.info["fps"] == 30
        self.num_frames, self.video_stride = 33, 4
        self.action_dim = self.state_dim = 8
        self.epoch = 0
        rows = [json.loads(s) for s in (self.root / "meta/episodes.jsonl").read_text().splitlines()]
        # Explicit opt-in preserves reproducibility of earlier saved configurations.
        self.excluded_ids = sorted(int(x) for x in self.info.get("discarded_episode_indices", [])) if config.get("exclude_discarded", False) else []
        rows = [x for x in rows if int(x["episode_index"]) not in self.excluded_ids]
        # Explicit subset for smoke tests; production config leaves this unset.
        if config.get("episode_ids") is not None:
            requested = set(int(x) for x in config.episode_ids)
            rows = [x for x in rows if int(x["episode_index"]) in requested]
            if {int(x["episode_index"]) for x in rows} != requested:
                raise ValueError("Requested smoke-test episodes are missing")
        ids = sorted(int(x["episode_index"]) for x in rows)
        if not 0 < int(config.val_episodes) < len(ids):
            raise ValueError("Need nonempty train and validation episode sets after filtering")
        permutation = np.random.default_rng(int(config.seed)).permutation(ids)
        self.val_ids = sorted(int(x) for x in permutation[:int(config.val_episodes)])
        self.train_ids = sorted(set(ids) - set(self.val_ids))
        chosen = self.train_ids if split == "train" else self.val_ids
        if split not in ("train", "val"):
            raise ValueError(split)
        self.episodes, self.windows, training_states = [], [], []
        for row in rows:
            ep = int(row["episode_index"])
            fields = dict(episode_index=ep, episode_chunk=ep // self.info["chunks_size"])
            path = self.root / self.info["data_path"].format(**fields)
            # Deliberately never read action.wbc or any left-arm/left-gripper values.
            table = pq.read_table(path, columns=["observation.state", "timestamp"])
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[:, RIGHT]
            ts = np.asarray(table["timestamp"].to_pylist())
            if states.shape != (row["length"], 8) or not np.isfinite(states).all():
                raise ValueError(f"Invalid right state: {path}")
            if not np.allclose(np.diff(ts), 1/30, atol=1e-4):
                raise ValueError(f"Nonuniform timestamps: {path}")
            if ep in self.train_ids:
                training_states.append(states)
            if ep not in chosen:
                continue
            videos = {}
            for key in CAMERAS:
                vp = self.root / self.info["video_path"].format(video_key=key, **fields)
                cap = cv2.VideoCapture(str(vp))
                count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if count != len(states) or abs(fps - 30) > 0.1:
                    raise ValueError(f"Video alignment mismatch: {vp}, {count}, {fps}")
                videos[key] = vp
            ei = len(self.episodes)
            self.episodes.append(dict(id=ep, state=states, path=str(path), videos=videos))
            self.windows.extend((ei, t) for t in range(0, len(states)-32, int(config.window_stride)))
        values = np.concatenate(training_states)
        self.lo, self.hi = values.min(0), values.max(0)
        self.stats = dict(min=self.lo.tolist(), max=self.hi.tolist(), constant=(self.hi-self.lo < 1e-6).tolist())
        self.normalization_stats_path = str(Path(config.artifact_dir) / "normalization_stats.npy")
        if split == "train":
            Path(config.artifact_dir).mkdir(parents=True, exist_ok=True)
            np.save(self.normalization_stats_path, {
                "wbc": self.stats, "wbc_state": self.stats,
                "source_action_key": "observation.state", "source_state_key": "observation.state",
                "indices": RIGHT, "units": "rad", "absolute": True,
                "alignment": "state[t] -> state[t+1:t+33]",
                "train_episode_ids": self.train_ids, "val_episode_ids": self.val_ids,
                "excluded_episode_ids": self.excluded_ids,
            })
        if not self.windows:
            raise ValueError("No complete 32-step windows")
        self.phase_enabled = bool(config.get("gripper_phase_enabled", False))
        self.phase_thresholds = (float(config.get("gripper_closed_threshold", 1.2)),
                                 float(config.get("gripper_open_threshold", 5.0)))
        if not self.phase_thresholds[0] < self.phase_thresholds[1]:
            raise ValueError("Gripper phase thresholds must be ordered in native radians")
        # Fit auxiliary class weights only on training states, never validation.
        counts = np.bincount(self.gripper_phases(values[:, -1]), minlength=3)
        self.phase_class_counts = counts.tolist()
        weights = np.sqrt(len(values) / (3 * np.maximum(counts, 1)))
        self.phase_class_weights = (weights / weights.mean()).tolist()

    def gripper_phases(self, angles):
        """Achieved-state regions, NOT the unobserved discrete command or intent.

        0=open, 1=intermediate, 2=closed. In this close-and-hold dataset the
        intermediate region is the closing transition; reopening would share it.
        """
        closed, opened = self.phase_thresholds
        angles = np.asarray(angles)
        return np.where(angles >= opened, 0, np.where(angles <= closed, 2, 1)).astype(np.int64)

    @classmethod
    def from_config(cls, config, split="train"):
        return cls(config, split)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.windows)

    def normalize(self, a):
        return np.clip(2*(a-self.lo)/np.maximum(self.hi-self.lo, 1e-6)-1, -1, 1).astype(np.float32)

    def denormalize_action(self, a):
        return (np.asarray(a)+1)*0.5*(self.hi-self.lo)+self.lo

    def _augment_parameters(self, index):
        # Independent of model RNG: paired experiments get identical augmentation.
        rng = random.Random(int(self.config.seed) + self.epoch*1000003 + index)
        out = {}
        for key in CAMERAS:
            if self.split != "train":
                out[key] = (1., 1., 1., None)
                continue
            crop = None
            if key == CAMERAS[0]:
                # openpi-style 95% ego crop; leave wrist geometry intact.
                crop = (rng.uniform(0, .05), rng.uniform(0, .05), .95)
            out[key] = (rng.uniform(.7,1.3), rng.uniform(.6,1.4), rng.uniform(.5,1.5), crop)
        return out

    @staticmethod
    def _image_transform(im, parameters):
        brightness, contrast, saturation, crop = parameters
        if crop is not None:
            x, y, scale = crop
            w, h = im.size
            im = im.crop((round(x*w), round(y*h), round((x+scale)*w), round((y+scale)*h)))
        im = ImageEnhance.Brightness(im).enhance(brightness)
        im = ImageEnhance.Contrast(im).enhance(contrast)
        return ImageEnhance.Color(im).enhance(saturation)

    def __getitem__(self, index):
        ei, t = self.windows[index]
        ep = self.episodes[ei]
        future = list(range(t, t+33, 4))
        raw_history = [t-60, t-40, t-20, t] if self.config.memory_enabled else []
        history = [max(0,x) for x in raw_history]
        requested = sorted(set(future + history))
        frames = {i: {} for i in requested}
        params = self._augment_parameters(index)
        for key, vp in ep["videos"].items():
            cap = cv2.VideoCapture(str(vp))
            cap.set(cv2.CAP_PROP_POS_FRAMES, requested[0])
            try:
                for pos in range(requested[0], requested[-1]+1):
                    ok, arr = cap.read()
                    if not ok:
                        raise RuntimeError(f"Decode failed {vp}:{pos}")
                    if pos in frames:
                        im = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                        frames[pos][key] = self._image_transform(im, params[key])
            finally:
                cap.release()
        def assemble(pos):
            # Two equal-height real views, aspect ratio preserved, no left/duplicate view.
            canvas = Image.new("RGB", (320,384), (0,0,0))
            for i,key in enumerate(CAMERAS):
                tile = ImageOps.pad(frames[pos][key], (320,192), method=Image.Resampling.BILINEAR)
                canvas.paste(tile, (0,i*192))
            return canvas
        video = [assemble(i) for i in future]
        action = self.normalize(ep["state"][t+1:t+33])
        state = self.normalize(ep["state"][t:t+1])
        result = dict(video=video, first_frame_image=[video[0]], vace_video=None,
                      action=torch.from_numpy(action), action_mask=torch.ones((32,8),dtype=torch.bool),
                      proprio=torch.from_numpy(state), proprio_mask=torch.ones((1,8),dtype=torch.bool),
                      video_mask=torch.ones(9,dtype=torch.bool), prompt=str(self.config.prompt),
                      episode_index=ep["id"], episode_path=ep["path"], start_frame=t,
                      end_frame=t+33, episode_length=len(ep["state"]), active_arm="right",
                      task_name="pick_up_test_tube")
        if self.config.memory_enabled:
            result.update(memory_images=[assemble(i) for i in history],
                          memory_state=torch.from_numpy(self.normalize(ep["state"][history])),
                          memory_valid=torch.tensor([i >= 0 for i in raw_history]),
                          memory_frame_indices=history)
        if self.phase_enabled:
            result["gripper_phase"] = torch.from_numpy(self.gripper_phases(ep["state"][t+1:t+33, -1]))
        return result
