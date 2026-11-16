"""Read real LeRobot observations, single initial target mark and causal history.

No action.wbc / teleop command is read. Each manifest row denotes a skill
segment; chunks and rolling history never cross its start/end. Initial target
reference persists across every window, including windows > history length.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageFilter, ImageOps

from openwam.dataloader.bases import BaseDataset
from lab_low.jitter import orientation_speed
from lab_low.photometric import sample_lighting, apply_lighting
from lab_low.augmentations import perturb_box, sample_camera, apply_camera
from lab_low.skills import ARM_INDICES, ACTION_DIM, STATE_DIM, CONTRACT_VERSION, action_mask, condition, validate_targets

CAMERAS = ["observation.images.ego_view", "observation.images.left_wrist", "observation.images.right_wrist"]


def finite_array(value, width, name):
    a = np.asarray(value, dtype=np.float32)
    if a.ndim != 2 or a.shape[-1] != width or not np.isfinite(a).all():
        raise ValueError(f"Invalid {name}: expected finite [T,{width}], got {a.shape}")
    return a


def normalize(value, stats):
    lo, hi = np.asarray(stats["min"]), np.asarray(stats["max"])
    result = 2*(value-lo)/np.maximum(hi-lo, 1e-6)-1
    return np.where(hi-lo < 1e-6, 0, result).clip(-1, 1).astype(np.float32)


def letterbox(image, size, is_mask=False):
    return ImageOps.pad(image, size, method=Image.Resampling.BILINEAR, color=0)


def compose(images, width, height):
    """Large ego view above two wrists; same geometry for pixels and marks."""
    canvas = Image.new(images[0].mode, (width, height), 0)
    top = 2*height//3
    canvas.paste(letterbox(images[0], (width, top)), (0, 0))
    for j, image in enumerate(images[1:]):
        canvas.paste(letterbox(image, (width//2, height-top)), (j*width//2, top))
    return canvas


def reference_prior(mark, image_size, annotation_dir, perturb=None, box_config=None):
    """Pixel xyxy box or mask in ORIGINAL camera coordinates; no red overlays."""
    w, h = image_size
    if "mask" in mark:
        path = (annotation_dir / mark["mask"]).resolve()
        image = Image.open(path).convert("L")
        if image.size != image_size:
            raise ValueError(f"Mask dimensions {image.size} != original image {image_size}")
        array = np.asarray(image, dtype=np.float32) / 255
        if array.max() == 0:
            raise ValueError("Initial target mask is empty")
        image = Image.fromarray(np.uint8(array*255)).filter(ImageFilter.GaussianBlur(3))
        return image
    box = np.asarray(mark["box"], dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box must be finite original-pixel [x1,y1,x2,y2]")
    if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
        raise ValueError(f"Box outside {image_size}: {box}")
    if perturb is not None:
        if box_config and box_config.get("mode", "legacy") == "structured":
            box = perturb_box(box, image_size, perturb, box_config)
        else:
            bw, bh = box[2]-box[0], box[3]-box[1]
            delta = perturb.uniform(-.06, .06, 4) * np.array([bw, bh, bw, bh])
            box = box + delta
    yy, xx = np.mgrid[:h, :w]
    dx = np.maximum(np.maximum(box[0]-xx, xx-box[2]), 0)
    dy = np.maximum(np.maximum(box[1]-yy, yy-box[3]), 0)
    sx, sy = max(2, (box[2]-box[0])*.15), max(2, (box[3]-box[1])*.15)
    p = np.exp(-.5*((dx/sx)**2+(dy/sy)**2))
    return Image.fromarray(np.uint8(p*255))


class LabDataset(BaseDataset):
    def __init__(self, config, split="train"):
        self.config, self.split, self.epoch = config, split, 0
        self.width, self.height = int(config.width), int(config.height)
        if self.width % 32 or self.height % 96:
            raise ValueError("Wan image width must divide by 32; three-view height by 96")
        self.num_frames, self.video_stride = 33, 4
        self.action_dim, self.state_dim = ACTION_DIM, STATE_DIM
        manifest_path = Path(config.manifest).resolve()
        manifest = json.loads(manifest_path.read_text())
        rows = manifest["segments"]
        explicit_split = {}
        if any("split" in row for row in rows):
            for row in rows:
                if row.get("split") not in ("train", "val"):
                    raise ValueError("Every manifest row must specify train/val when using explicit splits")
                key = (str(Path(row["dataset"]).expanduser().resolve()), int(row["episode_index"]))
                if key in explicit_split and explicit_split[key] != row["split"]:
                    raise ValueError("Episode crosses explicit train/val boundary")
                explicit_split[key] = row["split"]
            groups = {}
            for row in rows:
                group = row.get("source_group")
                if group is not None:
                    if group in groups and groups[group] != row["split"]:
                        raise ValueError("Source trajectory crosses train/val boundary")
                    groups[group] = row["split"]
        if split not in ("train", "val"):
            raise ValueError(split)
        self.segments, self.windows, self.excluded = [], [], []
        raw = []
        cache = {}
        for row in rows:
            root = Path(row["dataset"]).expanduser().resolve()
            ep = int(row["episode_index"])
            key = (str(root), ep)
            if key not in cache:
                info = json.loads((root/"meta/info.json").read_text())
                if ep in info.get("discarded_episode_indices", []):
                    self.excluded.append([str(root), ep])
                    continue
                fields = dict(episode_index=ep, episode_chunk=ep//int(info["chunks_size"]))
                data_path = root/info["data_path"].format(**fields)
                columns = ["observation.state", "timestamp", "observation.root_orientation"]
                jitter_key = config.get("jitter_observation_key")
                if jitter_key:
                    if not str(jitter_key).startswith("observation."):
                        raise ValueError("Jitter must come from observation.*")
                    columns.append(str(jitter_key))
                table = pq.read_table(data_path, columns=columns)
                state = finite_array(table["observation.state"].to_pylist(), 31, "state")
                names = info["features"]["observation.state"]["names"]
                if names[15] != "left_shoulder_pitch_joint" or names[30] != "right_hand_dex1_motor":
                    raise ValueError("Unsupported observation joint ordering")
                ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
                if not np.isfinite(ts).all() or not np.allclose(np.diff(ts), 1/30, atol=1e-4):
                    raise ValueError("This adapter requires synchronized 30 Hz observations")
                if jitter_key:
                    jitter = np.asarray(table[str(jitter_key)].to_pylist(), dtype=np.float32).reshape(-1)
                else:
                    jitter = orientation_speed(table["observation.root_orientation"].to_pylist(), ts)
                if len(jitter) != len(state) or not np.isfinite(jitter).all() or (jitter < 0).any():
                    raise ValueError("Invalid jitter observation")
                videos = [root/info["video_path"].format(video_key=c, **fields) for c in CAMERAS]
                for video in videos:
                    cap = cv2.VideoCapture(str(video))
                    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if count != len(state) or abs(fps-30) > .1:
                        raise ValueError(f"Video alignment mismatch: {video}, frames={count}, fps={fps}")
                cache[key] = dict(state=state, timestamp=ts, jitter=jitter,
                                  videos=videos, key=key)
            data = cache[key]
            start, end = int(row["start_frame"]), int(row.get("end_frame", len(data["state"])))
            if not 0 <= start < end <= len(data["state"]):
                raise ValueError(f"Invalid segment range: {row}")
            skill, hand = row["skill"], row["hand"]
            sid, hid, prompt = condition(skill, hand)
            marks = validate_targets(skill, row.get("targets"))
            if len({m.get("role", 0) for m in marks}) != len(marks):
                raise ValueError("Duplicate target role")
            for mark in marks:
                if mark.get("camera", CAMERAS[0]) not in CAMERAS or mark.get("role", 0) not in (0, 1):
                    raise ValueError("Invalid target camera/role")
                confidence = float(mark.get("confidence", .5))
                if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("Initial mark confidence must be in [0,1]")
            raw.append(dict(**data, start=start, end=end, skill=skill, hand=hand, sid=sid, hid=hid,
                            prompt=prompt, marks=marks, annotation_dir=manifest_path.parent,
                            binding=str(row.get("binding_id", f"{key}:{start}"))))
        keys = sorted(set(s["key"] for s in raw))
        if explicit_split:
            self.val_ids = [key for key in keys if explicit_split[key] == "val"]
        else:
            nval = int(config.val_episodes)
            if not 0 < nval < len(keys):
                raise ValueError("Require nonempty train/val episode sets")
            perm = np.random.default_rng(int(config.seed)).permutation(len(keys))
            self.val_ids = [keys[i] for i in perm[:nval]]
        self.train_ids = [key for key in keys if key not in self.val_ids]
        if not self.val_ids or not self.train_ids:
            raise ValueError("Require nonempty train/val episode sets")
        training = [s for s in raw if s["key"] in self.train_ids]
        state_values = np.concatenate([s["state"][s["start"]:s["end"]] for s in training])
        # Fit each action dimension ONLY from training segments where it is active.
        alo, ahi = np.zeros(ACTION_DIM), np.zeros(ACTION_DIM)
        actions = [self.raw_targets(s) for s in training]
        for d in range(ACTION_DIM):
            parts = [a[:, d] for s, a in zip(training, actions) if action_mask(s["skill"], s["hand"])[d]]
            if parts:
                vals = np.concatenate(parts)
                alo[d], ahi[d] = vals.min(), vals.max()
        self.stats = dict(action=dict(min=alo.tolist(), max=ahi.tolist()),
                          state=dict(min=state_values.min(0).tolist(), max=state_values.max(0).tolist()))
        self.normalization_stats_path = str(Path(config.artifact_dir)/"normalization_stats.npy")
        if split == "train":
            Path(config.artifact_dir).mkdir(parents=True, exist_ok=True)
            np.save(self.normalization_stats_path, dict(wbc=self.stats["action"], wbc_state=self.stats["state"],
                train_episodes=self.train_ids, val_episodes=self.val_ids, source="observation.state",
                alignment="state[t] -> state[t+1:t+33]", action_indices=ARM_INDICES,
                action_dim=ACTION_DIM, contract_version=CONTRACT_VERSION))
        chosen = self.train_ids if split == "train" else self.val_ids
        for s in raw:
            if s["key"] in chosen:
                idx = len(self.segments)
                self.segments.append(s)
                self.windows.extend((idx, t) for t in range(s["start"], s["end"]-32, int(config.window_stride)))
        if not self.windows:
            raise ValueError("No complete action chunks after initial annotation")

    @staticmethod
    def raw_targets(segment):
        start, end = segment["start"]+1, segment["end"]
        return segment["state"][start:end, ARM_INDICES]

    def __len__(self):
        return len(self.windows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @classmethod
    def from_config(cls, config, split="train"):
        return cls(config, split)

    def __getitem__(self, index):
        si, t = self.windows[index]
        s = self.segments[si]
        start = s["start"]
        offsets = list(self.config.history_offsets)  # strictly negative; current appended below
        if any(x >= 0 for x in offsets) or offsets != sorted(set(offsets)):
            raise ValueError("history_offsets must be unique sorted negative frame offsets")
        raw = [start] + [t+int(o) for o in offsets] + [t]
        frame_valid = np.array([True] + [j >= start for j in raw[1:-1]] + [True])
        history = [max(start, j) for j in raw]
        future = list(range(t, t+33, 4))
        needed = sorted(set(history+future))
        frames = {j: [] for j in needed}
        for vp in s["videos"]:
            cap = cv2.VideoCapture(str(vp))
            try:
                for j in needed:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                    ok, arr = cap.read()
                    if not ok:
                        raise RuntimeError(f"Decode failed {vp}:{j}")
                    frames[j].append(Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)))
            finally:
                cap.release()
        # Separate RNG keeps the existing initial-box perturbations unchanged.
        lighting = sample_lighting(self.config.get("photometric"),
            np.random.default_rng(int(self.config.seed) + self.epoch*1000003 + index + 987654321)) if self.split == "train" else None
        camera_motion = sample_camera(self.config.get("camera_augmentation"),
            np.random.default_rng(int(self.config.seed) + self.epoch*1000003 + index + 123456789)) if self.split == "train" else None
        packed = {j: compose([apply_camera(apply_lighting(im, lighting, ci), camera_motion, ci, (j-start)/30)
                             for ci, im in enumerate(images)],
                            self.width, self.height) for j, images in frames.items()}
        k = len(history)
        prior = np.zeros((k, 2, self.height, self.width), dtype=np.float32)
        mark_valid, confidence = np.zeros((k, 2), bool), np.zeros((k, 2), np.float32)
        rng = np.random.default_rng(int(self.config.seed) + self.epoch*1000003 + index)
        for mark in s["marks"]:
            role, camera = int(mark.get("role", 0)), mark.get("camera", CAMERAS[0])
            tiles = [Image.new("L", im.size, 0) for im in frames[start]]
            ci = CAMERAS.index(camera)
            tiles[ci] = reference_prior(mark, frames[start][ci].size, s["annotation_dir"],
                rng if self.split == "train" and self.config.get("perturb_initial_box", True) else None,
                self.config.get("box_augmentation"))
            tiles[ci] = apply_camera(tiles[ci], camera_motion, ci, 0., is_mask=True)
            prior[0, role] = np.asarray(compose(tiles, self.width, self.height), dtype=np.float32)/255
            mark_valid[0, role], confidence[0, role] = True, float(mark.get("confidence", .5))
        # Later rows intentionally have NO box: the initial reference binds identity.
        actions = s["state"][t+1:t+33, ARM_INDICES].copy()
        actions = normalize(actions, self.stats["action"])
        active = action_mask(s["skill"], s["hand"])
        actions[:, ~active] = 0
        state = normalize(s["state"][t:t+1], self.stats["state"])
        # No true executed command exists here. Memory uses PAST achieved motion
        # delta (observation[j]-observation[j-1]), never future action labels.
        previous = np.zeros((k, ACTION_DIM), np.float32)
        previous_valid = np.array([j > start for j in history]) & frame_valid
        for i, j in enumerate(history):
            if previous_valid[i]:
                previous[i] = s["state"][j, ARM_INDICES]-s["state"][j-1, ARM_INDICES]
        previous[:, ~active] = 0
        # Loss-only temporal max: covers all 30 Hz readings between sampled frames.
        jitter = [s["jitter"][t]] + [s["jitter"][j-3:j+1].max() for j in future[1:]]
        video = [packed[j] for j in future]
        return dict(video=video, first_frame_image=[video[0]], vace_video=None,
            action=torch.from_numpy(actions), action_mask=torch.from_numpy(np.broadcast_to(active, (32,ACTION_DIM)).copy()),
            proprio=torch.from_numpy(state), proprio_mask=torch.ones(1,31,dtype=torch.bool),
            video_mask=torch.ones(9,dtype=torch.bool), prompt=s["prompt"],
            episode_index=s["key"][1], start_frame=t, skill_id=s["sid"], hand_id=s["hid"],
            binding_id=s["binding"], reference_frame=start, history_indices=history,
            condition_images=[packed[j] for j in history],
            target_prior=torch.from_numpy(prior), mark_valid=torch.from_numpy(mark_valid),
            mark_confidence=torch.from_numpy(confidence),
            history_state=torch.from_numpy(normalize(s["state"][history], self.stats["state"])),
            previous_motion=torch.from_numpy(previous), previous_motion_valid=torch.from_numpy(previous_valid),
            history_valid=torch.from_numpy(frame_valid),
            relative_time=torch.tensor([(j-t)/30 for j in history],dtype=torch.float32),
            visual_jitter_score=torch.tensor(jitter,dtype=torch.float32))
