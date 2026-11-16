"""LeRobot v2.1 G1 reader: explicit state/command alignment and three views."""
from pathlib import Path
import json

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout


class SonicLeRobotDataset(BaseDataset):
    def __init__(self, config, split="train"):
        if split != "train":
            raise ValueError("This verification dataset has no held-out split configured")
        self.config = config
        self.root = Path(config.dataset_dir).expanduser()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.num_frames = int(config.num_frames)
        self.video_stride = int(config.video_stride)
        self.height, self.width = int(config.height), int(config.width)
        if (self.num_frames - 1) % self.video_stride:
            raise ValueError("num_frames - 1 must be divisible by video_stride")
        self.tasks = {x["task_index"]: x["task"] for x in
                      map(json.loads, (self.root / "meta/tasks.jsonl").read_text().splitlines())}
        self.episodes, self.windows = [], []
        for entry in map(json.loads, (self.root / "meta/episodes.jsonl").read_text().splitlines()):
            ep = int(entry["episode_index"])
            fields = dict(episode_index=ep, episode_chunk=ep // int(self.info["chunks_size"]))
            path = self.root / self.info["data_path"].format(**fields)
            columns = list(dict.fromkeys([config.action_key, config.state_key, "task_index", "timestamp"]))
            table = pq.read_table(path, columns=columns)
            action = np.asarray(table[config.action_key].to_pylist(), dtype=np.float32)
            state = np.asarray(table[config.state_key].to_pylist(), dtype=np.float32)
            timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
            if not np.isfinite(action).all() or not np.isfinite(state).all():
                raise ValueError(f"Nonfinite state/action in {path}")
            if action.ndim != 2 or state.ndim != 2 or len(action) != entry["length"]:
                raise ValueError(f"Invalid dimensions or length in {path}")
            if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 1 / self.info["fps"], atol=1e-4):
                raise ValueError(f"Nonuniform timestamps in {path}; resampling is required")
            videos = {}
            for key in config.camera_layout:
                video = self.root / self.info["video_path"].format(video_key=key, **fields)
                cap = cv2.VideoCapture(str(video))
                count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if count < len(action) or abs(fps - self.info["fps"]) > 0.1:
                    raise ValueError(f"Video/table alignment mismatch: {video}: frames={count}, fps={fps}")
                videos[key] = video
            self.episodes.append(dict(action=action, state=state, videos=videos, fields=fields,
                                      task=int(table["task_index"][0].as_py()), path=str(path)))
            self.windows.extend((len(self.episodes) - 1, start) for start in
                                range(0, len(action) - self.num_frames + 1, int(config.window_stride)))
        if not self.windows:
            raise ValueError("No full training windows found")
        self.action_dim = self.episodes[0]["action"].shape[1]
        self.state_dim = self.episodes[0]["state"].shape[1]
        self.stats = {}
        for field in ("action", "state"):
            values = np.concatenate([ep[field] for ep in self.episodes])
            lo, hi = values.min(0), values.max(0)
            self.stats[field] = dict(min=lo.tolist(), max=hi.tolist(),
                                     constant=(np.abs(hi - lo) < 1e-6).tolist())
        self.normalization_stats_path = str(self.root / "meta/sonic_normalization.npy")
        payload = {"source_action_key": str(config.action_key), "source_state_key": str(config.state_key),
                   "alignment": "state[t] -> command[t:t+horizon]",
                   "wbc": self.stats["action"], "wbc_state": self.stats["state"]}
        # Only rank zero writes; all ranks derive identical statistics from source tables.
        import os
        if int(os.environ.get("RANK", "0")) == 0:
            np.save(self.normalization_stats_path, payload, allow_pickle=True)

    @classmethod
    def from_config(cls, config, split="train"):
        return cls(config, split)

    def __len__(self):
        return len(self.windows)

    def _normalize(self, values, field):
        lo = np.asarray(self.stats[field]["min"], dtype=np.float32)
        hi = np.asarray(self.stats[field]["max"], dtype=np.float32)
        out = 2 * (values - lo) / np.maximum(hi - lo, 1e-6) - 1
        return np.clip(out, -1, 1).astype(np.float32)

    def denormalize_action(self, values):
        lo = np.asarray(self.stats["action"]["min"], dtype=np.float32)
        hi = np.asarray(self.stats["action"]["max"], dtype=np.float32)
        return (np.asarray(values) + 1) * 0.5 * (hi - lo) + lo

    def __getitem__(self, index):
        ep_index, start = self.windows[index]
        ep = self.episodes[ep_index]
        positions = list(range(start, start + self.num_frames, self.video_stride))
        frames_by_index = [dict() for _ in positions]
        for camera, path in ep["videos"].items():
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            target = dict(zip(positions, range(len(positions))))
            try:
                for frame_index in range(start, positions[-1] + 1):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Video decode failed at {frame_index}: {path}")
                    if frame_index in target:
                        frames_by_index[target[frame_index]][camera] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
        video = [assemble_multiview_layout(frames, list(self.config.camera_layout), self.height, self.width)
                 for frames in frames_by_index]
        # Commands are already paired with observations by the exporter. Do not shift them
        # as if they were achieved states: action.wbc is a command, observation.state is measured.
        horizon = self.num_frames - 1
        action = self._normalize(ep["action"][start:start + horizon], "action")
        proprio = self._normalize(ep["state"][start:start + 1], "state")
        return dict(video=video, first_frame_image=[video[0]], vace_video=None,
                    action=torch.from_numpy(action), action_mask=torch.ones(action.shape, dtype=torch.bool),
                    proprio=torch.from_numpy(proprio), proprio_mask=torch.ones(proprio.shape, dtype=torch.bool),
                    video_mask=torch.ones(len(video), dtype=torch.bool), prompt=self.tasks[ep["task"]],
                    episode_index=ep_index, episode_path=ep["path"], start_frame=start,
                    end_frame=start + self.num_frames, episode_length=len(ep["action"]),
                    task_name="pick_up_pipette", active_arm="both")
