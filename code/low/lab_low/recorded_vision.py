"""Dataset RGB/target reference for hybrid deployment; never read action/state labels."""
import base64
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from lab_low.skills import condition, validate_targets

CAMERAS = tuple("observation.images." + n for n in ("ego_view", "left_wrist", "right_wrist"))
TASKS = {
    "TAKE_TUBE": ("pick_up_test_tube", "right"),
    "TAKE_DROPPER_AND_ASPIRATE": ("take_dropper_and_aspirate", "left"),
    "DISPENSE_TO_HELD_TUBE": ("dispense_into_held_test_tube", "both"),
    "RETURN_DROPPER": ("return_dropper", "left"),
    "RETURN_TUBE_TO_RACK": ("return_test_tube_to_rack", "right"),
    "PLACE_TUBE_IN_BATH": ("place_test_tube_in_water_bath", "right"),
}


class RecordedDataset:
    def __init__(self, root, allowed_pairs=None):
        self.root = Path(root).expanduser().resolve()
        meta = self.root / "meta"
        self.info = json.loads((meta / "info.json").read_text())
        if self.info["fps"] != 30:
            raise ValueError("Recorded deployment uses the trained 30 Hz dataset")
        annotations = json.loads((meta / "initial_ego_boxes.json").read_text())
        self.annotation_digest = hashlib.sha256((meta / "initial_ego_boxes.json").read_bytes()).hexdigest()
        marks = {r["episode_index"]: r for r in annotations["episodes"]}
        episodes = {r["episode_index"]: r for r in map(json.loads, (meta / "episodes.jsonl").read_text().splitlines())}
        prompts = json.loads((meta / "primitive_prompts.json").read_text())
        self.allowed_pairs = None if allowed_pairs is None else {tuple(p) for p in allowed_pairs}
        self.entries = {}
        for row in json.loads((meta / "segment_provenance.json").read_text()):
            if row["task"] not in TASKS:
                continue
            ep = row["output_episode"]
            skill, hand = TASKS[row["task"]]
            mark, episode = marks[ep], episodes[ep]
            if mark["review_id"] != row["review_id"] or mark["task"] != row["task"]:
                raise ValueError(f"Annotation does not belong to episode {ep}")
            prompt = prompts[row["task"]]
            if prompt != condition(skill, hand)[2] or episode["tasks"] != [prompt]:
                raise ValueError(f"Dataset prompt differs from the trained skill/hand prompt: {ep}")
            targets = [] if row["task"] == "DISPENSE_TO_HELD_TUBE" else [
                dict(t, confidence=t.get("confidence", mark.get("prompt", {}).get("confidence", .5)))
                for t in mark["targets"]]
            validate_targets(skill, targets)
            self.entries[ep] = dict(episode=ep, skill=skill, hand=hand, prompt=prompt,
                frames=episode["length"], reference_frame=mark["frame_index"], targets=targets,
                reference_image=mark.get("reference_image"),
                reference_image_sha256=mark.get("reference_image_sha256"),
                review_id=mark["review_id"])

    def asset_paths(self, episode):
        fields = dict(episode_index=episode, episode_chunk=episode // self.info["chunks_size"])
        paths = [self.root / self.info["video_path"].format(video_key=c, **fields) for c in CAMERAS]
        reference = self.entries[episode]["reference_image"]
        return paths + ([] if not reference else [self.root / reference])

    def selection(self, episode, frame=0):
        if type(episode) is not int or type(frame) is not int:
            raise ValueError("Episode and frame must be integers")
        row = self.entries[episode]
        if not 0 <= frame < row["frames"]:
            raise ValueError(f"Episode {episode}: frame must be 0..{row['frames']-1}")
        if self.allowed_pairs is not None and (row["skill"], row["hand"]) not in self.allowed_pairs:
            raise ValueError(f"Selected checkpoint does not support {row['skill']}/{row['hand']}")
        missing = [str(path) for path in self.asset_paths(episode) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Recorded episode is not fully copied: " + ", ".join(missing))
        return dict(skill=row["skill"], episode=episode, frame=frame)

    def task(self, selection):
        if self.selection(selection["episode"], selection["frame"]) != selection:
            raise ValueError("Selected pose and recorded skill do not match")
        return copy.deepcopy(self.entries[selection["episode"]])

    def menu(self, skills=None):
        from lab_low.initial_poses import SKILL_NAMES_ZH
        lines = ["数据集轨迹（全局 episode 编号；帧从0开始；仅列出本机数据已齐全、当前模型支持的轨迹）："]
        for skill in sorted({r["skill"] for r in self.entries.values()}):
            rows = [r for r in self.entries.values() if r["skill"] == skill and
                    (self.allowed_pairs is None or (r["skill"], r["hand"]) in self.allowed_pairs)
                    and all(p.is_file() for p in self.asset_paths(r["episode"]))]
            if rows:
                prefix = "" if skills is None else f"动作编号 {skills.index(skill)} / "
                lines.append(prefix + SKILL_NAMES_ZH.get(skill, skill) + ": " + ", ".join(
                    f"{r['episode']}(帧0..{r['frames']-1})" for r in rows))
        return "\n".join(lines)


class RecordedEpisode:
    def __init__(self, dataset, episode, start_frame=0):
        self.dataset = dataset
        self.selection = dataset.selection(episode, start_frame)
        self.task = dataset.task(self.selection)
        self.frame = start_frame
        self.captures = []
        fields = dict(episode_index=episode, episode_chunk=episode // dataset.info["chunks_size"])
        try:
            for camera in CAMERAS:
                path = dataset.root / dataset.info["video_path"].format(video_key=camera, **fields)
                capture = cv2.VideoCapture(str(path))
                self.captures.append(capture)
                if not capture.isOpened():
                    raise ValueError(f"Cannot open {path}")
            self.reference_images = self.images(self.task["reference_frame"])
            # Use the exact annotated ego image when supplied by the dataset.
            if self.task["reference_image"]:
                path = dataset.root / self.task["reference_image"]
                digest = self.task["reference_image_sha256"]
                if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("Annotated reference image checksum mismatch")
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot decode reference image {path}")
                self.reference_images[CAMERAS[0]] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for mark in self.task["targets"]:
                im = self.reference_images[mark["camera"]]
                x1, y1, x2, y2 = mark["box"]
                if not np.isfinite(mark["box"]).all() or not (0 <= x1 < x2 <= im.shape[1] and 0 <= y1 < y2 <= im.shape[0]):
                    raise ValueError("Invalid box in annotated reference image")
            self.current_images = self.images(self.frame)
        except Exception:
            self.close()
            raise

    def images(self, frame):
        result = {}
        for camera, capture in zip(CAMERAS, self.captures):
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"Failed to decode episode {self.selection['episode']} frame {frame}: {camera}")
            result[camera] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return result

    @property
    def remaining(self):
        return self.task["frames"] - 1 - self.frame

    def advance(self):
        if not self.remaining:
            raise StopIteration("Recorded episode ended")
        self.frame += 1
        self.current_images = self.images(self.frame)

    @staticmethod
    def encode(images, extension=".png"):
        encoded = {}
        for name, rgb in images.items():
            ok, data = cv2.imencode(extension, rgb[..., ::-1],
                                   [cv2.IMWRITE_JPEG_QUALITY,95] if extension==".jpg" else [])
            if not ok:
                raise ValueError("PNG encode failed")
            encoded[name] = base64.b64encode(data).decode()
        return encoded

    def packet(self, state, index):
        state = np.asarray(state, dtype=float)
        if state.shape != (31,) or not np.isfinite(state).all():
            raise ValueError("Expected live observation.state[31]")
        return dict(frame_index=index, state=state.tolist(), images=self.encode(self.current_images))

    def reset_packet(self, state, binding):
        return dict(self.packet(state, 0), op="reset", binding_id=binding,
                    skill=self.task["skill"], hand=self.task["hand"], prompt=self.task["prompt"],
                    targets=self.task["targets"], reference_images=self.encode(self.reference_images))

    def close(self):
        for capture in self.captures:
            capture.release()
