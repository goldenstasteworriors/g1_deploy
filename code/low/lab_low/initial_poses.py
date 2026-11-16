"""Recorded achieved arm/Dex1 poses; never use action labels or recorded legs."""
import hashlib
import json
from pathlib import Path
import numpy as np

SKILL_NAMES_ZH = {
    "take_dropper_and_aspirate": "取滴管并吸液",
    "pick_up_test_tube": "取试管",
    "dispense_into_held_test_tube": "向手持试管加液",
    "return_dropper": "放回滴管",
    "mix_test_tube": "混匀试管",
    "return_test_tube_to_rack": "试管归架",
    "place_test_tube_in_water_bath": "放入水浴",
}


class PoseCatalog:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads(self.path.read_text())
        data = self.path.with_name(self.meta["array_file"])
        if hashlib.sha256(data.read_bytes()).hexdigest() != self.meta["sha256"]:
            raise ValueError("Initial-pose catalog checksum mismatch")
        self.arrays = np.load(data, allow_pickle=False)
        self.skills = sorted({e["skill"] for e in self.meta["episodes"]})
        self.entries = {(e["skill"], e["episode"]): e for e in self.meta["episodes"]}

    def pose(self, selection):
        skill, episode, frame = selection["skill"], selection["episode"], selection["frame"]
        entry = self.entries[(skill, episode)]
        if type(frame) is not int or not 0 <= frame < entry["frames"]:
            raise ValueError(f"Frame must be 0..{entry['frames']-1}")
        pose = self.arrays[entry["key"]][frame].astype(float)
        if pose.shape != (16,) or not np.isfinite(pose).all():
            raise ValueError("Invalid achieved pose")
        return pose

    def menu(self):
        lines = ["选择初始姿态：输入三个数字“动作编号 轨迹编号 帧编号”，然后按 Enter；Esc 取消。",
                 "轨迹编号是数据集全局编号，帧从0开始。括号内显示合法帧范围。这不会切换推理任务。"]
        for i, skill in enumerate(self.skills):
            episodes = [f"{e['episode']}(0..{e['frames']-1})" for e in self.meta["episodes"] if e["skill"] == skill]
            lines.append(f"动作 {i}：{SKILL_NAMES_ZH.get(skill, skill)} ({skill})；轨迹：" + ", ".join(episodes))
        return "\n".join(lines)

    def parse(self, value):
        parts = value.split()
        if len(parts) != 3 or any(not p.isdecimal() for p in parts):
            raise ValueError("Enter three nonnegative integers: skill_number episode frame")
        sid, episode, frame = map(int, parts)
        selection = dict(skill=self.skills[sid], episode=episode, frame=frame)
        self.pose(selection)
        return selection

    def load_selection(self, path):
        path = Path(path)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["catalog_sha256"] != self.meta["sha256"]:
                raise ValueError("Saved initial pose belongs to a different catalog")
            selection = saved["selection"]
        else:
            selection = dict(skill="pick_up_test_tube", episode=0, frame=0)
        self.pose(selection)
        return selection

    def save_selection(self, path, selection):
        self.pose(selection)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(dict(catalog_sha256=self.meta["sha256"], selection=selection), indent=2) + "\n")
        temporary.replace(path)
