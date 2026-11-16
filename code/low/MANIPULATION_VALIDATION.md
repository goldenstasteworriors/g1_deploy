# 纯操作版本验收（2026-11-02）

当前契约为 `spacetime_manip_v3`，仅支持七个实验操作技能。
输出为 16D：左臂七关节、左夹爪、右臂七关节、右夹爪；无底座速度通道。
输入保留完整 31D observation.state。手别编号为 left=0、right=1、both=2。
初始框绑定、四帧因果时空历史和抖动降权保持不变。

## 本地验证

使用现有 gigaworld-policy 环境，记录见
[manipulation_contract.json](validation/manipulation_contract.json)。

- 真实数据 567/182 个训练/验证窗口，动作标签严格来自未来 observation.state。
- 七技能与手别约束、16D 门控、未操作手保持当前姿态通过。
- 旧移动技能、旧 19D 命令与归一化文件被拒绝。
- 在线与离线状态历史、过去实测关节变化及有效掩码一致。
- 全尺寸 BF16 条件模块完成前向、反向和优化器更新；输出 10×4096 条件 token。

## 服务器验证

ai2-root-4080 上使用现有 OpenWAM 环境及 A800 80GB GPU 2，完整 Foundation
三步训练和新版 delta 热启动一步训练均完成。代码提交为 `b058fc4`。
详细结果见 [manipulation_gpu.json](validation/manipulation_gpu.json)。

- 三步训练目录：`/root/ykj/lab-low/outputs/manipulation_acceptance/2026-11-02_18-40-52`。
- 热启动目录：`/root/ykj/lab-low/outputs/manipulation_reload/2026-11-02_18-44-32`。
- 两轮各 13 项梯度检查全部有限且非零，包含四层空间注意力、一层时间注意力、
  技能、手别、目标读取、状态历史、动作头及视频/动作条件 context。
- 两轮训练和留出集损失均有限；三步训练的留出 action loss 为 1.56417，
  热启动一步后为 1.54595。这些数值只用于连通性检查，不用于效果结论。
- 峰值训练 allocated 显存 45,244.70 MiB（约 44.2 GiB）。
- 两轮保存权重均逐一读回 902 个 delta 张量，全部与内存模型精确一致。
- 权重形状确认动作输入/输出 16D、技能 embedding 7 类、手别 embedding 3 类、
  历史状态编码输入 48D（31 状态 + 16 过去实测关节变化 + 1 有效标记）。
- 热启动确实读取首轮保存的新版 delta；优化器重新初始化，不宣称精确恢复训练状态。

使用项目原启动脚本；未安装依赖，未改动显卡或 CUDA 驱动。临时本地测试脚本已删除。
短训练只验证管线，不代表已学会全部操作或达到闭环成功率要求。

## 本次文件变更

- `lab_low/skills.py`：移除移动技能，定义 16D 动作及左右手门控。
- `lab_low/conditioning.py`：七技能/三手别 embedding，48D 状态历史编码输入。
- `lab_low/architecture.py`：16D 架构检查与新版权重兼容性检查。
- `lab_low/dataset.py`：只从 observation 构建双臂标签及历史，更新归一化。
- `lab_low/runtime.py`：删除速度接口，更新在线历史、归一化和执行门控。
- `configs/lab.yaml`、`configs/dataloader/lab.yaml`：动作维度和独立数据缓存。
- `scripts/train_lab.py`、`scripts/prepare_lab_manifest.py`：训练记录与技能输入约束。
- `README.md`、`SPACETIME_VALIDATION.md`：更新当前说明与历史记录入口。
- 新增 `MANIPULATION_VALIDATION.md`、`validation/manipulation_contract.json`、
  `validation/manipulation_gpu.json`：本地和服务器验收记录。
