# 验收状态（2026-11-01）

> 本文件以下是旧版池化历史模块的验收记录。当前 `spacetime_v2` 改动的独立状态见
> [SPACETIME_VALIDATION.md](SPACETIME_VALIDATION.md)，旧版的 GPU 通过不代表新版通过。

**训练管线验收通过：真实 Foundation 初始化训练 3 步，再从保存的 delta 热启动训练 1 步。**
两次均完成验证、保存和全部 858 个 delta 张量的精确读回核验；视频/动作条件路径及
技能、手别、目标、历史相关的 8 项梯度均有限且非零。测试源代码为
`6d59244d4490e53a41939a7d7b0aab28fdf59c51`，后续提交仅整理验收文档和记录。
这确认了训练、验证与权重重建链路，未验证机器人闭环操作成功率。

## 最终 GPU 验收

- 服务器：ai2-root-4080，实际 NVIDIA A800-SXM4-80GB，GPU 2；单卡 BF16、batch size 1。
- 沿用环境 `/root/ykj/OpenWAM/.conda-envs/openwam`，PyTorch 2.7.1+cu128，未修改驱动或安装依赖。
- 使用完整 manifest：23 个训练 episode / 567 个窗口，7 个验证 episode / 182 个窗口。
  短验收只执行 3 次参数更新、抽取 2 个验证窗口，没有遍历全部窗口。
- 3 步 run：`/root/ykj/lab-low/outputs/acceptance/2026-11-01_20-05-34`。
  最后训练 total/video/action loss 为 1.776114 / 0.166714 / 1.609400；
  验证为 1.828111 / 0.096523 / 1.731589，全部有限。不同扩散时间步不能据此判断收敛。
- 修复后峰值 PyTorch 分配显存 45,197 MiB（约 44.1 GiB），不含全部进程显存开销。
  仅在 80 GB 卡验证，启动脚本默认要求至少 76,000 MiB 空闲显存。
- 保存 `checkpoint_step_3.safetensors`，大小 2,045,582,662 bytes。
  858 个保存张量逐个与内存模型完全相等；冻结视频权重由原 Foundation 提供。
- 热启动 run：`/root/ykj/lab-low/outputs/reload_check/2026-11-01_20-11-50`。
  transfer_report 确认原 Foundation + 上述 delta 加载，完成 1 步更新及 1 个窗口验证，
  total loss 为 0.183286，validation total loss 为 1.415564；再次保存并精确读回 858 个张量。
  这是模型权重热启动，优化器重新初始化，不是完整训练状态续训。
- 两次日志均有 COMPLETED，训练进程正常结束。未额外启动长期训练。

原始精简记录见 [acceptance_report.json](validation/acceptance_report.json) 和
[reload_report.json](validation/reload_report.json)。服务器日志为项目内
`.cache/acceptance.log`、`.cache/reload_check.log`。

## 已通过

- 本地和 ai2-root-4080 的全量真实数据检查：23 个训练 episode、7 个验证 episode，
  567/182 个完整窗口；两者按 episode 隔离。
- 监督来自未来 observation.state；不读取 action.wbc。验证了右手标签的精确时间对齐。
- 只在初始 reference 帧输入框，后续 mark_valid 均为 false，历史没有未来帧。
- 目标框、技能和手别改变会影响条件模块；条件模块反向梯度有限且非零。
- 闭合技能词表、非法手别拒绝、未选中手臂保持当前角度、移动/操作动作通道分离。
- 在线 buffer 与离线数据条件字段一致（本地：真实状态序列与目标几何；图像组装共用函数）。
- IMU 权重公式、四元数符号不变性和视频 latent 时间对齐。
- 修复后的实际 freeze_modules 方法在服务器 CPU 测试中保留输入梯度、冻结参数梯度，
  同时验证上游原方法会截断该梯度，作为回归对照。
- Python 编译检查、Bash 语法检查、Hydra 配置解析。
- 三路原始视频在服务器独立数据副本中补齐，未修改原 OpenWAM 数据目录。

数据/条件检查见 validation/local_contract.json 和 validation/contract_results.json；
冻结回归见 validation/freeze_regression.json；角速度分布见 validation/jitter_statistics.json。

## 未通过的首轮 GPU 检查，不作为最终验收

- 服务器：ai2-root-4080，实际硬件是 NVIDIA A800-SXM4-80GB；使用 GPU 2。
- 环境：/root/ykj/OpenWAM/.conda-envs/openwam，PyTorch 2.7.1+cu128；没有安装或更改驱动。
- 测试源代码：4e720df；修复视频梯度的提交：5b1f009。
- 真实 Foundation 初始化，两个完整 episode 的训练/验证子集，3 次优化器更新。
- 三步 total loss：0.254950、0.128630、1.520509；均有限，但不同随机扩散时间步下不可解读为收敛趋势。
- 当时峰值分配显存约 45,197 MiB。**这是视频梯度被截断时的数值，不能代表修复后的显存需求。**
- 缺失 video_context 梯度触发断言，因此未进入 validation/checkpoint readback。
- 详细记录见 validation/first_gpu_run.json。失败日志保留于服务器 .cache/smoke.log。

## 复验入口与边界

服务器上依次执行以下两行；第二行是一条完整命令，把 2 替换为实际空闲卡号。
此入口及这些参数已经实测通过：

```bash
cd /root/ykj/lab-low
CUDA_VISIBLE_DEVICES=2 bash scripts/train_lab.sh training.max_steps=3 training.save_steps=3 validation_every=3 validation_windows=2 training.output_path=outputs/acceptance
```

验收同时检查 COMPLETED、八项非零有限梯度（包括 video_context/action_context）、
验证集有限损失、checkpoint_readback.json 的 exact_match=true。
本次已完成这些检查及 Foundation + delta 热启动检查。临时测试脚本已在本机和服务器删除。

本批数据只能验证右手取试管的数据链路。其余技能、左手操作和移动需要对应轨迹；
移动必须有 observation.* 的机体坐标实测速度。当前抖动量来自 root_orientation 角速度代理，
不是相机独立陀螺仪。短训练、梯度检查和输入敏感性均不代表闭环抓取或目标绑定成功率。
