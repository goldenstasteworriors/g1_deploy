# LabWAM 推理优化与同显卡 Alpha 对照

日期：2026-11-14。服务器：pku4090。仅模型离线测试和 HTTP 推理验证，未发送机器人指令。

## 已接入的实现

1. `LabArchitecture.prepare_inference_tokens` 在每次生成前计算技能、手别、目标和历史条件，去噪期间复用同一份 token/mask。每次新请求重新计算，状态变化不会沿用旧条件。训练路径继续每次前向计算条件，显式拒绝训练使用预计算 token。
2. 每个 `InferenceSession` 独立持有最多 64 项图像 latent 缓存，按组合后图像内容、模式、尺寸、dtype、device 建键，复用初始目标参考和重复历史图像的冻结 VAE 编码。reset/stop 清空；operator/simulation 不共用。保留原 prepare_condition 请求内部的帧去重语义，没有顺带改变目标绑定逻辑。
3. 自定义 worker 接入上游 `apply_compile_optimizations`，使用已经测通的 `default` 固定形状 MoT 编译。
4. 显式可选的上游 `DiTVelocityCache`，每次 generate 前 reset。默认关闭。它缓存的是联合去噪速度场，不是机器人执行速度，不修改控制器阈值。
5. 模型服务可用一份真实记录的 reset observation 完成启动预热，预热完成才开放 HTTP 服务。该过程仅跑模型；预热后清空会话状态，真实部署仍需正常 reset。

未删除视频专家、未来视频 token、目标框或历史。未修改权重、动作维度、夹爪校正、重力前馈、急停逻辑、显卡驱动或 Python 环境。

## 测量方法

RTX 4090（系统报告 49140 MiB），PyTorch 2.5.1+cu124，BF16。使用真实 take_tube 30000-step 权重、episode 0 的三路 RGB、初始框、英文技能 prompt、右手条件、`observation.state` 与因果历史。

取当前帧 60、76、92、108、124、140、160、200、240，每组预热 3 次后，按时间顺序重复 3 轮，共 27 次测量。每轮重新清空 latent cache，避免重复同一帧的全缓存命中夸大收益。每帧种子一致；图像序列读取在计时之外。所有 GPU 计时同步 CUDA。

LabWAM 总耗时包含 buffer.sample、图像条件编码、条件 token 计算、generate 和动作恢复，不包含相机、HTTP、机械执行。generate 耗时还包含当前图像/文本等 pipeline 准备，不是纯去噪循环。动作差异在机器人侧夹爪校正之前计算。

## LabWAM 结果

| 配置 | 中位耗时 | P90 | generate 中位耗时 | 右夹爪最大差* |
|---|---:|---:|---:|---:|
| 原路径，20 步 | 1192 ms | 1199 ms | 1130 ms | 0 |
| 原路径，10 步 | 641 ms | 652 ms | 579 ms | 0 |
| 原路径，6 步 | 419 ms | 420 ms | 357 ms | 0 |
| 条件复用，20 步 | 1162 ms | 1162 ms | 1105 ms | **0** |
| 条件复用，10 步 | 617 ms | 617 ms | 560 ms | **0** |
| 编译＋条件复用，20 步 | **966 ms** | 973 ms | 910 ms | 0.01728 rad |
| 编译＋条件复用，10 步 | **524 ms** | 526 ms | 467 ms | 0.03456 rad |
| 编译＋条件复用，6 步 | **343 ms** | 344 ms | 286 ms | 0.15337 rad |
| 上述优化＋保守缓存，10 步 | **390 ms** | 423 ms | 332 ms | 0.06048 rad |
| 上述优化＋官方默认缓存，10 步 | **254 ms** | 255 ms | 197 ms | **1.14488 rad** |

*相对同一去噪步数、同一输入的未优化输出，缓存行包括编译造成的数值差异；不是相对真实动作标签的误差。保守缓存为 threshold=0.999、max_skips=1；官方默认缓存为 0.99、3。

- 仅条件复用在 20/10 步、全部测量帧上动作逐项一致。conditioner 从每次 20/10 遍降为 1 遍。
- 20 步编译＋复用的右臂单项最大差为 0.00924 rad，右夹爪最大差为 0.01728 rad。
- 保守缓存跳过 2–3/10 次网络计算；官方默认缓存跳过 6/10 次，所以虽然标称 10 步，后者只实际运行 4 次联合网络前向。
- 减少步数本身也会改变输出。10 步原实现相对 20 步，最差测量帧的右夹爪 MAE 为 0.11334 rad；6 步为 0.52601 rad。不能把减步当作无损工程优化。
- 编译预热最高约 14.82 秒（机器已有上一轮编译磁盘缓存），不计入稳定耗时；冷机器可能更久。

正式服务默认开启条件复用和预热后的编译，保留原默认 20 步，**不默认启用 DiT 缓存、不强制改客户端指定的步数**。客户端若原来请求 6 步，仍是 6 步，使用新的共享推理路径。

## 为什么曾看起来比 Alpha 慢很多

论文在第 18–19 页报告：RTX 5090，10 步同步联合去噪，编译/CUDA Graphs、速度场缓存、文本缓存、跳过视频解码，约 170 ms/chunk。

为避免只把差距归因于 GPU，这次额外加载了实际下载的原始 `OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors`，在同一张 4090、相同 320×384 图像、9 视频帧和 32 动作块下测速：

| 原始 Alpha，4090 | 中位耗时 |
|---|---:|
| 20 步，不编译、不用速度场缓存 | 1086 ms |
| 10 步，不编译、不用速度场缓存 | 554 ms |
| 10 步，编译 | 454 ms |
| 10 步，编译＋官方缓存 | **192 ms** |

Alpha 对照使用其原生 80D 动作头、零值 80D 归一化 proprio、全部 80 维激活；LabWAM 是实际 31D 状态和已训练的 16D 动作头。它是同 GPU 的运行开销对照，不是 RoboTwin 成功率复现，也不是两个模型的动作等价实验。

实际差距可分解为：

1. **比较条件不一致是主要原因。** 先前 1192 ms 是我们的 20 步未加速路径，而 170 ms 是官方 10 步、缓存跳步、编译后的去噪循环。原始 Alpha 在我们的机器上不加速跑 20 步也要 1086 ms。
2. **同配置后差距约 62 ms，而不是数倍。** 在 10 步编译＋官方缓存下，Alpha 是 192 ms，我们是 254 ms。我们的 generate 为 197 ms，与 Alpha 整个 generate 相差约 5 ms；另有约 57 ms 位于 sample/历史目标条件准备/动作恢复等外围路径。
3. **新增历史和目标条件有可测的开销。** 优化后条件 VAE 编码约 39 ms、条件 token 约 2.8 ms，其余包含三路图像组合、CPU 张量准备和动作恢复；这些部分原始 Alpha 没有同等输入。上述各项是中位数，不能当作每次请求完全精确的可加分解。
4. **还有硬件、软件和测量口径差异。** 官方是 5090，我们是 4090；我们的 PyTorch 为 2.5.1，而当前官方仓库要求至少 2.7.1。当前稳定路径是 `torch.compile(mode=default)`；此前 reduce-overhead 首次预热崩溃，未确认本环境实现论文描述的 CUDA Graph 重放。因此不能把剩余 192 vs 170 ms 全部量化归因于显卡。

## 运行与回退

取试管启动预热输入保存在 pku4090 的以下位置（数据集第 0 帧，仅用于加载阶段推理，不作为机器人实际初始姿态）：

```text
/home/yanjun/ykj/labwam_g1/runtime/warmup/take_tube.json
```

原模型启动命令保持可用。下面是一条命令，不需换行；已有 8849 服务运行时不要重复启动：

```bash
bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_pku_policy.sh take_tube 20
```

脚本发现该模型的预热文件后自动启用 default 编译并预热；其他没有预热文件的模型默认只启用条件复用。显式 `LABWAM_COMPILE_MODE` 可以覆盖此选择。共享服务优化同时适用于实时相机和数据集图像客户端。

恢复原计算路径的单行启动命令（停止现有模型服务后使用，不操作机器人控制器）：

```bash
LABWAM_COMPILE_MODE=none LABWAM_CONDITION_CACHE=0 bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_pku_policy.sh take_tube 20
```

实验缓存通过 `LABWAM_DIT_CACHE_THRESHOLD`、`LABWAM_DIT_CACHE_MAX_SKIPS` 显式开启。当前未设这两个变量。模型的 ready 信息报告编译预热状态；每次 predict 返回实际编译路径状态与缓存统计。

## 验证与范围

已检查配置透传、缓存内容建键/容量/清理、禁止训练使用预计算 token、GPU 输出有限性和同输入数值差异。LabWAM 每配置 27 次，共 270 次测量；Alpha 共 108 次测量。临时测试脚本按约定删除，保留 JSON 结果。

正式 8849 服务用原启动脚本 `take_tube 20` 启动并通过 HTTP 验证：health 为 ready/alive，condition_cache=true、compile_mode=default、compiled_active=true、dit_cache_threshold=null，启动推理预热耗时 14.70 秒。实际发送 simulation scope 的 reset、20/10/6 步 predict、observe、predict 和 stop，全部通过且每次输出 32×16 有限动作；测试后服务仍健康。该 HTTP 检查使用第 0 帧短历史，仅验证接口，不混入上面的完整历史性能统计。

本次仅验证取试管权重的计算速度和数值变化，没有操作机器人，没有新增成功判断，也没有证明其他技能的操作成功率。缓存和减少步数应继续检查夹爪时序，不能用右臂平均误差替代。

来源与实现：

- [官方论文，第 18–19 页](https://arxiv.org/pdf/2609.07398#page=19)
- [官方部署配置](https://github.com/OpenWAM-Official/OpenWAM/blob/main/configs/deploy.yaml)
- [官方环境要求](https://github.com/OpenWAM-Official/OpenWAM#native-python-environment)
- `lab_low/inference_optimizations.py`：选项、图像 latent 缓存、CLI 透传。
- `lab_low/architecture.py`：推理条件预计算，训练路径保留。
- `lab_low/deployment.py`：会话缓存生命周期及实际 generate 参数。
- `scripts/deploy_lab_worker.py`、`scripts/serve_lab_policy.py`、`scripts/start_pku_policy.sh`：服务接入、启动预热与环境选项。
- `validation/inference_optimization_20261114.json`：逐请求计时、数值差异和服务验证记录。
