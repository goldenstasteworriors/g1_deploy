# LabWAM 推理加速实测（2026-11-14）

本次对 pku4090 上实际部署的 `take_tube/checkpoint_step_30000.safetensors` 做离线推理测速，没有发送机器人指令。正式部署配置尚未切换为这些实验配置。

## 官方实现与适用边界

- [OpenWAM 官方部署配置](https://github.com/OpenWAM-Official/OpenWAM/blob/main/configs/deploy.yaml)：启用架构专用 `torch.compile`、DiT velocity cache（cosine threshold 0.99、最多连续跳过 3 次）、prompt cache；不解码预测视频。双系统 self-attention 的默认编译模式为 `default`。
- [GigaWorld Policy 官方推理代码](https://github.com/open-gigaai/giga-world-policy/blob/main/scripts/inference_openloop.py)：动作分支编译使用 `reduce-overhead`，并提供适用于其注意力结构的 action-only prefix cache。
- [wam.cpp](https://github.com/GigaAI-research/wam.cpp)：提供 GWP05/FastWAM 的 C++/CUDA 推理。当前自定义 LabArchitecture 的技能、手别、目标和历史条件没有现成转换器，不能把其速度直接当作本模型的速度。

现有部署已使用 BF16、prompt embedding cache，并关闭视频 VAE 解码。`scripts/deploy_lab_worker.py` 尚未调用上游 `apply_compile_optimizations`，`lab_low/deployment.py` 尚未向 `generate` 传入 DiT cache。本次通过临时离线程序接入这些已有接口实测。

本模型的 `configs/lab.yaml` 使用 mutual MoT：动作与未来视频 token 交互。关闭 VAE 解码不等于停止视频 DiT 运算；不能直接套用 GigaWorld action-only 路径而保证当前权重行为不变。需要改变注意力结构的优化应作为模型改造另行验证。

## 测试条件

- pku4090：RTX 4090，系统报告显存 49140 MiB；PyTorch 2.5.1+cu124，BF16。
- 同一取试管权重，episode 0 的第 60、160、240 帧；真实记录的三路图像、初始第 0 帧目标框、右手技能和英文 prompt、因果历史、`observation.state`。没有使用机器人当前状态。
- 每个配置先对三个输入分别预热，再各重复 3 次，共 9 个计时请求；同帧固定相同随机种子。
- 每次输出 32 个 16 维动作。计时包含条件准备、历史/参考图像编码、联合去噪和动作恢复；使用 CUDA synchronize。**不包括相机采集、HTTP 传输、控制器等待及机械执行。**
- 测试时暂时停止原取试管模型服务以释放其显存。另一个无关 GPU 服务保留，不改驱动或环境。
- 所有误差均相对同一输入下原实现的模型输出，不是相对真实成功动作；没有测抓取成功率。误差在部署夹爪校正之前统计。
- 样本只有一条轨迹的三个时刻，以下是初步吞吐/数值对照，不是完整技能评估。

## OpenWAM 默认编译模式结果

| 配置 | 中位耗时 | P90 | 相对原 20 步速度 |
|---|---:|---:|---:|
| 原实现，20 步 | 1190 ms | 1193 ms | 1.00× |
| 原实现，6 步 | 412 ms | 412 ms | 2.89× |
| 编译，20 步 | 1017 ms | 1023 ms | 1.17× |
| 编译，6 步 | 362 ms | 364 ms | 3.29× |
| DiT 缓存，20 步 | 414 ms | 416 ms | 2.87× |
| 编译＋DiT 缓存，20 步 | 366 ms | 367 ms | 3.25× |
| 编译＋DiT 缓存，6 步 | 222 ms | 223 ms | 5.37× |

编译实际生效，没有退回未编译路径；首次编译请求约 32.98 秒，已从稳定耗时中排除。条件准备约 47 ms，因此该部分也构成速度下限。

### 输出变化

- 20 步只编译：三个输入中右臂 MAE 最大 0.00097 rad，右臂单项最大差 0.00924 rad，右夹爪单项最大差 0.01728 rad。编译也不是逐位等价。
- 6 步只编译，相对未编译的 **6 步**：右夹爪单项最大差 0.15337 rad，右臂单项最大差 0.01047 rad。
- 20 步 DiT 缓存实际跳过 14/20 次联合网络计算。第 160 帧的右夹爪 MAE 为 0.69915 rad、单项最大差为 1.35604 rad；不能视为无损加速。
- 6 步原实现与 20 步原实现相比，第 160 帧右夹爪 MAE 为 0.52601 rad；减少步数本身也会改变夹爪动作。
- 最快的 6 步＋编译＋缓存相对未编译 6 步，右夹爪单项最大差为 0.33914 rad。相对 20 步，第 160 帧右夹爪 MAE 为 0.62957 rad。

建议优先评估保持步数的编译；缓存和减步的效果需要继续核对夹爪开闭时序与实际操作效果，不能仅用机械臂平均误差替代夹爪评估。正式默认配置本次未改为缓存模式。

计时是“一次生成整个 action chunk”的耗时，不是机器人每个控制周期的耗时。异步执行可以减少等待，但不会把 222 ms 一次的模型调用变成 30 Hz 的新模型预测。

## GigaWorld 编译模式补测

将同一 OpenWAM MoT 编译入口的模式改为 GigaWorld 使用的 `reduce-overhead`，再次加载同一权重和三个输入。第二次未编译基线为 20 步 1185 ms、6 步 411 ms，与第一轮接近。

随后首次 20 步编译预热中，Python 进程发生 segmentation fault，退出码 139；没有完成一次编译推理，因而没有可报告的加速速度。未升级 PyTorch、CUDA 或驱动。该结果只能说明当前 LabWAM/环境组合的这次尝试失败，不能据此判断 GigaWorld 原模型也会失败，也不能在没有调用栈证据时指定崩溃根因。

## 实现位置与复现参数

- 当前模型加载与精度：`scripts/deploy_lab_worker.py`。
- 当前请求路径：`lab_low/deployment.py:InferenceSession.predict`。
- 上游编译入口：`openwam/model/architectures/dual_system/joint_self_attn.py:apply_compile_optimizations`。
- 上游缓存实现：`openwam/deploy/optimizations/dit_cache.py:DiTVelocityCache`。
- 使用 `schedule_sync(video_scheduler, action_scheduler, steps, 5.0)`，`generate` 保留原尺寸 320×384、9 视频帧、33 action frames、`decode_video=False`。每个请求重置 DiT cache，防止跨请求复用速度场。
- 编译配置为 `enabled=True, self_attn={torch_mode: default, dynamic: False}`；缓存为 `DiTVelocityCache(0.99, 3)`。

临时测试程序按项目约定测试后删除；逐次计时、预热及误差原始记录保留在 `validation/inference_acceleration_20261114.json`。原部署服务已按 `take_tube 20` 恢复；服务健康检查结果也记录在该文件中。
