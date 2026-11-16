# 取试管 chunk 平滑诊断：排除等待（2026-11-14）

本轮只诊断抖动，不优化或评判推理速度，不启动模型服务或机器人控制。
输入为已保存的取试管30k模型预测：轨迹0、第60～321帧、10次去噪、32/16调度。
来源为 outputs/async_20261114/take_tube_ep0_f60_async10_smooth10_official_count/episode_000000。
15个完整32步窗口用于录制轨迹平滑性参照，固定重放261个目标，删除所有等待周期。

## 确认的现象

1. **chunk内也有高频小幅变化。**
   原始预测右臂关节二阶差分RMS为7.293 rad/s²，录制参照1.937，约3.77倍。
   增量反号比例为43.65%对11.56%；包含微小变化，不能解释成同等比例的大幅抖动。
   这些是指令空间指标，不是实际电机加速度。原模型反馈来自仿真，
   因此相同数据帧窗口的录制轨迹只作平滑性参照，不作同条件监督误差。

2. **当前边界平滑接住位置，没有匹配速度。**
   完全相同预测的未平滑边界最大单步角度变化为0.1501 rad，offset10为0。
   但15个边界全部出现一个离散零速度点：新块首目标被设置成上一已发送目标。
   平均边界速度变化由2.332降至0.2496 rad/s；速度匹配6步候选为0.0981，
   无这15个强制零速度点，但尚未部署或验证任务效果。
   本轮固定预测的0.1501不能与其他反馈配置下旧视频的0.54直接混用。

3. **当前30Hz到50Hz转换独立产生指令速度脉动。**
   使用真实ArmTargetBuffer、虚拟机器人模型和同一套offset10目标，不经过ROS/DDS。
   当前bridge按接收时刻+20ms到达目标，对照按33.33ms训练时间轴采样，同样输出50Hz。
   二者指令二阶差分RMS分别为17.519和7.926 rad/s²。
   匀速人工输入独立复现40%的50Hz周期重复角度。这是指令采样结果，不是实机停顿比例。

## 离线候选

各完整预测chunk内部对活动右臂做Savitzky-Golay 5点二阶平滑，再裁剪，夹爪不改。
chunk内二阶差分RMS从7.293降至2.610；再结合6步速度匹配，
整段指令二阶差分RMS从原始14.061降至3.362，现有offset10为7.327。
这只是固定预测后处理，无新模型闭环或实机成功证据；滤波可能改变接触时序。

FK附加检查固定其他关节，使用腕部局部X方向10cm的名义夹爪点，不是标定TCP。
原始chunk内该点二阶差分RMS约2.717 m/s²，录制参照0.513；
原始边界最大位移39.17mm。速度匹配后非零相邻位移可以是正常运动，不能只追求目标重复。

## 尚未证实的原因

- Lab每次使用seed=42+last_frame，OpenWAM默认42。这是确认的设置差异，
  未做同观测/同噪声实验，不能认定为根因。
- 当前action latent与flow运算使用bf16，可能有量化影响，但未做精度隔离实验。
- 不能直接断言训练不足。应在同观测、同seed和理想时钟下比较10/20/30去噪，
  保存原始chunk、处理后目标、50Hz指令及实测状态。
- 旧视频动力学直接以30Hz施加目标，不含正式bridge的50Hz插值，不能当真机链路完整复现。

建议先统一仿真与部署时间轴/50Hz采样，再匹配边界速度，最后独立评估chunk内去噪和滤波。
不盲目拉长offset窗口，不因30Hz模型而重训成50Hz。

## 理想仿真时钟

现有离线RPC异步仿真入口新增下列参数，接在原命令末尾整行运行：

```text
--inference-clock ideal
```

缓冲为空且结果未就绪时只等待程序，物理、视频、图像、历史均不推进，不插入HOLD周期。
仍保留32/16选择。默认measured不变，真实控制入口不开放此选项。
4秒预测耗时的虚拟future验证：80个目标只推进80个仿真周期，等待数为0。
原模型服务当前未运行，本轮未重生成模型闭环视频，以免干扰用户加速工作。
固定预测重放不能替代真正加速后、观测反馈也改变的闭环评估。

## 产物与依据

- scripts/analyze_chunk_smoothness.py：统计、真实bridge无传输对照、FK、三张图。
- outputs/smoothness_20261114_take_tube_report/：metrics.json、fixed_predictions.npz及PNG。
- validation/chunk_smoothness_diagnosis_20261114.json：归档数据与验证。
- [GR00T抖动分型](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md#fixing-jittering)
- [Isaac OS插值与过渡](https://github.com/nvidia-isaac/isaac-os/blob/main/operators/gr00t_action_handler/gr00t_action_handler_op.cc)
- [OpenWAM默认seed](https://github.com/OpenWAM-Official/OpenWAM/blob/main/openwam/deploy/engine.py)
