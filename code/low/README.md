# Lab Low — OpenWAM laboratory policy

本项目基于服务器 OpenWAM 提交 `51ec4b66bf1c5124becf8d14aee1dea47416d50e`
（先复用本机 `WM/openwam_deployment/test_tube_20261027` 副本，再补齐并核对源码），
保留原生双分支 MoT、flow matching、Accelerate/DeepSpeed 训练循环。
原项目：https://github.com/OpenWAM-Official/OpenWAM 。许可证见 LICENSE。
实现集中在 `lab_low/`，上游改动只有架构注册；环境沿用服务器 OpenWAM 已安装环境。
当前纯操作版本为 `spacetime_manip_v3`，验收见 [MANIPULATION_VALIDATION.md](MANIPULATION_VALIDATION.md)。
历史模块的早期检查见 [SPACETIME_VALIDATION.md](SPACETIME_VALIDATION.md)。
旧版池化后历史融合的 GPU 训练记录保留在 [VALIDATION.md](VALIDATION.md)，不能视为新版验收。
短训练不代表闭环操作或液体操作阶段识别成功。

## 已落实的数据和模型约定

- 高层只在技能开始给**一次初始框**。永久保留该帧目标参考，不要求逐帧 SAM 框。
  标记文件支持初始原图像素 xyxy 框或同尺寸单通道 mask；点提示不会使用。
- 加液技能 `dispense_into_held_test_tube` 可省略 targets 或传空列表，手别仍必须为 both：
  左手持滴管、右手持试管。无框时 prior/confidence 为零、mark_valid 为 false，
  初始图像不提供目标身份锚点；技能/手别调制后的全画面特征、历史和本体状态继续
  生成双分支条件 token。不会伪造全图框。其他技能仍要求初始框/mask。
  训练 manifest 与 OnlineConditionBuffer.reset 使用同一校验；新技能仍重置历史。
  参数形状不变，可加载原纯操作版权重，但无框加液能力需要相应示范训练验证。
- 初始参考经独立空间/单帧编码后软区域池化，与共享 query 结合，
  搜索已融合历史的当前全画面。技能/手别 embedding 通过 FiLM 调制读取特征。
  初始框先验保留非零背景底值并按置信度缩放身份锚点；不把历史框套到当前帧。
  不裁掉框外信息，也不把旧框坐标复制到新图像。两目标 role 槽位支持操作物与容器/槽位。
- 一直保留 reference，但它不加入滚动窗口的时间注意力。滚动窗口保持
  `t-60,t-40,t-20,t`，对应 Cortex 表 10 公开的 4 帧、stride 20 配置。
  在本批 30 Hz 数据下覆盖两秒；这不是用户指定的时长，也不是已核实的 Cortex 固定时长。
  Cortex 原文没有明确低层历史采样流的帧率；其约 10 Hz 动作生成频率不能替代数据帧率。
  先独立冻结 VAE 编码，再经 4 层可训练空间注意力，第 4 层加入因果时间注意力；
  在目标池化之前融合历史，最后仅保留当前帧空间网格。缺失历史被屏蔽。
  四帧归一化状态与过去实测关节变化分别编码成四个 token，保留时间顺序。
  输出默认 10 个条件 token：技能/手别 1 + 目标 4 + 全局 1 + 滚动状态 4。
  参考 [Cortex 表 10](https://arxiv.org/html/2607.05377#A5.SS2) 和
  [MEM 短时视频记忆](https://arxiv.org/html/2603.03596#S3.SS3)，但本实现是
  完整条件模块共约 4.8M 参数，其中包含新增的 VAE 潜空间适配器；
  不是 MEM 原版零新增参数的 ViT 改造，也不把历史作为独立创新声明。
- 用户确认轨迹均成功完成。通过 observation 动作监督学习准备、执行和完成后的
  行为差异；没有额外完成分类头，不伪造逐帧阶段标签。相似当前画面可以利用近期
  运动区分阶段，但完全不可观测的液体结果或窗口外事件不能保证辨别。
- 单一动作模型，仅包含 7 个操作技能；已删除移动技能和底座速度输出。
  手别条件为 left=0、right=1、both=2；技能、左右手不是复杂路由器。
  固定左手和双手技能由注册表严格检查；未定义的技能会拒绝。
- 新条件 token 同时进入视频和动作专家的 cross attention，保留原有 mutual MoT。
  默认冻结基础视频权重，但梯度仍经视频计算路径流回条件模块。
  LabArchitecture 单独覆盖冻结方法：只设 requires_grad=False，不沿用上游对子树
  递归加 no_grad 的行为，否则视频条件会被截断；真实训练会强制检查此梯度。
- 输出 16D：`[left_arm7,left_Dex1,right_arm7,right_Dex1]`。
  本体输入为完整 31D observation.state；监督为 `state[t+1:t+33]`，只取双臂/夹爪。
  **不读取 action.wbc 或 planner 命令。** 未来实测位置是 achieved-state 标签，不等同原始控制指令。
- 当前试管数据只验证右手操作，不能证明另外六种技能或左手已学会。
- 动作损失按技能/手别屏蔽通道。执行前必须调用 `skills.gate_action`：
  未选中的手保持**当前角度**，不能向关节发送零角度。接口不产生底座命令。
- 抖动公式与 GigaWorld 相同：`floor+(1-floor)/(1+(score/scale)^2)`，
  只降低视频预测 MSE，分母不按抖动权重重新归一化，保留动作监督。
  默认 .025 rad/s、floor .1 是旧实验参数，需按新数据消融调整。
  当前数据没有 gyro，因此从 root_orientation 得角速度代理；不是独立相机 IMU，
  不能检测头部相对底座的全部抖动。可配置真实 observation 抖动分数替换。
  机体正常姿态变化也会提高这个代理分数，阈值需要按操作数据检验。
  未来 IMU 只进 loss，不进入模型前向条件。

## 数据清单

先用 `scripts/prepare_lab_manifest.py` 将 high 的 prompts_v1.json 转换。
每个 segment 包含 dataset、episode_index、start_frame、可选 end_frame（不含末帧）、
skill、hand、binding_id 和 targets。框在 start_frame 生效；此前不采样，窗口不跨技能边界。
按 episode 分割训练/验证，归一化仅拟合训练集；排除 info.json 的 discarded episodes。
三个真实相机按 ego/left wrist/right wrist 顺序等比例补边，标记使用完全相同变换。

多技能数据通过一个 manifest 添加多个 dataset/segment 即可，不需要另建模型。
每个目标对象变更或新技能必须开始新 segment，避免旧目标参考混入。
在线输入使用 `lab_low.runtime.OnlineConditionBuffer`：新技能时 reset 一次传初始框，
随后 observe 只接收图像和本体状态。sample 产生与训练一致的条件字典。
部署时 `LabArchitecture.prepare_condition` 接收与数据集相同的当前/历史字典，
把输出作为 `generate(..., lab_condition=..., active_action_mask=...)` 的参数。
buffer 的 decode_and_gate 使用独立的 31D 状态/16D 动作归一化；不要用原版
服务中共用的动作 normalizer 去归一化 31D 状态。调用 generate 时保持
architecture.normalizer=None，随后只在 decode_and_gate 中反归一化一次。
它**不接受未来画面**。完整机器人网络协议、技能完成判断和水浴 release
仍由高层/控制器负责，不在训练代码里假定已实现。

## ai2-root-4080 运行

服务器代码目录为 `/root/ykj/lab-low`；原环境为
`/root/ykj/OpenWAM/.conda-envs/openwam`，PyTorch 2.7.1+cu128。
脚本默认单 GPU 2，不更改驱动；启动前检查空闲显存，避免挤占正在训练的卡。
通过 CUDA_VISIBLE_DEVICES 选择空闲卡。以下两行依次执行：

```bash
cd /root/ykj/lab-low
bash scripts/train_lab.sh
```

短验证用同一入口。以下是一条命令，不需要换行：

```bash
bash scripts/train_lab.sh training.max_steps=3 training.save_steps=3 validation_every=3 validation_windows=2
```

所有缓存/临时文件在项目 `.cache`，无 /tmp 写入。W&B 默认关闭；打开时 run 名含日期。
项目启动入口只面向单卡；多卡需另做验证，不默认为已支持。

## 权重与验证边界

默认保存 delta checkpoint：包含完整动作分支、本体编码器、条件模块；
冻结视频权重从配置指定的原 Foundation checkpoint 加载，**必须保留该基础权重**。
这避免每次重复写巨大视频权重。配置、tokenizer、normalization stats 随训练保存。
每 200 步保存，最新两份 + 最佳验证 action loss。默认不保存优化器状态；
用 finetune_ckpt_path 从 delta 热启动会重启优化器，不宣称是精确续训。
`spacetime_manip_v3` 与旧 19D 动作模型及旧版历史模块结构不兼容；旧 delta 会明确拒绝加载，
需要从原 Foundation 开始新训练，或加载新版生成的 delta，不能静默丢弃旧权重继续。
归一化文件同样检查版本与维度，新数据缓存保存到 `.cache/lab_data_manipulation`。

日志包含实际 loss、IMU 权重、梯度检查、held-out loss、checkpoint 读回核验。
这些只能确认训练管线连通，不能代替目标绑定鲁棒性、闭环抓取或机器人成功率验证。
首轮应重点比较不同初始框、同类干扰物、框偏差/部分遮挡、无历史和无目标参考。
