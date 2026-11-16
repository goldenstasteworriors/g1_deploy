# G1 双臂 Dex1 部署

需要“当前实机状态 + 数据集图像、初始框、技能及 prompt”时，使用独立入口
`scripts/start_g1_recorded_operator.sh`。轨迹与起始帧选择、播放时序和命令见
[数据集视觉部署说明](G1_RECORDED_DEPLOYMENT_zh.txt)。该入口的 `p` 同时选择姿态和推理任务，
实时相机入口的 `p` 仍只选择初始姿态。

## 当前推荐入口：三次 Enter 的交互部署

不需要另开站立控制程序。本入口在一个控制进程中运行 SONICMJ 下半身站立策略和
LabWAM 双臂/Dex1 控制；不要同时启动旧控制器或 PICO 目标发布器。模型服务随程序启动，
正常退出时一并回收。三相机 SensorServer 仍需按照实际接线独立启动。

当前连接方式：本机通过 Tailscale SSH 连接 pku4090，机器人 DDS 和相机数据仍走
pku4090 的 enp132s0 有线网卡。相机入口默认连接机器人 192.168.123.164:5555，
可用 LABWAM_CAMERA_HOST / LABWAM_CAMERA_PORT 或 --camera-host / --camera-port 覆盖。
SSH 使用的 Tailscale 地址不作为机器人 DDS 网卡或本地模型服务地址。

### 双臂重力补偿

正式相机部署、数据集图像部署与 XR 测试入口现已共用 lab_low/gravity.py 的目标角前馈和官方增益。
默认启用双臂本体及每侧名义550g Dex1-1的重力前馈。使用实际下发的双臂目标角、实测腰角和 pelvis IMU 的 wxyz 四元数，
把世界重力旋转到机身坐标系，用 Pinocchio 的静态重力项计算14个手臂关节的补偿力矩，
按关节映射进入 Unitree LowCmd.motor_cmd[].tau。肩肘位置 Kp/Kd=80/3，三个腕关节40/1.5，
与 Unitree xr_teleoperate G1_29_ArmController 一致；腿、腰及夹爪电机不加此项。
目标角取原启动渐变及插值之后、与位置命令相同的角度；前馈计算使用 RNEA(q,0,0) 等价静态项。
不修改原 SONICMJ 源码或下半身策略。控制周期沿用 SONICMJ；惯量仍按当前本体及显式载荷配置，
不是复制官方 Dex3 手模型。启动日志和 gravity_config.json 显示增益与前馈计算位置。

待机和第一次 Enter 的站立阶段不接管双臂，因此不施加重力补偿；第二次 Enter 开始手臂初始化后
才启用，并在 ready / 推理 / 暂停保持阶段持续计算。d 急停直接清零力矩，不等待平滑衰减；
r 恢复并开始控制双臂后重新启用计算值。按用户要求，不再增加力矩限幅、变化率限制或间隔超时清零。
缺失或非法 IMU/关节状态不会用固定重力方向替代；这不代表已完成实机载荷标定。

动力学在内存中移除 g1_29dof.urdf 每侧170g橡胶手惯量，再加入每侧550g名义Dex1-1载荷；
不改SONICMJ源模型，不用43维三指手惯量冒充Dex1。质心由官方G1-Dex1模型推导，默认标准直装。
CAD夹爪总质量0.364946kg与官网产品标称550g有差异；采用CAD质心及标称总质量，是尚未实测标定的近似。
参数与来源在 configs/dex1_1_payload.json。额外转接件、相机和持物仍需配置。
提供参数后可用 --arm-gravity-payload 指定 JSON；每侧包含 mass_kg 和 com_m，质心以该侧
wrist_yaw_joint 坐标系表达，单位米。该文件按侧覆盖夹爪+全部附件/持物的总外加载荷，不是在默认Dex1上再次相加。

可用 --arm-gravity-scale 调整比例；--no-enable-gravity-compensation 可明确关闭。
旧的 --arm-gravity-max-torque 和 --arm-gravity-slew-rate 参数已移除。
启动和 --check-config 均打印Dex1名义质量/质心、移除的橡胶手质量及未标定附件。

当前共享实现验证见 validation/formal_xr_shared_20261112.json；先前数值检查见
validation/native_controls_only_20261112.json：包含机身倾斜时与浮动基座动力学对照、
势能梯度校验、关节到电机力矩映射、无额外限幅、急停与完整模式切换。未向实机发送补偿力矩。
旧的 arm_gravity_20261112.json 保留为历史记录，其中限幅/渐入测试不再代表当前行为。

以下每个代码块都是一条完整命令，无需手动换行。在 pku4090 的交互终端执行。
先检查配置、真实运动学和下半身 ONNX（不初始化机器人 ROS/DDS，不发电机命令）：

```bash
bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_g1_operator.sh --check-config
```

实际部署入口（默认通用模型；单技能模型可追加 `--model take_tube` 等）：

```bash
bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_g1_operator.sh --execute
```

| 操作 | 程序行为 |
|---|---|
| 刚启动，idle | 读取状态、加载模型；不接管厂商模式，不写身体或夹爪目标 |
| 第一次 Enter | 接管控制、启用零移动指令的下半身策略；双臂 KP/KD 为零，不控制 Dex1 |
| 等到 standing_ready，再按 Enter | 双臂和双夹爪缓慢进入所选轨迹帧的实测姿态 |
| 等到 ready，再按 Enter | 按当前 marks 文件开启新的推理会话；模型未就绪时保持 ready |
| d | 立即锁存阻尼、停用模型目标、终止推理客户端；也可在初始化中使用 |
| 推理中直接 r | 立即停止当前推理，从实测姿态返回所选双臂/夹爪初始姿态；下半身保持站立，不需先按 d；到 ready 后再按 Enter 才重新推理 |
| 阻尼后 r | 重新启用站立，再自动回到所选姿态；等到 ready 后一次 Enter 即重新推理 |
| idle / standing_ready / ready / damped 时 p | 选择并保存动作、轨迹和帧；ready 时提交选择后自动移动到新姿态 |
| Ctrl+C | 阻尼退出，清理本程序创建的推理客户端和模型服务 |

每次 Enter 都应等对应状态提示，不要提前连续按三次。standing_ready 仅表示启用下半身策略
原启动渐变结束，不是物理站稳检测；确认实际站稳后再进入手臂初始化。
`d` 是全身阻尼，不会继续主动站立，需有支撑/保护。`r` 是站立姿态下的恢复流程，
不是跌倒后的自主起身。正常推理、ready、初始化中以及仍站立的暂停/故障状态可以直接按 r 回位，
不重置下半身策略；回位中再按 r 从最新实测姿态重新开始。这里的启动、急停和恢复尚未在真实 G1 上动作验收。

终端现在显示中文阶段：0/3 待机、1/3 下半身、2/3 双臂初始化、3/3 推理。
阶段2每秒显示轨迹进度和实测双臂/夹爪最大误差，误差仅显示，不触发拦截。
ready 表示初始化插值已发送完毕，不判断实际到位；请目视确认后再按 Enter。
fault 不等于阻尼急停，下半身控制仍可能运行。

### 原生控制与手动急停

按用户 2026-11-12 的要求，去掉适配层新增的关节范围拦截、肩侧摆0.35 rad回入特例、
0.5 rad/s手臂限速、0.8 rad/s夹爪初始化限速、初始化误差/收敛等待及超时故障。
初始化从实测姿态插值到所选帧，时长取原 SONICMJ JointSafetyMonitor 的启动渐变配置
（当前非 T-pose 为2秒）；ready 后由操作者确认姿态，不添加自动到位检测。

推理动作直接复用原 SONICMJ InterpolationPolicy，采用原 upper_body_joint_speed 配置
（原默认1000）；保留原 JointSafetyMonitor 告警/启动处理和 Dex1.command_position 的原生规则。
不再重复裁剪 Dex1 目标。原 SONICMJ 源文件未修改。

去掉0.25秒命令过期、0.15秒图像/状态中断故障、75ms采样间隔故障、0.5秒绑定应答期限、
30秒相机初始等待期限、0.1rad推理姿态偏移拒绝、120秒推理HTTP期限及300帧历史溢出中断。
相机仍使用 SONICMJ 原 ImageBundleBuffer 的默认同步规则；无配组图像时等待输入。
命令保持到新目标或明确停止，不因延迟自动清空。历史按模型确认进度回收，不因固定队列上限终止。
默认持续推理，只有明确传入 --chunks 才按次数结束，不将次数结束声称为技能成功。

d 的原键盘回调独立于模型HTTP、相机采集及厂商模式RPC；保留阻尼锁存、取消未处理按键、
停止客户端及 epoch/binding 隔离，防止迟到动作在急停/恢复后重新生效。
保留维度、有限数值、技能/手别和消息身份检查；它们是数据协议，不是新增动作阈值。
厂商模式请求恢复原5秒单次RPC超时和持续释放重试，不再加10次上限。
模型就绪探测以及退出时回收子进程仍有等待时间，但不会因它们自动中断机器人动作。

验证记录见 validation/native_controls_only_20261112.json。
旧 initial_pose_reentry_20261112.json 的受限回入路径已经废弃，仅保留历史记录。

### 选择轨迹初始姿态

默认取试管 episode 0、frame 0。姿态库来自 boxed_v3 数据集全部 350 条轨迹的
`observation.state`（145608 帧），只取双臂14关节和左右 Dex1 两个角度；不读取 action，
也不把轨迹中的腿/腰状态作为控制目标。只校验姿态维度和有限数值，范围告警由原 SONICMJ 处理。

按 `p` 后屏幕显示动作编号、各轨迹编号及合法帧范围，输入三个数字并按 Enter：

```text
动作编号 轨迹编号 帧编号
```

例如选择取试管第0条轨迹第0帧，三个数字在同一行输入，末尾按一次 Enter：

```text
1 0 0
```

当前动作编号：0 加液、1 取试管、2 放入水浴、3 归还滴管、4 试管归架、5 取滴管吸液。
轨迹编号是数据集全局 episode，帧编号从0开始；不是每种动作内重新编号。Backspace 修改，
Esc 取消。选择后持久保存，下次 `r` 和重启程序均沿用：

```text
/home/yanjun/ykj/labwam_g1/runtime/selected_initial_pose.json
```

姿态库在 assets/initial_poses/catalog.json 与 catalog.npz，总计约6 MB，带 SHA256 校验。
初始化沿用原启动渐变时长；插值发送结束进入 ready，实测误差只用于显示。
姿态帧只规定关节角，不代表物体也已经在相应位置或夹爪中。

### 推理技能及初始框

`p` 只选择初始姿态，不会修改推理技能或模型权重。每次第三次 Enter（或恢复后的 Enter）
重新读取以下文件；初始化并稳定后，根据当前真实画面更新初始框，再开始推理：

```text
/home/yanjun/ykj/labwam_g1/runtime/marks.json
```

加液无需框，文件内容为：

```json
{"skill":"dispense_into_held_test_tube","hand":"both","targets":[]}
```

取试管使用 `pick_up_test_tube`、`right` 和初始框。以下是结构说明；将 x1/y1/x2/y2
替换为当前 ego 原图中的实际数字后保存，不能直接复制这些占位符作为 JSON：

```text
{"skill":"pick_up_test_tube","hand":"right","targets":[{"role":0,"camera":"observation.images.ego_view","box":[x1,y1,x2,y2],"confidence":0.5}]}
```

只在推理 reset 时提供初始框，后续用当前图像和历史。更换姿态或移动物体后不要沿用失效的框。
本入口没有自动成功检测，默认持续执行，使用 d 停止；模型/相机客户端异常退出时清空目标，
进入 fault/paused，按 r 回到初始姿态后才能重启。旧会话目标通过 epoch 检查拒绝。

已完成真实运动学/ONNX + 模拟身体/Dex1 通信的全生命周期验证，报告见
validation/operator_lifecycle_20261112.json；pku4090 复测及真实 GPU 服务启停结果见
validation/pku4090_operator_20261112.json。加载中退出和加载完成后退出均已确认回收模型子进程。
这些验证不等同于实机闭环操作成功。
本次交互入口是部署后的源码更新，原 ModelScope 归档尚不包含这一更新和姿态库。

以下分进程入口保留用于诊断，正常部署优先使用上面的统一入口。

pku4090 有线地址 yanjun@192.168.123.222，项目根目录：

```text
/home/yanjun/ykj/labwam_g1
```

OpenWAM 改进版 spacetime_manip_v3：31D 实测状态、三路 RGB、技能/手别、一次初始框及
t-60/t-40/t-20/t 历史，输出32×16双臂和夹爪目标。无移动输出、无成功分类头；加液无需框。

## 来源与布局

两个私有 ModelScope 仓库：
- [Foundation 和四组30k权重](https://modelscope.cn/models/ykjykj/LabWAM-BoxedV3-30k-20261111)
- [部署代码、Dex1控制器和下半身ONNX](https://modelscope.cn/models/ykjykj/LabWAM-G1-Deployment-20261112)

两个仓库均已下载回 pku4090 并校验；权重包67个文件约39.3GB。网络差时从 ModelScope 经 SSH 流式
下载到服务器，不以本机旧权重替代。所有载荷按 MANIFEST 校验 SHA256。

| 目录 | 内容 |
|---|---|
| code/low | 运行源码 |
| assets/modelscope_bundle | 原始 Foundation 和四组模型 |
| assets/deployment_bundle | ModelScope部署归档 |
| runtime/models | 重定位配置及权重软链接 |
| sonicmj | Dex1解耦控制器快照 |
| logs | 下载与推理日志 |
| .venv_inference | 项目环境，继承现有wam的只读依赖 |

当前环境基于已有 wam 的 PyTorch 2.5.1+cu124；额外依赖只装项目环境，没有改驱动或原wam。
原wam仍是运行依赖，删除后本环境不能独立运行。最终版本和GPU结果记录在validation中。

## 已训练组合

| 名称 | 技能/手别 |
|---|---|
| general | 右手取试管、左手取滴管吸液、双手加液、左手归还滴管、右手放入水浴 |
| take_tube | 右手取试管 |
| dispense | 双手加液 |
| take_dropper | 左手取滴管吸液 |

此包不含混匀和试管归架，不支持左手取试管。运行时按
configs/deploy_g1_capabilities.json 拒绝未训练的技能/手别组合。

## pku4090推理服务

以下每个代码块是一条完整命令，无需手动换行：

```bash
LABWAM_BIND=192.168.123.222 bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_pku_policy.sh general 20
```

general可替换为take_tube、dispense、take_dropper。默认GPU0，端口8849，一次启动一个模型。

```bash
curl --noproxy '*' http://192.168.123.222:8849/
```

服务没有电机执行权。POST /rpc 串行处理reset、observe_batch、predict、stop；
每个观测为连续30Hz帧号。错误清空绑定，必须重新reset。初始框只随reset提交，后续
只补图像/实测状态。图像为三路原始RGB的Base64，坐标为原图像素xyxy。

## SONICMJ接入

采用本机thinkpad-wired-pico-onboard工作树快照，修订号写入发布MANIFEST。
这是decoupled_wbc中的GEAR-WBC Balance/Walk ONNX解耦分支，不是SONIC C++全身latent入口。
旧分进程入口保留原CollectionMode；新的交互入口替换模式切换层，复用原阻尼接口和下半身策略。
原get_action最终填回腿和腰的15个关节，LabWAM不获得腿腰目标控制权。

输入31D为body29 motor顺序加左/右Dex1实测角度；输出16D为左臂7、左Dex1、右臂7、右Dex1。
单位均为弧度。原运动学是43D，通过robot_model的body actuator索引映射，不能直接取前29项。
Dex1使用原command_position连续角度接口，不经过旧二值抓取转换。

非工作手保持接管时或上一条有效命令的目标位置，不逐帧追随实测角；手臂复用原 SONICMJ 插值器，Dex1继承反馈±0.18rad单步限制。
无新增命令过期看门狗；手动停止或退出当前推理会话后撤销旧绑定。
机器人端客户端与控制循环部署在同一主机，不得同时运行PICO目标发布器。
保留原控制器唯一DDS写入者。

机器人端继续使用现有decoupled_vla_collection环境和三相机SensorServer（端口5555）。
控制启动器默认diagnostic_read_only；客户端默认只观测和预测。两者分别添加--execute才
允许写电机/发布目标，仍需原控制器人工进入running。此部署没有完成真实G1动作验收。

下面示例在机器人控制主机执行，DEPLOY_ROOT须替换为实际解压位置；每行分别执行：

```bash
export DEPLOY_ROOT=/home/yanjun/ykj/labwam_g1
```

```bash
conda activate decoupled_vla_collection
```

仅检查导入，不初始化ROS/DDS：

```bash
python "$DEPLOY_ROOT/code/low/scripts/run_g1_lab_controller.py" --sonic-root "$DEPLOY_ROOT/sonicmj" --check-imports
```

运行控制循环时去掉--check-imports；机器人网卡等参数沿用现场SONICMJ配置，转传原
ControlLoopConfig。先只读检查，再在现场添加--execute。原raw键盘模式的]启用、o阻尼保留。
不要启动第二个DDS写入进程。

加液标记保存为项目目录的dispense_marks.json：

```json
{"skill":"dispense_into_held_test_tube","hand":"both","targets":[]}
```

取试管为pick_up_test_tube/right，targets使用初始帧实际像素框，格式如下。
x1等须替换为数字，不能直接复制为合法JSON；不能复用其他轨迹的框：

```text
{"role":0,"camera":"observation.images.ego_view","box":[x1,y1,x2,y2],"confidence":0.5}
```

仅观测和推理：

```bash
python "$DEPLOY_ROOT/code/low/scripts/run_g1_lab_client.py" --sonic-root "$DEPLOY_ROOT/sonicmj" --marks "$DEPLOY_ROOT/dispense_marks.json" --chunks 1
```

现场控制器running后添加--execute才发布上肢目标。默认执行一个动作块前8帧；
--chunks是动作块数量，不代表任务成功次数。完成判断保留给外部流程。

## 时序与验证边界

模型加载和同步去噪调用 OpenWAM；动作生成经当前训练模型输出32×16目标。
执行逻辑与 openwam/deploy/executors/sync_executor.py 的 inference_horizon 一致：
只执行一块的前缀，丢弃尾部，再根据新观测预测。但 G1 客户端是单独适配的，
没有直接调用 WAMPolicy/SyncInferenceExecutor，也没有使用 AsyncInferenceExecutor。
32帧来自本项目训练配置，默认前缀8帧是本部署设置，并非 OpenWAM 固定默认值
（其通用配置 inference_horizon=null 会执行整块）。当前按30 Hz执行8帧约0.267秒，
然后等待下一次推理，等待时间另计；不是持续30 Hz生成新动作，也未启用异步延迟补偿。

采用停等：推理期间持续保持上一条有效上肢目标，三相机/状态持续30Hz采集，执行短前缀再推理。
接管时只初始化一次保持目标，后续测量只用于反馈和重力补偿；不把每次测量回写为位置目标。
否则下沉会让目标一起下沉，削弱位置闭环，即使重力补偿开启也可能因未建模载荷而持续下垂。
急停释放旧目标；恢复时重新从实测姿态初始化。未新增限位、限速或超时看门狗。
不增加推理期间手臂偏移阈值。不能声称模型30Hz实时推理。
相机软件配组最大30ms偏差，并非硬件同步；状态按最新接收数据配对。
三相机完整键分别为observation.images.ego_view、observation.images.left_wrist、
observation.images.right_wrist。

本机与pku4090的CPU检查均使用真实G1 Pinocchio模型与Balance ONNX，验证下半身输出保留、
非工作手保持、原生插值、急停锁存和旧会话隔离；当前已去掉新增限速及超时清空。
四模型GPU和有线HTTP回放已通过，详细记录见 validation/pku4090_deployment_20261112.json。
记录图像推理不等于闭环抓取成功；真实相机、夹爪方向、碰撞/接触和现场急停需要接入G1后验收。

## pku4090 项目内控制环境与启动入口

完整控制环境已从本机验证过的 decoupled_vla_collection 用 conda-pack 迁至
.conda_controller，部署时未修改原环境。打包排除了指向本机工作树的 editable 入口；
控制源码和 Unitree SDK 来自下载的 SONICMJ 快照。打包工具和固定目标前缀的用法参见
[conda-pack 官方说明](https://conda.github.io/conda-pack/cli.html)。
原环境的历史 setuptools/GUI 缓存文件缺失在打包时跳过，已在目标机器复测实际控制导入、
运动学/ONNX 和隔离 ROS 回放。该环境包的前缀固定在当前项目，不能直接挪到另一绝对路径。
依赖环境迁移与 ModelScope 权重下载是两个过程；模型权重、控制 ONNX 和部署源码均来自 ModelScope。

在 pku4090 检查完整配置（不连接机器人 I/O），每个代码块一条命令：

```bash
bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_g1_controller.sh --check-config --interface enp132s0
```

连接 G1 后，先以只读方式读取机器人状态；enp132s0 是当前有线网卡名。
真正控制时另加 --execute，并使用原控制器的人工模式切换：

```bash
bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_g1_controller.sh --interface enp132s0
```

客户端使用同一个项目环境。先以默认只观测模式运行，实机条件核验后才加 --execute：

```bash
bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_g1_client.sh --marks /home/yanjun/ykj/labwam_g1/dispense_marks.json --chunks 1
```

三相机使用 SONICMJ 原 camera.composed_camera 服务；需要在实际接入相机后配置各自
设备类型和序列号/USB路径。不能用电脑上其他摄像头的设备号替代。已打包的入口可查看参数：

```bash
PYTHONPATH=/home/yanjun/ykj/labwam_g1/sonicmj /home/yanjun/ykj/labwam_g1/.conda_controller/bin/python -m gear_sonic.camera.composed_camera --help
```

重建推理项目环境时使用下列已在 pku4090 验证的入口。它要求原 wam 基础环境仍在，
仅向 .venv_inference 安装固定版本覆盖，不升级 torch 或驱动：

```bash
bash /home/yanjun/ykj/labwam_g1/code/low/scripts/setup_pku_inference.sh
```

完整下载后，可重新检查权重包：

```bash
/home/yanjun/ykj/labwam_g1/.venv_inference/bin/python /home/yanjun/ykj/labwam_g1/code/low/scripts/download_deployment_bundle.py --bundle /home/yanjun/ykj/labwam_g1/assets/modelscope_bundle --verify-only
```

部署归档使用同一工具加 --kind deployment，并将 --bundle 指向 assets/deployment_bundle。
首次下载去掉 --verify-only，由标准输入提供 ModelScope token，不要放进命令行参数。
网络中断可用 relay_modelscope_bundle.py 的 --resume-sparse 复用同一清单下的已有数据块；
只有整文件 SHA256 匹配才提交文件，不将“占用过的块”当作校验通过。

## 2026-11-12 实测结果

- 权重包67个文件全部SHA256匹配，原始下载未修改。
- 通用模型5种技能、3个单技能模型，共8组真实图像/实测状态预测，全部输出有限32×16动作。
- 使用第60帧及完整历史；取放有初始框，加液无框；非工作手保持检查通过。
- 256个预测动作均通过实际G1控制桥的关节范围检查；这不代表接触或避碰成功。
- 有线端到端回放使用真实GPU模型、隔离ROS域、回放相机和固定记录状态，2轮预测/142条上肢
  目标/1条停止消息均通过；没有连接G1电机。
- 目标机器是48GB版4090。推理时整卡显存读数约25025–25027MiB（含桌面等开销）。

| 模型 | 本次predict内部耗时 |
|---|---|
| general | 首次2.18s，后续4种技能1.27–1.28s |
| take_tube | 首次1.83s |
| dispense | 首次1.83s |
| take_dropper | 首次1.82s |

这些耗时不含启动加载、HTTP传输与observe_batch补帧。20步去噪，未编译。
通用模型另一次有线回放为1.81s/1.22s。采用停等执行，不是30Hz模型推理。
测试完成后已关闭测试策略服务，不常驻占用GPU；按前文命令启动需要的模型。

当前物理相机设备与真实G1动作尚未验收。初始框必须对应reset提供的原始图像；
CLI手工标框流程须在当前画面上标记并保持机器人/相机位置，不能复用旧轨迹的框。
高层程序可通过reset协议原子提交同一帧的图像、状态和初始标记。

## USB 直连与机器人网口并存（2026-11-12）

USB 直连使用独立的 `192.168.124.0/30` 网段，避免与机器人网段重叠：

| 设备 | 网口 | 地址 | 用途 |
|---|---|---|---|
| 本机 | enp7s0 | 192.168.124.1/30 | 连接 yanjun 的 USB 网卡；原 192.168.123.221/24 保留 |
| pku4090 | enx00e04c36c4e8 | 192.168.124.2/30 | 本机 SSH、文件传输 |
| pku4090 | enp132s0 | 192.168.123.222/24 | 原机器人连接，未修改 |

两端通过 NetworkManager 保存配置。pku4090 新连接名为 `labwam-usb-direct`，
开启自动连接，不设置网关或 DNS，`ipv4.never-default=yes`；本机在已有
`Wired connection 1` 中增加直连地址，并保留原地址和不设默认路由的设置。
联网继续走 Wi-Fi，USB 不承接默认路由，不开启网络共享或转发。

从本机执行下面一行即可通过 USB 登录，无需换行：

```bash
ssh yanjun@192.168.124.2
```

已经通过有线 SSH 和三次 Ping 检查（平均约 0.9 ms，无丢包），并确认
`192.168.123.0/24` 在 pku4090 上仍通过 `enp132s0`、源地址仍为
`192.168.123.222`。本项检查没有启动机器人控制或模型推理。

## 补充单技能权重

以下模型复用 `assets/modelscope_bundle/foundation`，原四组模型及其配置保留。
原始下载文件不修改；运行目录复制小文件、重定位 Foundation 路径，并软链接权重。

| 模型目录名 | 技能与手别 | 最终步数 | 最佳步数 |
|---|---|---|---|
| return_dropper | 左手放回滴管 | 30000 | 30000，与最终权重相同 |
| water_bath | 右手将试管放入水浴 | 30000 | 21000 |
| return_tube | 右手试管归架 | 30000 | 20000 |
| take_tube_light | 右手取试管，训练开启光照增强 | 30000 | 26000 |

前三组来自 [Additional 仓库](https://modelscope.cn/models/ykjykj/LabWAM-Additional-30k-20261112)，
原始文件位于 `assets/modelscope_additional_20261112`。
光照增强版来自 [TakeTube-Light 仓库](https://modelscope.cn/models/ykjykj/LabWAM-TakeTube-Light-30k-20261112)，
原始文件位于 `assets/modelscope_take_tube_light_20261112`。
成功部署后的模型目录为 `runtime/models/<模型目录名>`。

下载程序按仓库 MANIFEST 校验每个载荷文件的大小及 SHA256，再校验共用 Foundation、
检查 checkpoint 元数据和归一化统计，最后生成运行配置。完成标志分别是
`runtime/additional_staging_20261112.json` 和 `runtime/take_tube_light_staging_20261112.json`
中的 `runtime_staged=true`；下载日志中的进度数字不能替代完成标志。
本步骤仅部署文件，没有运行 GPU 推理或机器人控制。

现有操作器及 `start_pku_policy.sh` 的模型选项仍为原四组；补充文件不会自动切换正在使用的模型，
也不会将试管归架能力添加到通用模型。使用新单技能模型时需接入对应的显式 checkpoint 路径。

2026-11-12 已在 pku4090 完成上述四组的下载、SHA256 校验和运行目录配置；
新增七份 checkpoint，共约 14.45 GB（含配置、tokenizer 等载荷）。
验收记录为 `validation/pku4090_additional_download_20261112.json` 和
`validation/pku4090_take_tube_light_download_20261112.json`。
已确认原四组及新增四组的最终权重、配置和归一化文件都存在。
