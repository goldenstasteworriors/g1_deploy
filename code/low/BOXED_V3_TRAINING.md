# 2026-11-10 四组单卡训练

模型为本目录从 OpenWAM Foundation 修改的 spacetime_manip_v3，动作 16D、状态 31D，
保留双 MoT、技能/手别条件、初始框、时空历史及视频抖动降权。加液不使用框。
不使用相邻 wam 目录中动作分支从零初始化的模型。

输入数据为 lab_primitives_train_20261110_boxed_v3；排除全部 42 段放回试管及 14 段行走。
通用模型包括取试管、取滴管并吸液、加液、放回滴管和放入水浴，数据没有混匀技能。

| 实验 | 训练段 | 验证段 | 训练窗口 | 验证窗口 | 新训练步数 |
|---|---:|---:|---:|---:|---:|
| general | 262 | 46 | 13353 | 2162 | 30000 |
| take_tube | 150 | 26 | 5156 | 815 | 30000 |
| dispense | 35 | 6 | 4224 | 630 | 30000 |
| take_dropper | 37 | 7 | 2381 | 417 | 30000 |

四组均从原始 Foundation 独立开始，单技能模型不是从通用模型继续训练。
按原始 dataset+episode 分组，固定 seed 42，约 15% 组留出并确保每类训练/验证均非空。
单技能子集直接沿用通用模型的同一分组划分。统计仅拟合各实验训练集。
框来自该版 frame 0 原图标注，保留原角色与置信度、待复核状态，不伪造人工批准。
不读取 action.wbc 标签。未操作手由在线门控保持当前姿态。

batch 1、BF16、学习率 5e-5；每 1000 步验证并保存，最新两份 delta 加最佳权重。
验证固定覆盖各技能；验证 flow loss 不等于闭环成功率。
默认没有完整优化器恢复状态；中断后不可把权重热启动称为精确续训。

目标服务器 A800-1，部署根目录 /share/personal/yekangjie/lab-low，代码位于 code 子目录。
最初仅 GPU 3 空闲，因此四组顺序运行。2026-11-10 用户明确要求停止
CaPtain（8604、8605）、Molmo2（8122）、SAM3（8114）服务并并行训练。
核实服务命令后向对应进程组发送 SIGTERM，确认进程退出，GPU 0/1/2
各释放到 81228 MiB 空闲显存；没有停止其他服务或通用模型。
现由独立 Supervisor 分别在 GPU 0/1/2 启动 take_tube/dispense/take_dropper，
master port 分别 29650/29651/29652，每组仍是单卡 30000 步。
GPU 3 上已运行的 general 保持原训练状态。
scripts/train_boxed_experiment.sh 检测并行所有权状态：原队列后续只等待对应模型，
核实最终 30000 步权重读回后返回，不重复训练。
并行状态和日志位于 outputs/boxed_v3_30k/parallel，配置为
configs/boxed_parallel_supervisor.conf；失败不自动重启成新的训练。
队列状态为 code/outputs/boxed_v3_30k/queue_status.json，各实验具有独立日志及输出目录。
scripts/run_boxed_suite.py 以前台子进程启动训练，完成必须核实 step 30000 和权重精确读回。

## 验证状态

四组本机真实数据加载及样本维度检查通过，见 validation/boxed_v3_data.json。
连接重试后已通过 ai2ykj 恢复 A800-1 访问。原 Foundation（约 24 GiB）及
原 OpenWAM 环境（约 7.7 GiB）已复制到部署根目录 assets 与 envs。
Foundation 两端 SHA-256 一致；目标环境已通过 torch、hydra、accelerate、
deepspeed、cv2、pyarrow 导入，torch 为 2.7.1+cu128。
环境沿用原前缀兼容链接，不改环境权限或驱动。

2026-11-10 连接恢复检查时，选定数据的 1244 个文件共 1729302093 字节，
156 个文件已校验一致，剩余 1088 个正在补传；此处是当时快照，不代表实时进度。
scripts/sync_boxed_data.py 在前台运行，按远端 SHA-256 续传，失败 30 秒后重试。
仅全部文件验证通过后才生成 .boxed_suite_ready.json；完整性摘要固定在
configs/boxed_suite_integrity.json。实时传输状态见本机 .cache/boxed_transfer/status.json。

SSH 项目脚本设置 ConnectTimeout=180、ConnectionAttempts=10、
ServerAliveInterval=30、ServerAliveCountMax=10，并复用项目内跳板/目标连接。
不修改用户 SSH 全局配置；SSH 本身的连接重试不等于重复执行失败的远程命令。
目标服务器没有 rsync，数据传输使用 tar 经 SSH；代码使用 Git bundle，
因为目标服务器 GitHub SSH 认证失败。本机统一仓库正常通过 SSH 推送。

configs/boxed_suite_supervisor.conf 管理前台 readiness 进程；入口为
scripts/prepare_and_run_boxed_suite.py。依次等待选定数据完整性标记、复核原权重、
生成服务器数据清单、等待 GPU 3 空闲、执行通用/无框加液各 3 步真实 GPU 检查，
最后进入四组正式队列。阶段状态见 code/outputs/boxed_v3_readiness/status.json。
失败会停止并记录原因，不把失败重启成新训练或假装完成。
本记录不是训练完成报告；当前尚未核实 GPU smoke、正式训练及训练后仿真结果。

## 文件变更

- lab_low/dataset.py：支持显式划分，拒绝同源轨迹跨训练/验证。
- scripts/prepare_boxed_suite.py：标注映射、排除技能及四组可复现数据清单。
- scripts/train_lab.py：固定验证窗口覆盖每种技能。
- scripts/train_lab.sh：缓存默认使用项目目录，允许指定 HF_HOME。
- scripts/train_boxed_experiment.sh：每组 30000 step 单卡训练入口。
- scripts/run_boxed_suite.py：前台顺序运行和可观察的完成/失败状态。
- scripts/a800_ssh.sh：长超时、连接复用及项目内 SSH 策略。
- scripts/sync_boxed_data.py：按 SHA-256 校验的前台数据续传与失败重试。
- scripts/prepare_and_run_boxed_suite.py：数据/权重/GPU 检查及正式队列入口。
- configs/boxed_suite_integrity.json：本次选定数据的固定完整性摘要。
- configs/boxed_suite_supervisor.conf：可观察的项目专用 Supervisor 配置。
- scripts/run_boxed_parallel.py：独立单卡任务、所有权锁和最终权重验证。
- configs/boxed_parallel_supervisor.conf：三个单技能实验的独立受管服务。
- validation/boxed_v3_data.json：四组真实数据检查。
- BOXED_V3_TRAINING.md：本记录。
