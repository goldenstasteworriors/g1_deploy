# 正式部署控制记录

正式相机部署和数据集图像部署默认开启记录，启动命令不变。重启控制程序后生效。
本次不增加动作阈值、看门狗或自动停止条件。

## 文件位置

在 pku4090 项目目录下：

- 相机部署：`logs/operator/control_trace_日期_时间.jsonl`。
- 数据集图像部署：`logs/recorded_operator/control_trace_日期_时间.jsonl`。
- 每次推理会话的模型动作块：同目录 `inference_轮次_时间戳.log` 中的 `model_prediction` JSON 行。
- 旧分进程控制入口：`logs/controller/control_trace_日期_时间.jsonl`，支持 `--trace-dir` 指定本项目其他记录目录。

启动时打印实际路径；每次控制程序、推理会话生成独立文件，不覆盖旧记录。
模型日志记录所有32步，包括未执行的尾部；`execute_prefix` 表示实际选择的前缀长度。

## 对齐方式与字段

`binding_id` 区分推理会话；`prediction_id` 区分动作块；`action_index` 为块内从0开始的动作编号。
控制记录中的 `model` 字段携带后两个编号及对应模型目标。
`command_sequence` 是客户端发送序号，控制器50Hz和客户端30Hz并非一对一；同一动作可以有多次电机发布。
初始化和模型尚未返回时，`model` 可以为空；等待期间保持上次动作，继续记录反馈与下发命令。

| 字段 | 含义 |
| --- | --- |
| `requested_target16` | 客户端请求的16维目标，尚未经过控制器插值和启动渐变 |
| `model.model_target16` | 策略返回的该步目标：反归一化及技能/手别门控后的16维输出 |
| `sent_arm_q` | 实际传给身体 DDS 发布器的14个手臂目标角，已经过插值和原启动处理 |
| `measured_arm_q` / `measured_arm_dq` | 本轮下发前最近一次观测的14关节角/速度 |
| `measured_gripper_q` | 同次观测的左右 Dex1 实测角 |
| `sent_arm_kp` / `sent_arm_kd` | 同一电机消息中的实际增益 |
| `sent_arm_tau` | 同一电机消息中的前馈力矩；不是电机反馈力矩，也不是 PD 加前馈的总力矩 |
| `epoch` / `phase` | 控制器推理轮次和阶段 |
| `observation_monotonic_ns` | 本机读取观测后的时间，并非传感器硬件采样时间 |
| `send_started_monotonic_ns` / `send_finished_monotonic_ns` | 本机 DDS Write 调用的起止时间 |
| `wall_time_ns` / `monotonic_ns` | 记录生成时间；单调时钟用于同机进程间关联 |
| `local_write_ok` | 本地发布器返回值；不表示机器人电机已收到或执行。空值表示发布器没有返回状态 |
| `damping` | 阻尼锁存状态；急停消息的实际零Kp、零前馈和原Kd=8也会记录 |

第一行 `schema` 给出14个关节名、电机槽位、单位和16维目标排列。
14维顺序是左臂7关节、右臂7关节；16维是左臂7、左夹爪、右臂7、右夹爪。
身体 DDS 记录不包含夹爪驱动器内部的限步后目标和 Kp/Kd；夹爪这里只记录模型/请求目标及实测角。
遥测在发布边界捕获，再交给独立线程写盘；急停不等待日志写盘。
磁盘/序列化异常会明确打印“控制记录失败”，不静默声称记录完整，也不改变控制动作。
退出时先执行原阻尼流程，再刷新关闭日志。强制断电或 SIGKILL 可能损失尚未写入的末尾记录。

验证见 `validation/control_trace_20261112.json`：实际 LowCmd/CRC 序列化、发布器捕获、字段关联、
阻尼记录及录制图像动作编号传递均使用离线替身检查，没有向机器人发送动作。
