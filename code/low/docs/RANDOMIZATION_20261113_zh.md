# 2026-11-13 随机化实验

四个独立单卡实验均从同一个 OpenWAM Foundation 开始，目标 30000 step，复用 boxed_v3 原训练/验证划分。仅训练集启用增强；输出仍是双臂关节目标和夹爪，不增加移动。

| 实验 | 数据 | 相对于原轻光照取试管模型的变化 |
| --- | --- | --- |
| take_tube_local_light | 取试管 | 在全局光照增强上叠加平滑的局部亮区/阴影；局部增益限制为 0.65–1.35 |
| take_tube_box_robust | 取试管 | 初始框随机平移、外扩或单边收缩；保留原框中心、至少 75% 面积且 IoU ≥ 0.6 |
| general_light_balanced | 原通用五技能 | 原轻光照增强，加按技能窗口数倒数加权的采样，五技能期望概率各 20% |
| take_tube_camera | 取试管 | 原轻光照增强，加小幅相机平移/旋转和低频时序扰动 |

通用模型沿用原五技能：取试管、取滴管并吸液、加液、放回滴管、水浴；不含试管归架。框增强只是几何约束，不能保证不包含紧邻物体。相机增强是二维近似，不能代替真实视角变化的数据。

同一样本的历史、当前、未来图像共享光照参数。相机轨迹按原始 30 Hz 帧索引求值，初始框先验使用初始图像的同一个变换；后续帧仍不提供框。动作、状态、时间戳和实测 IMU 信息保持不变。

实现：`lab_low/augmentations.py`、`lab_low/photometric.py`、`lab_low/dataset.py`；技能均衡采样在 `scripts/train_lab.py`；四个配置在 `configs/experiment/`，入口 `scripts/train_share.py`。

CPU 验证见 `validation/randomization_cpu_20261113.json`：6000 次框几何检查、300 次局部光照范围检查、RGB/先验变换对齐、四种配置的真实数据样本标签不变、验证集不增强及技能采样权重检查均通过。此记录不表示模型已开始训练或机器人效果已验证。

运行必须使用 `/share/personal/yekangjie/lab-low/envs/openwam/bin/python`，代码目录需位于同一项目父目录下。启动器要求 GPU 至少有 76000 MiB 空闲显存，使用每 GPU 文件锁，输出目录包含开始日期。先使用对应实验参数加 `--check` 验证依赖与配置；真实启动和完成情况以服务器日志、step 和 checkpoint 回读记录为准。
