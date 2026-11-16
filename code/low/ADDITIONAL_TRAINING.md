# 四个补充单技能实验

沿用 boxed_v3 数据、原 Foundation 初始化及 spacetime_manip_v3，每组单卡 30000 步。
不是继续训练昨天的 30000 步权重。监督仍取 observation.state，16D 操作输出，
仅初始目标标记、技能和手别条件、时空历史、抖动降权均保留。

| GPU | 实验 | 训练/验证段 | 光照增强 |
|---|---|---|---|
| 0 | return_dropper：放回滴管 | 34/6 | 关 |
| 1 | water_bath：放入水浴 | 6/1 | 关 |
| 2 | take_tube_light：取试管 | 150/26 | 开 |
| 3 | return_tube：试管归架 | 35/7 | 关 |

原有通用模型清单不变。新的三个技能沿用相同源轨迹划分；归架新增源轨迹用固定哈希划分，
已有源轨迹严格保留旧划分。归架使用右手及已有目标放置区域框，保留标注的待复核状态，
不把这些框标为已经人工批准。水浴数据量很少，验证损失不能代表广泛的泛化或任务成功率。

光照增强仅作用于取试管训练集：80% 样本增强、20% 原图。亮度 0.8–1.2、
对比度 0.85–1.15、饱和度 0.8–1.2、Gamma 0.85–1.15、各 RGB 通道增益 0.92–1.08；
每相机水平/垂直平滑照明梯度各 ±0.15；增强样本中 10% 另加 0.2–0.6 像素高斯模糊。
同一窗口内初始参考、历史、当前图像和未来视频标签使用同一参数；同一 epoch/index 可复现，
跨 epoch 改变。没有几何增强、翻转、遮挡擦除或虚构反光，不改状态/动作/目标框。
验证和在线推理默认关闭这些增强。它模拟曝光/色温/平滑照明变化，不等同于真实门窗反射建模。

所有远端代码、环境、数据、Foundation、缓存与输出均位于 `/share/personal/yekangjie/lab-low`。
首次准备新清单，执行下面这一行：

```bash
/share/personal/yekangjie/lab-low/envs/openwam/bin/python /share/personal/yekangjie/lab-low/code/scripts/prepare_additional_suite.py --dataset /share/personal/yekangjie/lab-low/data/lab_primitives_train_20261110_boxed_v3
```

四卡前台启动（下列为一条命令，无需换行；仅在没有同组训练运行时执行）：

```bash
/share/personal/yekangjie/lab-low/envs/openwam/bin/python /share/personal/yekangjie/lab-low/code/scripts/run_additional_suite.py
```

末尾加 `--check` 做同一入口的四组依赖/文件/配置检查；不构建 GPU 模型。
真实训练每卡使用独立锁与 29700–29703 端口，每 1000 步验证保存，保留最新两份和最佳 delta。
程序不自动重启失败任务，也不把权重热启动伪称为优化器精确续训。W&B 关闭，运行名含日期。
父进程前台管理四个子进程；状态写入 `outputs/additional_launches/<日期时间>/status.json`，
每项的启动日志指向 `outputs/share_runs/<实验>/<日期时间_PID>/train.log` 与 weights 目录。
确认最终 30000 步权重逐张量读回一致后才写 completed.json。
