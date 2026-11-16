# 正式部署：10 步＋官方默认去噪缓存

2026-11-14，按用户明确选择，将正式部署默认值改为：

- 去噪 10 步，实时相机/数据集图像、同步/异步入口保持一致。
- DiT velocity cache 开启，cosine_threshold=0.99，max_consecutive_skips=3。
- 保留条件特征复用和已有编译/启动预热。
- 显式传入的去噪步数仍被尊重；不更改历史窗口、动作块长度或控制频率。

此前优化报告中“正式缓存关闭、默认20步”描述的是切换前状态。原始实验数字保留，本文件说明当前部署配置。

模型原启动脚本不带步数时默认 10 步。已有 8849 服务运行时不要重复启动。以下是一条命令，不需换行：

```bash
bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_pku_policy.sh take_tube
```

实时相机和数据集图像的原启动入口继续使用；未显式指定 `--denoise-steps` 时均为 10 步，异步入口不再自动改成 6 步。

缓存开关为 `LABWAM_DIT_CACHE`，默认 1。需要手动关闭缓存时，以下是一条命令；其余优化保留：

```bash
LABWAM_DIT_CACHE=0 bash /home/yanjun/ykj/labwam_g1/code/low/scripts/start_pku_policy.sh take_tube 10
```

该改动只调整部署默认值。模型内部 API 的显式选项及离线评估参数不强制重写，训练不受影响。模型服务经预热后进行 HTTP reset/predict/stop 验证；两种机器人启动入口仅运行 `--execute --check-config`，不连接机器人传输、不执行站立或手臂动作。

实际验收：三个启动配置（实时同步、数据集同步、数据集异步）均报告 10 步且 robot_io_opened=false。正式服务 health 为 ready/alive，缓存参数为 0.99/3；连续三次不显式指定步数的 HTTP predict 均使用 10 步、跳过 6 次联合网络计算、返回 32×16 有限动作。首次请求约 248 ms，重复第 0 帧约 210 ms；该短历史接口检查不替代此前完整历史约 254 ms 的基准。

修改文件：`scripts/start_pku_policy.sh`、`scripts/run_g1_operator.py`、`scripts/run_g1_recorded_client.py`、`scripts/run_g1_lab_client.py`、`lab_low/operator_control.py`、`scripts/deploy_lab_worker.py`、`scripts/serve_lab_policy.py`。验证记录见 `validation/deployment_cache_default_20261114.json`。
