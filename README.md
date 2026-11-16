# G1 LabWAM 部署项目

这是 pku4090 上 `/home/yanjun/ykj/labwam_g1` 的正式部署快照，导出日期 **2026-11-16**。
`code/low` 对应原统一仓库提交 **353faea**，保持正式运行的控制逻辑；没有混入本机后续未部署的训练实验。

包含 OpenWAM 改进版、SONICMJ 双臂 Dex1 控制源码、G1 模型与网格、Balance/Walk ONNX、Unitree Python SDK、初始姿态库、部署机已有的示例轨迹图像/框/视频、启动脚本和中文说明。
大模型权重、Python 环境本体、运行日志和密钥不在 Git 中。所有来源和逐文件 SHA256 见 `provenance/`。

## 目录

| 路径 | 内容 |
|---|---|
| `code/low` | 实际运行的 LabWAM/OpenWAM、模型服务、实机与数据图像入口 |
| `sonicmj` | 已部署的解耦控制器和原生急停、相机代码、运动学与下半身 ONNX |
| `assets/initial_poses` | 350 条轨迹的双臂/夹爪初始姿态库 |
| `assets/datasets` | 部署机实际已有的示例数据，非完整训练集 |
| `runtime/warmup` | 已有取试管模型预热请求，不发机器人动作 |
| `environment` | 两个实际环境的 Python 和包版本清单；不是跨机器安装锁文件 |
| `docs/pku4090` | 部署机原有中文 TXT 指令，原绝对路径保留作历史参考 |
| `scripts/prepare_runtime.py` | 将已下载的模型配置重定位到当前 checkout；不覆盖已有 runtime 模型 |

## 权重下载

使用有权限的 ModelScope 账号；本仓库不携带凭据。模型仓库是私有的：

- [Foundation、通用模型及取试管/加液/取滴管](https://modelscope.cn/models/ykjykj/LabWAM-BoxedV3-30k-20261111)
- [归还滴管、试管归架、水浴](https://modelscope.cn/models/ykjykj/LabWAM-Additional-30k-20261112)
- [带光照增强的取试管](https://modelscope.cn/models/ykjykj/LabWAM-TakeTube-Light-30k-20261112)（准确仓库 ID 以 `provenance/model_repositories.json` 为准）
- [原始 SONICMJ 部署归档](https://modelscope.cn/models/ykjykj/LabWAM-G1-Deployment-20261112)

在安装了 `modelscope_hub` 的项目 Python 环境中，下面是一条命令，无需手动换行；脚本从标准输入读取 ModelScope token，下载后按 MANIFEST 校验全部文件：

```bash
python code/low/scripts/download_deployment_bundle.py --bundle assets/modelscope_bundle
```

不要用下载归档里的旧代码覆盖本 Git 仓库。下载完成后，在含 `omegaconf` 的项目环境中执行一行：

```bash
python scripts/prepare_runtime.py --bundle assets/modelscope_bundle
```

如另外下载了附加模型，可在首次准备 runtime 时追加 `--extra-bundle assets/modelscope_additional_20261112` 等参数。
附加下载同样需要按其 MANIFEST 校验；原下载器的 `--kind deployment` 仅用于部署归档，不能拿来假冒附加权重包。

## 在现有 pku4090 上使用这个 checkout

先克隆并进入目录，以下两行分别执行：

```bash
git clone git@github.com:goldenstasteworriors/g1_deploy.git
cd g1_deploy
```

复用现有环境与已经从 ModelScope 下载、校验过的模型，不需要重新安装或复制大权重。仅在新 checkout 中执行下面三行；不会改动现有环境内容：

```bash
ln -s /home/yanjun/ykj/labwam_g1/.conda_controller .conda_controller
ln -s /home/yanjun/ykj/labwam_g1/.venv_inference .venv_inference
.venv_inference/bin/python scripts/prepare_runtime.py --bundle /home/yanjun/ykj/labwam_g1/assets/modelscope_bundle --extra-bundle /home/yanjun/ykj/labwam_g1/assets/modelscope_additional_20261112
```

这两个环境有外部依赖；跨机器不能直接复制上述软链接。新机器需要按 `environment/` 清单与原部署说明建立对应环境。
`sonicmj/install_scripts/install_decoupled_vla_collection.sh` 是完整 SONICMJ 数据采集安装来源，不是本裁剪部署快照的一键安装器；不要直接运行它并假定缺失的全项目依赖已经包含。

先执行配置检查（不打开机器人控制 I/O），一行：

```bash
bash code/low/scripts/start_g1_operator.sh --model take_tube --check-config
```

真实图像入口，一行：

```bash
bash code/low/scripts/start_g1_live_operator.sh --model take_tube --episode 0 --frame 0 --target-description "the rightmost transparent test tube in the rack" --denoise-steps 10 --execute
```

数据集图像入口，一行：

```bash
bash code/low/scripts/start_g1_recorded_operator.sh --model take_tube --episode 0 --frame 0 --denoise-steps 10 --execute
```

启动后：第一次 Enter 站立 → 等待 → 第二次 Enter 双臂初始化 → 实时图像模式按 **b** 取图打框并等待成功 → 第三次 Enter 推理。数据图像模式使用数据里的初始框，不需要 b。
**d** 全身阻尼；**r** 按既有流程回初始化；**p** 选择姿态。r 不是跌倒后的自主起身。
同步推理、10 步去噪、官方默认 DiT 缓存；只控上半身双臂/Dex1，下半身由 SONICMJ 控制。

默认机器人网口 `enp132s0`，相机地址 `192.168.123.164:5555`。可用 `LABWAM_G1_INTERFACE`、`LABWAM_CAMERA_HOST`、`LABWAM_CAMERA_PORT` 覆盖。
在线自动框依赖另行运行的 Qwen 服务，默认 `http://127.0.0.1:19113`；本仓库包含客户端，不包含远端高层模型服务或其权重。
克隆仓库不会建立此前依赖 ykj 本机的 SSH 转发，需要自己提供对应服务或使用 `--grounding-url` 指向可达的现有服务。

## 验证与已知问题

本次导出核对正式源码、资产哈希、Python/脚本语法和无动作配置检查，不重新执行机器人动作。
历史验证详见 `code/low/validation`，旧文档描述的是各自日期的验收范围，不能等同当前实机闭环成功。
2026-11-14 的实时取试管曾出现接近动作不足；另一次在 ready 阶段全身电机失能、11 号电机报告 `0x200`。这两个问题未因上传仓库而解决，不能将快照视为已完成实机可靠性验收。
未新增阈值、看门狗或改变急停/重力前馈逻辑。

## 许可证

LabWAM/OpenWAM 保留 `code/low/LICENSE`；SONICMJ 保留 `sonicmj/LICENSE`；SDK 保留其目录内许可证；下半身 ONNX 使用同目录的 NVIDIA Open Model License。第三方许可证分别适用。
非当前双臂部署必需的 Manus 二进制与 PICO 安装包未提交，清单见 `provenance/excluded_vendor_binaries.json`。
