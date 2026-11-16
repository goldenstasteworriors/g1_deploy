# 在新分配服务器上重新训练四组模型

先确认服务器挂载同一份 `/share`，且有四张空闲的兼容 NVIDIA GPU。
每个任务要求至少 76000 MiB 空闲显存。系统驱动、系统库仍由服务器提供；
不需要旧机器的 `/root` 环境链接，也不需要重新下载模型或数据。

下面四条分别在四个独立终端或 tmux 窗口执行，每个代码块都是一行命令，无需手动换行。
不是一项四卡分布式训练，而是四个独立的单卡模型。

GPU 0：通用模型。

```bash
/share/personal/yekangjie/lab-low/envs/openwam/bin/python /share/personal/yekangjie/lab-low/code/scripts/train_share.py general --gpu 0
```

GPU 1：取试管。

```bash
/share/personal/yekangjie/lab-low/envs/openwam/bin/python /share/personal/yekangjie/lab-low/code/scripts/train_share.py take_tube --gpu 1
```

GPU 2：加液，无框。

```bash
/share/personal/yekangjie/lab-low/envs/openwam/bin/python /share/personal/yekangjie/lab-low/code/scripts/train_share.py dispense --gpu 2
```

GPU 3：取滴管并吸液。

```bash
/share/personal/yekangjie/lab-low/envs/openwam/bin/python /share/personal/yekangjie/lab-low/code/scripts/train_share.py take_dropper --gpu 3
```

每条命令末尾追加 `--check` 可执行同一配置的 CPU 路径、依赖和 Hydra 解析检查，
不会构建 GPU 模型或开始训练。检查会生成自己的运行目录及导入/配置日志。
这不能替代新服务器上的实际 CUDA 前向/反向兼容性验证。

## 数据与训练设置

- 数据为 `lab_primitives_train_20261110_boxed_v3`，清单已存入共享代码的 `data/boxed_v3_suite`。
- 通用模型包括数据中五种操作，排除放回试管与移动；三个单技能子集沿用同一训练/验证划分。
- 四组各从原始 Foundation 重新训练 30000 步；不读取旧并行队列状态，不是从已完成权重续训。
- 保持 spacetime_manip_v3、16D 操作输出、技能与左右手条件、初始框及历史；加液无框。
- 监督来自 observation.state；保留抖动损失降权与框扰动，不加入尚未实现的光照增强。
- 每 1000 步保存和验证；保留最新两份及最佳 delta，原 Foundation 必须保留用于加载。
- 使用 `/share` Python 的 `-m torch.distributed.run`，不执行含旧环境 shebang 的 torchrun 脚本。
- 四张卡使用 29700–29703 端口；同机器同卡有文件锁，启动时检查空闲显存。

## 存储和日志

代码、环境、数据、Foundation 均位于 `/share/personal/yekangjie/lab-low`。
每次启动创建独立目录，不覆盖上一轮：

```text
/share/personal/yekangjie/lab-low/code/outputs/share_runs/<experiment>/<日期时间_PID>/
```

目录包含 `launch.json`、`train.log`、`weights`、`cache` 和 `hydra`。
最终 30000 步保存并读回一致后才写 `completed.json`。
依赖缓存、编译缓存及临时目录均显式指向此运行目录；HF 离线，W&B 关闭。

脚本前台运行，应放在持久终端或平台作业中。它不提供训练结束后的 GPU 保活；
不要把前台进程、tmux 或 Supervisor 等同于平台空闲回收所需的 GPU 保活。
