# 时空历史模块验收（2026-11-01）

本文为当日历史记录。2026-11-02 起使用纯操作版本，最新状态见
[MANIPULATION_VALIDATION.md](MANIPULATION_VALIDATION.md)。

当前实现：`spacetime_v2`。本地 CPU 契约验证通过；服务器完整 Foundation GPU 复验待完成。
ai2ykj、备用 zzx 跳板和同一目标直连在本次多次 SSH 尝试中均超时。
代码提交 `1521819`；Git SSH push 同样因网络超时失败，服务器尚未同步本次改动。
本机 Tailscale backend 为 Running，目标 peer 的在线标记为 true，但实际 Tailscale ping
连续两次无响应；未修改网络、SSH 或系统配置。临时测试脚本已删除。

## 变更范围

- 四帧滚动历史为 `t-60,t-40,t-20,t`，参考 Cortex 表 10 的 4 帧、stride 20。
  两秒来自本批数据 30 Hz 的换算，不是用户指定或 Cortex 已核实的固定时间跨度。
  初始目标参考独立保留。
- 目标池化前进行空间/因果时间注意力；输出当前网格后再通过初始框身份查询读取目标。
- 四个状态历史 token 直接进入双分支 context，缺失状态 token 连同其 mask 一起处理。
- 默认完整条件模块 4,771,072 参数，输出 10 个 4096 维 token。
- 原 observation-only 动作监督、手别门控、视频抖动降权和数据划分保持不变。
- 用户确认轨迹成功完成；本次没有新增完成分类器或逐帧阶段标签。
- 新旧历史模块 checkpoint 不兼容；新 delta 写入 conditioner_version 并明确拒绝旧 delta。

## 验证

使用已有本机 `gigaworld-policy` 环境，不安装依赖、不修改驱动。记录见
[spacetime_contract.json](validation/spacetime_contract.json)。检查包含：

- 修改未来帧不影响较早时刻的输出；修改有效过去帧会影响当前空间特征。
- 无效历史即便包含 NaN，也不会污染视觉历史结果；无历史时与单帧路径一致。
- 当前观测保持相同，仅改变过去观测时，当前目标 token 会改变。
- 初始框、技能、手别和历史状态输入均影响输出；关键参数梯度有限且非零。
- 缺失状态 token 输出为零并在后续 context 中屏蔽。
- 拒绝未来时间、乱序有效历史、逐帧跟踪框。
- 全尺寸条件模块以 BF16 执行前向、反向、一次优化器更新和严格状态字典重新加载。
- 真实数据仍为 23/7 个训练/验证 episode、567 个训练窗口；无未来观测泄漏。

这里的输入敏感性不代表“历史指导了正确动作”。目前只有右手取试管数据，尚不能评估
吸液/加液的阶段区分率。需要后续用相似当前画面、不同近期动作过程的配对样本，
比较错误重复执行率、漏执行率与闭环成功率；不能只看训练 loss 或条件变化。

## 服务器复验入口

在 ai2-root-4080 依次运行两行；第二行是一条完整命令，GPU 编号按实际空闲情况替换：

```bash
cd /root/ykj/lab-low
CUDA_VISIBLE_DEVICES=2 bash scripts/train_lab.sh training.max_steps=3 training.save_steps=3 validation_every=3 validation_windows=2 training.output_path=outputs/spacetime_acceptance
```

新版训练强制检查动作头、技能/手别/目标、四层空间注意力、时间注意力、状态编码及
视频/动作 context 的 13 项有限非零梯度，并执行验证和 checkpoint 精确读回。
完整 GPU 训练及新版 delta 热启动结果需单独记录，不能复用旧版验收。

## Cortex 采样来源核查

2026-11-01 核对 [附录 A.5.2、表 10](https://arxiv.org/html/2607.05377#A5.SS2)
及官方仓库提交 `d2a9795b15aff369314b260d2727d7a0cdbf0b46`：

- 论文明确给出 4 帧、stride 20、启用历史状态 token、每 4 层时间注意力。
- 论文称数秒窗口，但未明确该低层历史采样流的帧率。10 Hz 是动作生成频率，不能代替它。
- 官方 `assets/dump_bin_bigbin/meta/info.json` 的确写 fps=30；但同目录 README 明确说明
  它是标注流程演示数据，不用于训练/报告结果，不能外推整个实机 System-1 数据的帧率。
- 检查官方仓库完整文件树，当前公开内容没有低层 MEM 执行器的数据加载实现，
  因此无法进一步核实它是否重采样以及严格的物理时间跨度。当前实现对齐公开帧数/stride，
  不宣称已对齐未公开的时间采样细节。
