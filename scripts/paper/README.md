# Paper Experiment Scripts

这套脚本把论文预实验按 `claim -> suite` 重新组织，避免继续在阶段性 exploratory 脚本上叠加逻辑。

## 入口

- `run_f1_representation_gap.sh`
  证据: `F1` representation-generation gap
- `run_f2_task_peaks.sh`
  证据: `F2` task-dependent peaks
- `run_f3_read_vs_write.sh`
  证据: `F3` read vs write
- `run_c1_main_results.sh`
  主结果: `C1`
- `run_c2_ablations.sh`
  消融: `C2`
- `run_c3_efficiency.sh`
  效率: `C3`
- `run_c4_cross_model.sh`
  跨模型: `C4`
- `run_fgvc_baselines.sh`
  FGVC baseline 总表，含 `KeCO`

## 共享逻辑

- `registry.py`
  统一定义任务、方法、任务组、方法组
- `run_suite.py`
  通用 runner，负责:
  1. 构建 seed-aware subsets
  2. 跑 `main.py`
  3. 写出统一 manifest
  4. 调用统一 summary
- `summarize_suite.py`
  生成跨模型/跨 seed 的 markdown 总表
- `analyze_representation_gap.py`
  从 `RSEv2` diagnostics 提取 best component / oracle upper bound
- `export_component_tables.py`
  导出 `F2` heatmap 所需的 `(level, layer)` 长表

## 默认模型

- `qwen2_vl`
- `qwen3_vl`
- `idefics3`

所有 wrapper 都支持通过环境变量覆盖:

```bash
MODELS=qwen2_vl,qwen3_vl
SEEDS=42,43,44
TASK_GROUPS=all_tasks_current
METHOD_GROUPS=main_results
TIMESTAMP=20260325_120000
```

GPU 调度也可以通过环境变量控制:

```bash
PAPER_GPU_WORKERS=auto
```

`PAPER_GPU_WORKERS` 的格式是“分号分 worker，逗号分给单个 worker 的 GPU”:

- `auto`
  自动把当前可见 GPU 拆成单卡 worker。例如当前可见 GPU 是 `0,1,2,3`，就等价于 `0;1;2;3`
- `all`
  单个实验使用当前可见的全部 GPU
- `"0;1;2;3"`
  4 个并发 worker，每个实验独占 1 张卡
- `"0,1;2,3"`
  2 个并发 worker，每个实验可用 2 张卡，适合模型分片

如果已经先用 `CUDA_VISIBLE_DEVICES` 限制了可见卡，`auto` / `all` 会基于这个子集继续工作。

常见用法:

```bash
# 保持原行为：串行运行，继承当前 CUDA_VISIBLE_DEVICES
bash scripts/paper/run_c1_main_results.sh

# 自动把所有可见 GPU 都利用起来：每张卡并发跑一个实验
PAPER_GPU_WORKERS=auto bash scripts/paper/run_c1_main_results.sh

# 两组双卡 worker：每个实验最多吃两张卡
PAPER_GPU_WORKERS="0,1;2,3" bash scripts/paper/run_c1_main_results.sh

# 先限制到 2/3 号卡，再自动拆成两个单卡 worker
CUDA_VISIBLE_DEVICES=2,3 PAPER_GPU_WORKERS=auto bash scripts/paper/run_c1_main_results.sh

# 单个实验使用全部可见 GPU（显式写法）
CUDA_VISIBLE_DEVICES=0,1 PAPER_GPU_WORKERS=all bash scripts/paper/run_c1_main_results.sh
```

如果希望脚本结束后自动发飞书通知，额外设置:

```bash
export FEISHU_WEBHOOK_URL='https://open.feishu.cn/open-apis/bot/v2/hook/your-webhook'
bash scripts/paper/run_c1_main_results.sh
```

通知会在脚本成功或失败时发送，默认包含 suite 名称、状态、模型、seed、任务组、方法组，以及生成的 summary 路径。

## 输出位置

统一写到 `swap/paper/`:

- `swap/paper/subsets/<timestamp>_<suite>/`
- `swap/paper/logs/<timestamp>_<suite>/`
- `swap/paper/outputs/<timestamp>_<suite>/manifest.tsv`
- `swap/paper/records/<suite>_<timestamp>.md`

这套路径和旧的 exploratory `swap/outputs` / `swap/records` 分开，方便后续只保留 paper 结果。

并发模式下，`manifest.tsv` 会额外记录 `sequence_index`、`worker_id` 和 `cuda_visible_devices`，每个运行日志开头也会写入对应的 GPU 分配信息，方便回溯。
