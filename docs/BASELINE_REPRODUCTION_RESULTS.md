# Baseline 复现与实验结果总结

## 范围

当前仓库中保留并可直接运行的 baseline 如下：

| 方法 | 实现文件 | 配置文件 | 说明 |
| --- | --- | --- | --- |
| `zero_shot` | `src/methods/zero_shot.py` | 无单独配置 | 作为 query-only 参照基线 |
| `sav` | `src/methods/sav.py` | `conf/method/sav.yaml` | 仓库原生方法 |
| `mimic` / `mimcl` | `src/methods/mimic.py` | `conf/method/mimic.yaml` / `conf/method/mimcl.yaml` | 参考 MimIC 思路独立重写 |
| `stv` | `src/methods/stv.py` | `conf/method/stv.yaml` | 参考 STV 思路独立重写 |
| `i2cl` | `src/methods/i2cl.py` | `conf/method/i2cl.yaml` | 参考 I2CL 思路独立重写 |

说明：

- `mimic`、`stv`、`i2cl` 都是为了当前框架做的本地 baseline 实现，没有直接 import 外部拉取仓库的代码。
- `m2iv` 相关内容已按当前需求移除，不再纳入本次总结和实验对比。

## 复现方式概述

### SAV

- 位置：`src/methods/sav.py`
- 角色：当前仓库的主方法/强基线。
- 特点：直接在现有框架中运行，无需额外训练脚本耦合。

### MimIC

- 位置：`src/methods/mimic.py`
- 复现思路：保留其“学习 attention shift，并在预测时做 query-only intervention”的核心思想。
- 适配方式：改写为当前框架统一接口 `fit(train_data) -> predict(sample)`，不依赖外部 MimIC 仓库。

### STV

- 位置：`src/methods/stv.py`
- 复现思路：保留“敏感 head 差分、cluster bank、cluster 选择、task vector 注入”的主流程。
- 适配方式：使用本地 hook 和 `torch` 实现，不依赖外部 STV 仓库。
- 调参后已作为当前默认配置固化到 `conf/method/stv.yaml`。

### I2CL

- 位置：`src/methods/i2cl.py`
- 复现思路：保留“从 support demos 提取 layer-wise latent，构造 context vector，并在 query-only 推理时回注”的主流程。
- 适配方式：改造成与当前多模态生成式框架兼容的版本，使用 answer-only LM loss 对注入强度做校准。
- 额外修正：修复了 `inject_pos=last` 下的 inplace autograd 问题，避免训练时报错。
- 调参后已作为当前默认配置固化到 `conf/method/i2cl.yaml`。

## Benchmark 设置

本轮方法对比统一使用同一套小规模 benchmark，便于快速验证 baseline 是否有效。

| 项目 | 设置 |
| --- | --- |
| 模型 | `qwen2_vl` |
| 数据集 | `eurosat_sample` |
| 训练子集 | `dataset/benchmarks/eurosat_train50.json` |
| 验证子集 | `dataset/benchmarks/eurosat_val50.json` |
| 规模 | train 50 / val 50，10 类，每类 5 个样本 |
| 指标 | `raw_accuracy` |
| 输出目录 | `outputs/eval/` |

统一运行形式如下：

```bash
.venv/bin/python main.py \
  model=qwen2_vl \
  method=<method_name> \
  dataset=eurosat_sample \
  dataset.train_path=dataset/benchmarks/eurosat_train50.json \
  dataset.val_path=dataset/benchmarks/eurosat_val50.json \
  evaluator=raw \
  run.progress_bar=false \
  method.params.progress_bar=false
```

## 实验结果

### 当前结果总表

| 方法 | 准确率 | 正确数 | 结果说明 |
| --- | --- | --- | --- |
| `sav` | `0.68` | `34/50` | 当前最强 baseline |
| `mimic` | `0.58` | `29/50` | 明显优于默认 `i2cl/stv` 初版，和调优后 `i2cl` 持平 |
| `i2cl` | `0.58` | `29/50` | 调参后显著提升，当前默认配置已更新 |
| `zero_shot` | `0.56` | `28/50` | 作为 query-only 参照 |
| `stv` | `0.56` | `28/50` | 调参后恢复正常，不再出现明显类别塌缩 |

对应结果文件：

- `sav`: `outputs/eval/bench_qwen2_eurosat50_sav.metrics.json`
- `mimic`: `outputs/eval/bench_qwen2_eurosat50_mimic.metrics.json`
- `zero_shot`: `outputs/eval/bench_qwen2_eurosat50_zero_shot.metrics.json`
- `i2cl`: `outputs/eval/qwen2_vl_eurosat_i2cl_raw.metrics.json`
- `stv`: `outputs/eval/qwen2_vl_eurosat_stv_raw.metrics.json`

### 调参前后对比

| 方法 | 初版默认结果 | 调参后结果 | 变化 |
| --- | --- | --- | --- |
| `i2cl` | `0.48` | `0.58` | `+0.10` |
| `stv` | `0.26` | `0.56` | `+0.30` |

说明：

- `i2cl` 的有效组合为：`20 shots + late layers + mlp-only + last-token inject + 12 epochs + lr 5e-3 + no noise`。
- `stv` 的有效组合为：`4 shots + 20 examples + topk 4 + 3 clusters + 15 selection epochs + lr 0.02`。
- `stv` 初版有明显类别塌缩，50 个验证样本中有 32 个被预测为 `Residential Buildings`；调参后这一问题显著缓解。

## 当前结论

1. `sav` 仍然是这组 baseline 中最稳、效果最好的方法，当前 benchmark 上达到 `0.68`。
2. `mimic` 和调优后的 `i2cl` 处于第二梯队，均为 `0.58`，已经高于 `zero_shot`。
3. `stv` 初版表现较差，但经过一轮更合理的超参调整后恢复到 `0.56`，至少达到可用 baseline 水平。
4. 这组结果来自一个小规模 EuroSAT 子集，适合做快速对比和方向验证，但还不能直接替代完整数据集上的正式结论。

## 当前默认配置建议

如果后续继续在当前仓库里做 baseline 对比，建议直接使用现有默认配置：

- `method=sav`
- `method=mimic`
- `method=i2cl`
- `method=stv`

其中：

- `i2cl` 和 `stv` 的默认 YAML 已经更新为本轮调优后的版本。
- 如果要做正式论文级对比，建议下一步扩展到更大验证集或完整 benchmark，并固定随机种子做多次重复实验。
