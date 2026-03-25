# Paper Experiment Playbook

## Claims To Scripts

| Claim | Purpose | Run Script | Extra Analysis |
| --- | --- | --- | --- |
| `F1` | Representation-generation gap | [run_f1_representation_gap.sh](/root/autodl-tmp/FGVC/scripts/paper/run_f1_representation_gap.sh) | [analyze_representation_gap.py](/root/autodl-tmp/FGVC/scripts/paper/analyze_representation_gap.py) |
| `F2` | Task-dependent component peaks | [run_f2_task_peaks.sh](/root/autodl-tmp/FGVC/scripts/paper/run_f2_task_peaks.sh) | [export_component_tables.py](/root/autodl-tmp/FGVC/scripts/paper/export_component_tables.py) |
| `F3` | Read vs write | [run_f3_read_vs_write.sh](/root/autodl-tmp/FGVC/scripts/paper/run_f3_read_vs_write.sh) | suite summary |
| `C1` | Main results | [run_c1_main_results.sh](/root/autodl-tmp/FGVC/scripts/paper/run_c1_main_results.sh) | suite summary |
| `C2` | Mahalanobis ablations | [run_c2_ablations.sh](/root/autodl-tmp/FGVC/scripts/paper/run_c2_ablations.sh) | suite summary |
| `C3` | Efficiency | [run_c3_efficiency.sh](/root/autodl-tmp/FGVC/scripts/paper/run_c3_efficiency.sh) | suite summary timing section |
| `C4` | Cross-model generalization | [run_c4_cross_model.sh](/root/autodl-tmp/FGVC/scripts/paper/run_c4_cross_model.sh) | suite summary |
| `Baseline` | FGVC baseline table incl. KeCO | [run_fgvc_baselines.sh](/root/autodl-tmp/FGVC/scripts/paper/run_fgvc_baselines.sh) | suite summary |

## Task Groups

| Group | Coverage |
| --- | --- |
| `all_tasks_current` | 当前仓库已有的全任务集合 |
| `fgvc_core_large` | `eurosat / pets / cub` 的较大 FGVC 子集 |
| `fgvc_extended` | `fgvc_core_large + flowers + tinyimage` |
| `cross_model_core8` | 跨模型核心 8 任务 |
| `ablation_core6` | 当前 RSE/RSEv2 关键 6 任务 |
| `efficiency_core6` | 效率评测默认 6 任务 |

## Method Groups

| Group | Methods |
| --- | --- |
| `gap_findings` | `Zero-shot, SAV, RSEv2` |
| `task_peaks` | `RSEv2` |
| `read_vs_write` | `Zero-shot, SAV, RSEv2, STV, I2CL, MimIC` |
| `main_results` | `Zero-shot, SAV, RSEv2` |
| `rsev2_ablation` | `RSE, RSE-ZScore, RSE-Adaptive, RSEv2, RSEv2-HeadOnly, RSEv2-EqualWeight, RSEv2-Top1/4/8/All` |
| `efficiency` | `Zero-shot, SAV, RSEv2, MimIC, I2CL, STV` |
| `fgvc_baselines` | `Zero-shot, KeCO, SAV, RSE, RSEv2, STV, I2CL, MimIC` |

## Notes

- 新 runner 默认支持三种模型: `qwen2_vl`, `qwen3_vl`, `idefics3`
- subsets 现在支持 `seed + shuffle`，可以真正做多 seed support-set 采样
- `runner.py` 现在会统一记录 `fit_time_sec / predict_time_sec / avg_predict_time_sec`
- `RSE` / `RSEv2` diagnostics 现在包含 `oracle_summary`，可以直接支撑 `F1-b`
