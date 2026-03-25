from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..data import build_prompt
from .base import MethodBase
from .zero_shot import ZeroShotMethod


@dataclass(frozen=True)
class _RSELayerSpec:
    layer_idx: int
    block_name: str
    attn_name: str
    attn_proj_name: str
    mlp_name: str


class RSEMethod(MethodBase):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        top_k: int = 8,
        representation_levels: Sequence[str] = ("head", "layer", "attn", "mlp"),
        normalize_features: bool = True,
        fallback_margin_threshold: float = -1.0,
        fallback_max_new_tokens: int = 16,
        fallback_do_sample: bool = False,
        fallback_temperature: float = 0.0,
        progress_bar: bool = True,
    ):
        super().__init__(model=model, dataset_name=dataset_name, label_space=label_space)
        self.top_k = int(top_k)
        self.representation_levels = self._normalize_levels(representation_levels)
        self.normalize_features = bool(normalize_features)
        self.fallback_margin_threshold = float(fallback_margin_threshold)
        self.progress_bar = bool(progress_bar)

        if self.top_k <= 0:
            raise ValueError("RSE top_k must be positive")

        self.layer_specs = self._infer_layer_specs()
        self.num_layers = len(self.layer_specs)
        self.n_heads = self._infer_num_heads()

        self.class_labels: List[str] = []
        self.label_to_index: Dict[str, int] = {}
        self.centroids: Dict[str, torch.Tensor] = {}
        self.fdr_scores: Dict[str, torch.Tensor] = {}
        self.selected_components: List[Tuple[str, int]] = []
        self.selected_component_weights: Dict[Tuple[str, int], float] = {}
        self._component_eval_stats: Dict[Tuple[str, int], Dict[str, float]] = {}
        self._final_correct = 0
        self._final_total = 0
        self._fallback_used = 0
        self._margins: List[float] = []
        self._fitted = False

        self.zero_shot_fallback = None
        if self.fallback_margin_threshold >= 0.0:
            self.zero_shot_fallback = ZeroShotMethod(
                model=model,
                dataset_name=dataset_name,
                label_space=label_space,
                max_new_tokens=fallback_max_new_tokens,
                do_sample=fallback_do_sample,
                temperature=fallback_temperature,
            )

    @staticmethod
    def _normalize_levels(levels) -> List[str]:
        if isinstance(levels, str):
            levels = [levels]
        normalized = [str(level).strip().lower() for level in levels]
        supported = {"head", "layer", "attn", "mlp"}
        invalid = [level for level in normalized if level not in supported]
        if invalid:
            raise ValueError(
                f"Unsupported RSE representation levels {invalid}. Supported levels: {sorted(supported)}"
            )
        if not normalized:
            raise ValueError("RSE requires at least one representation level")
        return list(dict.fromkeys(normalized))

    def _iter_with_progress(self, data: Sequence[Dict], desc: str):
        if self.progress_bar:
            return tqdm(data, desc=desc)
        return data

    @staticmethod
    def _sort_key(module_name: str):
        nums = [int(v) for v in re.findall(r"\d+", module_name)]
        return tuple(nums) if nums else (10**9,)

    def _infer_num_heads(self) -> int:
        config = self.model.model.config
        text_config = getattr(config, "text_config", config)
        for attr_name in ["num_attention_heads", "n_heads", "num_heads"]:
            if hasattr(text_config, attr_name):
                return int(getattr(text_config, attr_name))
        raise RuntimeError("Unable to infer number of attention heads for RSE")

    def _infer_layer_specs(self) -> List[_RSELayerSpec]:
        named_modules = dict(self.model.model.named_modules())
        attn_names = []
        for name, module in named_modules.items():
            if not name.endswith("self_attn"):
                continue
            if hasattr(module, "o_proj") or hasattr(module, "out_proj"):
                attn_names.append(name)

        if not attn_names:
            raise RuntimeError("Could not find decoder self-attention modules for RSE")

        layer_specs: List[_RSELayerSpec] = []
        for layer_idx, attn_name in enumerate(sorted(attn_names, key=self._sort_key)):
            layer_prefix = attn_name.rsplit(".", 1)[0]
            if f"{attn_name}.o_proj" in named_modules:
                attn_proj_name = f"{attn_name}.o_proj"
            elif f"{attn_name}.out_proj" in named_modules:
                attn_proj_name = f"{attn_name}.out_proj"
            else:
                raise RuntimeError(f"Could not resolve attention output projection for '{attn_name}'")

            mlp_candidates = [
                name
                for name in named_modules
                if name.startswith(f"{layer_prefix}.") and name.endswith("mlp")
            ]
            if f"{layer_prefix}.mlp" in named_modules:
                mlp_name = f"{layer_prefix}.mlp"
            elif len(mlp_candidates) == 1:
                mlp_name = mlp_candidates[0]
            else:
                raise RuntimeError(f"Could not uniquely determine MLP module for layer '{layer_prefix}'")

            layer_specs.append(
                _RSELayerSpec(
                    layer_idx=layer_idx,
                    block_name=layer_prefix,
                    attn_name=attn_name,
                    attn_proj_name=attn_proj_name,
                    mlp_name=mlp_name,
                )
            )
        return layer_specs

    @staticmethod
    def _last_token(hidden_like) -> torch.Tensor:
        hidden = hidden_like[0] if isinstance(hidden_like, tuple) else hidden_like
        if not torch.is_tensor(hidden) or hidden.ndim < 3:
            raise RuntimeError(f"Expected hidden states with ndim >= 3, got {type(hidden_like).__name__}")
        return hidden[0, -1, :].detach().to(dtype=torch.float32).cpu()

    def _extract_multilevel_features(self, sample: Dict) -> Dict[str, torch.Tensor]:
        images = self._load_images(sample)
        prompt = build_prompt(self.dataset_name, sample)

        captured_head: Dict[str, torch.Tensor] = {}
        captured_attn: Dict[str, torch.Tensor] = {}
        captured_mlp: Dict[str, torch.Tensor] = {}
        captured_layer: Dict[str, torch.Tensor] = {}

        def head_pre_hook(module, args, module_name=None):
            if not args:
                return
            hidden = args[0]
            if torch.is_tensor(hidden):
                captured_head[module_name] = hidden.detach()

        def tensor_hook(module, args, output, module_name=None):
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                if module_name in attn_names:
                    captured_attn[module_name] = hidden.detach()
                elif module_name in mlp_names:
                    captured_mlp[module_name] = hidden.detach()
                elif module_name in block_names:
                    captured_layer[module_name] = hidden.detach()

        attn_proj_names = [spec.attn_proj_name for spec in self.layer_specs]
        attn_names = {spec.attn_name for spec in self.layer_specs}
        mlp_names = {spec.mlp_name for spec in self.layer_specs}
        block_names = {spec.block_name for spec in self.layer_specs}

        handles = []
        handles.extend(self._ensure_list(self.model.register_forward_pre_hook(attn_proj_names, head_pre_hook)))
        handles.extend(self._ensure_list(self.model.register_forward_hook(list(attn_names), tensor_hook)))
        handles.extend(self._ensure_list(self.model.register_forward_hook(list(mlp_names), tensor_hook)))
        handles.extend(self._ensure_list(self.model.register_forward_hook(list(block_names), tensor_hook)))

        try:
            with torch.no_grad():
                _ = self.model.forward(images, [prompt])
        finally:
            for handle in handles:
                handle.remove()

        head_features = []
        attn_features = []
        mlp_features = []
        layer_features = []
        for spec in self.layer_specs:
            if spec.attn_proj_name not in captured_head:
                raise RuntimeError(f"Missing head-level capture for '{spec.attn_proj_name}'")
            if spec.attn_name not in captured_attn:
                raise RuntimeError(f"Missing attention output capture for '{spec.attn_name}'")
            if spec.mlp_name not in captured_mlp:
                raise RuntimeError(f"Missing MLP output capture for '{spec.mlp_name}'")
            if spec.block_name not in captured_layer:
                raise RuntimeError(f"Missing block output capture for '{spec.block_name}'")

            head_last = self._last_token(captured_head[spec.attn_proj_name])
            head_dim = head_last.shape[-1] // self.n_heads
            if head_dim * self.n_heads != head_last.shape[-1]:
                raise RuntimeError(
                    f"Hidden dim {head_last.shape[-1]} is not divisible by n_heads={self.n_heads}"
                )
            head_features.append(head_last.view(self.n_heads, head_dim).reshape(-1))
            attn_features.append(self._last_token(captured_attn[spec.attn_name]))
            mlp_features.append(self._last_token(captured_mlp[spec.mlp_name]))
            layer_features.append(self._last_token(captured_layer[spec.block_name]))

        return {
            "head": torch.stack(head_features, dim=0),
            "attn": torch.stack(attn_features, dim=0),
            "mlp": torch.stack(mlp_features, dim=0),
            "layer": torch.stack(layer_features, dim=0),
        }

    @staticmethod
    def _ensure_list(handles):
        if isinstance(handles, list):
            return handles
        return [handles]

    def _collect_feature_table(self, data: Sequence[Dict], desc: str) -> Dict[str, torch.Tensor]:
        per_level_rows = {level: [] for level in self.representation_levels}
        for item in self._iter_with_progress(data, desc=desc):
            sample_features = self._extract_multilevel_features(item)
            for level in self.representation_levels:
                per_level_rows[level].append(sample_features[level])
        return {
            level: torch.stack(rows, dim=0)
            for level, rows in per_level_rows.items()
        }

    def _compute_fdr_vector(
        self,
        values: torch.Tensor,
        label_indices: torch.Tensor,
        num_classes: int,
    ) -> float:
        overall_mean = values.mean(dim=0)
        between = torch.tensor(0.0, dtype=torch.float32)
        within = torch.tensor(0.0, dtype=torch.float32)

        for class_idx in range(num_classes):
            mask = label_indices == class_idx
            if not bool(mask.any()):
                continue
            class_values = values[mask]
            class_mean = class_values.mean(dim=0)
            between = between + float(class_values.shape[0]) * (class_mean - overall_mean).pow(2).sum()
            within = within + (class_values - class_mean).pow(2).sum()

        return float((between / (within + 1e-6)).item())

    def _compute_fdr_scores(
        self,
        feature_table: Dict[str, torch.Tensor],
        label_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_classes = len(self.class_labels)
        fdr_scores: Dict[str, torch.Tensor] = {}
        for level, table in feature_table.items():
            scores = []
            for layer_idx in range(table.shape[1]):
                scores.append(self._compute_fdr_vector(table[:, layer_idx, :], label_indices, num_classes))
            fdr_scores[level] = torch.tensor(scores, dtype=torch.float32)
        return fdr_scores

    def _compute_centroids(
        self,
        feature_table: Dict[str, torch.Tensor],
        label_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        centroids: Dict[str, torch.Tensor] = {}
        num_classes = len(self.class_labels)
        for level, table in feature_table.items():
            level_centroids = []
            for layer_idx in range(table.shape[1]):
                class_centroids = []
                layer_values = table[:, layer_idx, :]
                for class_idx in range(num_classes):
                    mask = label_indices == class_idx
                    if not bool(mask.any()):
                        raise RuntimeError(f"RSE missing support examples for class index {class_idx}")
                    class_centroids.append(layer_values[mask].mean(dim=0))
                level_centroids.append(torch.stack(class_centroids, dim=0))
            centroids[level] = torch.stack(level_centroids, dim=0)
        return centroids

    def _select_components(self) -> None:
        all_components: List[Tuple[float, str, int]] = []
        for level, scores in self.fdr_scores.items():
            for layer_idx, score in enumerate(scores.tolist()):
                all_components.append((float(score), level, layer_idx))

        all_components.sort(key=lambda item: (-item[0], item[1], item[2]))
        chosen = all_components[: min(self.top_k, len(all_components))]
        self.selected_components = [(level, layer_idx) for _, level, layer_idx in chosen]

        raw_weights = torch.tensor([max(score, 0.0) for score, _, _ in chosen], dtype=torch.float32)
        if raw_weights.numel() == 0:
            self.selected_component_weights = {}
            return
        if float(raw_weights.sum().item()) <= 0.0:
            raw_weights = torch.ones_like(raw_weights)
        norm_weights = raw_weights / raw_weights.sum()
        self.selected_component_weights = {
            component: float(weight.item())
            for component, weight in zip(self.selected_components, norm_weights)
        }

    def _reset_eval_diagnostics(self) -> None:
        self._component_eval_stats = {
            (level, layer_idx): {"correct": 0.0, "count": 0.0}
            for level in self.representation_levels
            for layer_idx in range(self.num_layers)
        }
        self._final_correct = 0
        self._final_total = 0
        self._fallback_used = 0
        self._margins = []

    def fit(self, train_data: Sequence[Dict]) -> None:
        if not train_data:
            raise ValueError("RSE fit requires non-empty train_data")

        train_labels = [str(item["label"]) for item in train_data]
        unique_train_labels = {label for label in train_labels}
        self.class_labels = [label for label in self.label_space if label in unique_train_labels]
        self.label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
        label_indices = torch.tensor([self.label_to_index[label] for label in train_labels], dtype=torch.long)

        feature_table = self._collect_feature_table(train_data, desc="RSE fit features")
        self.fdr_scores = self._compute_fdr_scores(feature_table, label_indices)
        self.centroids = self._compute_centroids(feature_table, label_indices)
        self._select_components()
        self._reset_eval_diagnostics()
        self._fitted = True

    def _component_scores(
        self,
        sample_features: Dict[str, torch.Tensor],
    ) -> Dict[Tuple[str, int], torch.Tensor]:
        all_scores: Dict[Tuple[str, int], torch.Tensor] = {}
        for level in self.representation_levels:
            level_features = sample_features[level]
            level_centroids = self.centroids[level]
            for layer_idx in range(level_features.shape[0]):
                query = level_features[layer_idx]
                centroids = level_centroids[layer_idx]
                if self.normalize_features:
                    query = F.normalize(query.unsqueeze(0), dim=-1).squeeze(0)
                    centroids = F.normalize(centroids, dim=-1)
                scores = F.cosine_similarity(centroids, query.unsqueeze(0), dim=-1)
                all_scores[(level, layer_idx)] = scores.to(dtype=torch.float32)
        return all_scores

    def _record_eval(
        self,
        true_label: str,
        final_label: str,
        margin: float,
        all_scores: Dict[Tuple[str, int], torch.Tensor],
        fallback_used: bool,
    ) -> None:
        if true_label not in self.label_to_index:
            return

        self._final_total += 1
        self._final_correct += int(final_label == true_label)
        self._fallback_used += int(fallback_used)
        self._margins.append(float(margin))

        true_idx = self.label_to_index[true_label]
        for key, scores in all_scores.items():
            pred_idx = int(scores.argmax().item())
            self._component_eval_stats[key]["correct"] += float(pred_idx == true_idx)
            self._component_eval_stats[key]["count"] += 1.0

    def predict(self, sample: Dict) -> str:
        if not self._fitted:
            raise RuntimeError("RSE method is not fitted. Call fit(train_data) first.")

        sample_features = self._extract_multilevel_features(sample)
        all_scores = self._component_scores(sample_features)

        final_scores = torch.zeros(len(self.class_labels), dtype=torch.float32)
        for component in self.selected_components:
            weight = self.selected_component_weights[component]
            final_scores = final_scores + weight * all_scores[component]

        top_values, top_indices = torch.topk(final_scores, k=min(2, final_scores.numel()))
        pred_label = self.class_labels[int(top_indices[0].item())]
        margin = float(top_values[0].item() - top_values[1].item()) if top_values.numel() > 1 else float(top_values[0].item())
        fallback_used = False

        if self.zero_shot_fallback is not None and margin < self.fallback_margin_threshold:
            pred_label = self.zero_shot_fallback.predict(sample)
            fallback_used = True

        self._record_eval(str(sample.get("label", "")), pred_label, margin, all_scores, fallback_used)
        return pred_label

    def export_diagnostics(self) -> Dict:
        if not self._fitted:
            return {}

        component_table = []
        selected_set = set(self.selected_components)
        for level in self.representation_levels:
            scores = self.fdr_scores[level]
            for layer_idx in range(len(scores)):
                key = (level, layer_idx)
                stats = self._component_eval_stats.get(key, {"correct": 0.0, "count": 0.0})
                count = float(stats["count"])
                component_table.append(
                    {
                        "level": level,
                        "layer_idx": layer_idx,
                        "fdr": float(scores[layer_idx].item()),
                        "selected": key in selected_set,
                        "weight": float(self.selected_component_weights.get(key, 0.0)),
                        "val_accuracy": (float(stats["correct"]) / count) if count > 0 else None,
                    }
                )

        component_table.sort(key=lambda row: (-row["fdr"], row["level"], row["layer_idx"]))
        selected_components = [
            {
                "level": level,
                "layer_idx": layer_idx,
                "weight": float(self.selected_component_weights[(level, layer_idx)]),
                "fdr": float(self.fdr_scores[level][layer_idx].item()),
                "val_accuracy": next(
                    (
                        row["val_accuracy"]
                        for row in component_table
                        if row["level"] == level and row["layer_idx"] == layer_idx
                    ),
                    None,
                ),
            }
            for level, layer_idx in self.selected_components
        ]

        margin_mean = (sum(self._margins) / len(self._margins)) if self._margins else None
        margin_min = min(self._margins) if self._margins else None
        margin_max = max(self._margins) if self._margins else None

        return {
            "method": "rse",
            "selection_metric": "fdr",
            "representation_levels": list(self.representation_levels),
            "num_layers": self.num_layers,
            "class_labels": list(self.class_labels),
            "selected_components": selected_components,
            "component_table": component_table,
            "eval_summary": {
                "final_accuracy": (self._final_correct / self._final_total) if self._final_total > 0 else None,
                "final_correct": self._final_correct,
                "final_total": self._final_total,
                "fallback_used": self._fallback_used,
                "margin_mean": margin_mean,
                "margin_min": margin_min,
                "margin_max": margin_max,
            },
        }
