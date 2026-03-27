from __future__ import annotations

from dataclasses import dataclass
import math
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
        selection_metric: str = "fdr",
        selection_strategy: str = "topk",
        ensemble_weighting: str = "metric",
        score_normalization: str = "none",
        routing_mode: str = "none",
        adaptive_margin_threshold: float = -1.0,
        adaptive_margin_quantile: float = 0.5,
        greedy_pool_size: int = 24,
        cv_folds: int = 5,
        cv_seed: int = 42,
        centroid_shrinkage: str = "none",
        shrinkage_alpha: float = 1.0,
        feature_reduction: str = "none",
        pca_dim: int = 128,
        fallback_margin_threshold: float = -1.0,
        fallback_margin_quantile: float = -1.0,
        fallback_margin_source: str = "best_component",
        fallback_max_new_tokens: int = 16,
        fallback_do_sample: bool = False,
        fallback_temperature: float = 0.0,
        progress_bar: bool = True,
    ):
        super().__init__(model=model, dataset_name=dataset_name, label_space=label_space)
        self.top_k = int(top_k)
        self.representation_levels = self._normalize_levels(representation_levels)
        self.normalize_features = bool(normalize_features)
        self.selection_metric = str(selection_metric).strip().lower()
        self.selection_strategy = str(selection_strategy).strip().lower()
        self.ensemble_weighting = str(ensemble_weighting).strip().lower()
        self.score_normalization = str(score_normalization).strip().lower()
        self.routing_mode = str(routing_mode).strip().lower()
        self.adaptive_margin_threshold = float(adaptive_margin_threshold)
        self.adaptive_margin_quantile = float(adaptive_margin_quantile)
        self.greedy_pool_size = int(greedy_pool_size)
        self.cv_folds = int(cv_folds)
        self.cv_seed = int(cv_seed)
        self.centroid_shrinkage = str(centroid_shrinkage).strip().lower()
        self.shrinkage_alpha = float(shrinkage_alpha)
        self.feature_reduction = str(feature_reduction).strip().lower()
        self.pca_dim = int(pca_dim)
        self.fallback_margin_threshold = float(fallback_margin_threshold)
        self.fallback_margin_quantile = float(fallback_margin_quantile)
        self.fallback_margin_source = str(fallback_margin_source).strip().lower()
        self.progress_bar = bool(progress_bar)

        if self.top_k <= 0:
            raise ValueError("RSE top_k must be positive")
        if self.selection_metric not in {"fdr", "loo_accuracy", "cv_accuracy"}:
            raise ValueError("RSE selection_metric must be 'fdr', 'loo_accuracy', or 'cv_accuracy'")
        if self.selection_strategy not in {"topk", "greedy_forward"}:
            raise ValueError("RSE selection_strategy must be 'topk' or 'greedy_forward'")
        if self.ensemble_weighting not in {"metric", "uniform"}:
            raise ValueError("RSE ensemble_weighting must be 'metric' or 'uniform'")
        if self.score_normalization not in {"none", "zscore"}:
            raise ValueError("RSE score_normalization must be 'none' or 'zscore'")
        if self.routing_mode not in {"none", "top1", "top2", "adaptive"}:
            raise ValueError("RSE routing_mode must be one of: none, top1, top2, adaptive")
        if self.adaptive_margin_quantile > 1.0:
            raise ValueError("RSE adaptive_margin_quantile must be <= 1.0")
        if self.greedy_pool_size <= 0:
            raise ValueError("RSE greedy_pool_size must be positive")
        if self.cv_folds <= 1:
            raise ValueError("RSE cv_folds must be > 1")
        if self.centroid_shrinkage not in {"none", "auto", "fixed"}:
            raise ValueError("RSE centroid_shrinkage must be one of: none, auto, fixed")
        if not (0.0 <= self.shrinkage_alpha <= 1.0):
            raise ValueError("RSE shrinkage_alpha must be in [0, 1]")
        if self.feature_reduction not in {"none", "pca"}:
            raise ValueError("RSE feature_reduction must be 'none' or 'pca'")
        if self.pca_dim <= 0:
            raise ValueError("RSE pca_dim must be positive")
        if self.fallback_margin_source not in {"best_component", "ensemble"}:
            raise ValueError("RSE fallback_margin_source must be 'best_component' or 'ensemble'")
        if self.fallback_margin_quantile > 1.0:
            raise ValueError("RSE fallback_margin_quantile must be <= 1.0")

        self.layer_specs = self._infer_layer_specs()
        self.num_layers = len(self.layer_specs)
        self.n_heads = self._infer_num_heads()
        self.route_k = {"none": 0, "top1": 1, "top2": 2, "adaptive": 0}[self.routing_mode]

        self.class_labels: List[str] = []
        self.label_to_index: Dict[str, int] = {}
        self.centroids: Dict[str, torch.Tensor] = {}
        self.fdr_scores: Dict[str, torch.Tensor] = {}
        self.loo_accuracy_scores: Dict[str, torch.Tensor] = {}
        self.cv_accuracy_scores: Dict[str, torch.Tensor] = {}
        self.train_component_loo_scores: Dict[Tuple[str, int], torch.Tensor] = {}
        self.train_component_cv_scores: Dict[Tuple[str, int], torch.Tensor] = {}
        self.metric_scores: Dict[Tuple[str, int], float] = {}
        self.feature_transforms: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}
        self.selected_components: List[Tuple[str, int]] = []
        self.selected_component_weights: Dict[Tuple[str, int], float] = {}
        self.selected_component_margin_thresholds: Dict[Tuple[str, int], float] = {}
        self.selection_train_accuracy: float | None = None
        self.resolved_fallback_threshold: float | None = None
        self.effective_cv_folds: int | None = None
        self._train_label_indices: torch.Tensor | None = None
        self._component_eval_stats: Dict[Tuple[str, int], Dict[str, float]] = {}
        self._final_correct = 0
        self._final_total = 0
        self._fallback_used = 0
        self._margins: List[float] = []
        self._oracle_component_hits: List[Dict[str, object]] = []
        self._fitted = False

        self.zero_shot_fallback = None
        if self.fallback_margin_threshold >= 0.0 or self.fallback_margin_quantile >= 0.0:
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
        attn_names = self._filter_text_backbone_module_names(attn_names)

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

    @staticmethod
    def _ensure_list(handles):
        if isinstance(handles, list):
            return handles
        return [handles]

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

    def _resolve_shrinkage_alpha(self, class_count: int, dim: int) -> float:
        if self.centroid_shrinkage == "none":
            return 1.0
        if self.centroid_shrinkage == "fixed":
            return self.shrinkage_alpha
        return float(class_count) / float(class_count + dim)

    def _compute_class_centroids(
        self,
        values: torch.Tensor,
        label_indices: torch.Tensor,
        num_classes: int,
        allow_missing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        overall_mean = values.mean(dim=0)
        centroids = []
        present_mask = []
        for class_idx in range(num_classes):
            mask = label_indices == class_idx
            present = bool(mask.any())
            present_mask.append(present)
            if not present:
                if not allow_missing:
                    raise RuntimeError(f"RSE missing support examples for class index {class_idx}")
                centroids.append(overall_mean.clone())
                continue

            class_values = values[mask]
            class_mean = class_values.mean(dim=0)
            alpha = self._resolve_shrinkage_alpha(int(class_values.shape[0]), int(values.shape[-1]))
            centroid = alpha * class_mean + (1.0 - alpha) * overall_mean
            centroids.append(centroid)

        return torch.stack(centroids, dim=0), torch.tensor(present_mask, dtype=torch.bool)

    def _score_queries_against_centroids(
        self,
        query_features: torch.Tensor,
        centroids: torch.Tensor,
        present_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_features.ndim == 1:
            query_features = query_features.unsqueeze(0)

        query = query_features.to(dtype=torch.float32)
        class_centroids = centroids.to(dtype=torch.float32)
        if self.normalize_features:
            query = F.normalize(query, dim=-1)
            class_centroids = F.normalize(class_centroids, dim=-1)

        scores = query @ class_centroids.transpose(0, 1)
        if present_mask is not None and not bool(present_mask.all()):
            scores[:, ~present_mask] = -1e9
        return scores

    def _score_queries_from_train(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        query_features: torch.Tensor,
        num_classes: int,
        allow_missing: bool = False,
    ) -> torch.Tensor:
        centroids, present_mask = self._compute_class_centroids(
            train_features,
            train_labels,
            num_classes,
            allow_missing=allow_missing,
        )
        return self._score_queries_against_centroids(query_features, centroids, present_mask=present_mask)

    def _fit_feature_transforms(
        self,
        feature_table: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self.feature_transforms = {}
        if self.feature_reduction == "none":
            return feature_table

        transformed_table: Dict[str, torch.Tensor] = {}
        for level, table in feature_table.items():
            level_rows = []
            for layer_idx in range(table.shape[1]):
                features = table[:, layer_idx, :].to(dtype=torch.float32)
                target_dim = min(self.pca_dim, int(features.shape[0]) - 1, int(features.shape[-1]))
                if target_dim < 1:
                    level_rows.append(features)
                    continue

                mean = features.mean(dim=0)
                centered = features - mean.unsqueeze(0)
                try:
                    _, _, basis = torch.pca_lowrank(centered, q=target_dim, center=False)
                    basis = basis[:, :target_dim]
                except RuntimeError:
                    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                    basis = vh.transpose(0, 1)[:, :target_dim]

                self.feature_transforms[(level, layer_idx)] = {
                    "mean": mean.cpu(),
                    "basis": basis.cpu(),
                }
                level_rows.append((centered @ basis).cpu())
            transformed_table[level] = torch.stack(level_rows, dim=1)
        return transformed_table

    def _transform_component_values(
        self,
        level: str,
        layer_idx: int,
        values: torch.Tensor,
    ) -> torch.Tensor:
        transform = self.feature_transforms.get((level, layer_idx))
        if transform is None:
            return values.to(dtype=torch.float32)

        mean = transform["mean"].to(dtype=torch.float32)
        basis = transform["basis"].to(dtype=torch.float32)
        if values.ndim == 1:
            return (values.to(dtype=torch.float32) - mean) @ basis
        return (values.to(dtype=torch.float32) - mean.unsqueeze(0)) @ basis

    def _build_stratified_folds(self, label_indices: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(label_indices, minlength=len(self.class_labels))
        min_count = int(counts.min().item()) if len(counts) > 0 else 0
        if min_count < 2:
            self.effective_cv_folds = None
            raise ValueError("RSE CV scoring requires at least 2 support examples per class")

        effective_folds = min(self.cv_folds, min_count)
        assignments = torch.empty_like(label_indices)
        generator = torch.Generator()
        generator.manual_seed(self.cv_seed)

        for class_idx in range(len(self.class_labels)):
            sample_indices = (label_indices == class_idx).nonzero(as_tuple=False).squeeze(-1)
            shuffled = sample_indices[torch.randperm(sample_indices.shape[0], generator=generator)]
            for order, sample_idx in enumerate(shuffled.tolist()):
                assignments[sample_idx] = order % effective_folds

        self.effective_cv_folds = effective_folds
        return assignments

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
                layer_values = table[:, layer_idx, :]
                class_centroids, _ = self._compute_class_centroids(
                    layer_values,
                    label_indices,
                    num_classes,
                    allow_missing=False,
                )
                level_centroids.append(class_centroids)
            centroids[level] = torch.stack(level_centroids, dim=0)
        return centroids

    def _compute_loo_stats(
        self,
        feature_table: Dict[str, torch.Tensor],
        label_indices: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], Dict[Tuple[str, int], torch.Tensor]]:
        num_classes = len(self.class_labels)
        loo_accuracy_scores: Dict[str, torch.Tensor] = {}
        component_scores: Dict[Tuple[str, int], torch.Tensor] = {}
        num_samples = int(label_indices.shape[0])
        sample_indices = torch.arange(num_samples, dtype=torch.long)
        label_indices_long = label_indices.to(dtype=torch.long)

        for level, table in feature_table.items():
            level_scores = []
            for layer_idx in range(table.shape[1]):
                features = table[:, layer_idx, :].to(dtype=torch.float32)
                scores = torch.empty((num_samples, num_classes), dtype=torch.float32)
                for sample_idx in range(num_samples):
                    keep_mask = sample_indices != sample_idx
                    scores[sample_idx] = self._score_queries_from_train(
                        features[keep_mask],
                        label_indices_long[keep_mask],
                        features[sample_idx],
                        num_classes,
                        allow_missing=True,
                    )[0]
                accuracy = float((scores.argmax(dim=-1) == label_indices_long).to(dtype=torch.float32).mean().item())
                level_scores.append(accuracy)
                component_scores[(level, layer_idx)] = scores.cpu()

            loo_accuracy_scores[level] = torch.tensor(level_scores, dtype=torch.float32)

        return loo_accuracy_scores, component_scores

    def _compute_cv_stats(
        self,
        feature_table: Dict[str, torch.Tensor],
        label_indices: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], Dict[Tuple[str, int], torch.Tensor]]:
        num_classes = len(self.class_labels)
        label_indices_long = label_indices.to(dtype=torch.long)
        fold_assignments = self._build_stratified_folds(label_indices_long)
        cv_accuracy_scores: Dict[str, torch.Tensor] = {}
        component_scores: Dict[Tuple[str, int], torch.Tensor] = {}

        for level, table in feature_table.items():
            level_scores = []
            for layer_idx in range(table.shape[1]):
                features = table[:, layer_idx, :].to(dtype=torch.float32)
                scores = torch.empty((features.shape[0], num_classes), dtype=torch.float32)
                for fold_idx in range(int(self.effective_cv_folds or 0)):
                    train_mask = fold_assignments != fold_idx
                    val_mask = fold_assignments == fold_idx
                    scores[val_mask] = self._score_queries_from_train(
                        features[train_mask],
                        label_indices_long[train_mask],
                        features[val_mask],
                        num_classes,
                        allow_missing=False,
                    )
                accuracy = float((scores.argmax(dim=-1) == label_indices_long).to(dtype=torch.float32).mean().item())
                level_scores.append(accuracy)
                component_scores[(level, layer_idx)] = scores.cpu()

            cv_accuracy_scores[level] = torch.tensor(level_scores, dtype=torch.float32)

        return cv_accuracy_scores, component_scores

    def _all_components_with_metric(self) -> List[Tuple[Tuple[str, int], float]]:
        all_components: List[Tuple[Tuple[str, int], float]] = []
        for level in self.representation_levels:
            if self.selection_metric == "loo_accuracy":
                score_tensor = self.loo_accuracy_scores[level]
            elif self.selection_metric == "cv_accuracy":
                score_tensor = self.cv_accuracy_scores[level]
            else:
                score_tensor = self.fdr_scores[level]
            for layer_idx, score in enumerate(score_tensor.tolist()):
                component = (level, layer_idx)
                metric_value = float(score)
                self.metric_scores[component] = metric_value
                all_components.append((component, metric_value))
        all_components.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
        return all_components

    def _active_train_component_scores(self) -> Dict[Tuple[str, int], torch.Tensor]:
        if self.selection_metric == "cv_accuracy" and self.train_component_cv_scores:
            return self.train_component_cv_scores
        return self.train_component_loo_scores

    def _compute_component_weights(
        self,
        components: Sequence[Tuple[str, int]],
    ) -> Dict[Tuple[str, int], float]:
        if not components:
            return {}

        if self.ensemble_weighting == "uniform":
            weights = torch.ones(len(components), dtype=torch.float32)
        else:
            weights = torch.tensor(
                [max(self.metric_scores.get(component, 0.0), 0.0) for component in components],
                dtype=torch.float32,
            )
            if float(weights.sum().item()) <= 0.0:
                weights = torch.ones_like(weights)

        weights = weights / weights.sum().clamp_min(1e-6)
        return {
            component: float(weight.item())
            for component, weight in zip(components, weights)
        }

    def _normalize_score_tensor(self, scores: torch.Tensor) -> torch.Tensor:
        if self.score_normalization == "none":
            return scores

        if scores.ndim == 1:
            mean = scores.mean()
            std = scores.std(unbiased=False).clamp_min(1e-6)
            return (scores - mean) / std

        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (scores - mean) / std

    @staticmethod
    def _compute_margin(scores: torch.Tensor) -> torch.Tensor:
        if scores.shape[-1] <= 1:
            return scores[..., 0]
        top_values = torch.topk(scores, k=2, dim=-1).values
        return top_values[..., 0] - top_values[..., 1]

    def _combine_scores_batch(
        self,
        component_scores: Dict[Tuple[str, int], torch.Tensor],
        components: Sequence[Tuple[str, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not components:
            raise ValueError("RSE requires at least one component to combine")

        stacked_scores = torch.stack(
            [self._normalize_score_tensor(component_scores[component]) for component in components],
            dim=1,
        )
        component_margins = self._compute_margin(stacked_scores)

        base_weights = torch.tensor(
            [self.selected_component_weights.get(component, 0.0) for component in components],
            dtype=stacked_scores.dtype,
            device=stacked_scores.device,
        )
        if float(base_weights.sum().item()) <= 0.0:
            base_weights = torch.ones_like(base_weights)

        if self.routing_mode == "adaptive":
            thresholds = torch.tensor(
                [self.selected_component_margin_thresholds.get(component, -1e9) for component in components],
                dtype=stacked_scores.dtype,
                device=stacked_scores.device,
            )
            used_weights = (component_margins >= thresholds.unsqueeze(0)).to(dtype=stacked_scores.dtype)
            used_weights = used_weights * base_weights.unsqueeze(0)
            empty_rows = used_weights.sum(dim=1) <= 0
            if bool(empty_rows.any()):
                used_weights[empty_rows] = base_weights
        elif self.route_k <= 0 or len(components) <= self.route_k:
            used_weights = base_weights.unsqueeze(0).expand(stacked_scores.shape[0], -1)
        else:
            top_indices = torch.topk(component_margins, k=self.route_k, dim=1).indices
            used_weights = torch.zeros_like(component_margins)
            used_weights.scatter_(1, top_indices, 1.0)
            used_weights = used_weights * base_weights.unsqueeze(0)

        used_weights = used_weights / used_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        final_scores = (stacked_scores * used_weights.unsqueeze(-1)).sum(dim=1)
        final_margin = self._compute_margin(final_scores)
        best_component_margin = component_margins.max(dim=1).values
        return final_scores, final_margin, best_component_margin

    def _combine_scores_single(
        self,
        component_scores: Dict[Tuple[str, int], torch.Tensor],
        components: Sequence[Tuple[str, int]],
    ) -> tuple[torch.Tensor, float, float]:
        batch_scores = {
            component: scores.unsqueeze(0)
            for component, scores in component_scores.items()
        }
        final_scores, final_margin, best_component_margin = self._combine_scores_batch(batch_scores, components)
        return final_scores[0], float(final_margin[0].item()), float(best_component_margin[0].item())

    def _subset_train_accuracy(self, components: Sequence[Tuple[str, int]]) -> float:
        if self._train_label_indices is None:
            return 0.0
        final_scores, _, _ = self._combine_scores_batch(self._active_train_component_scores(), components)
        preds = final_scores.argmax(dim=-1)
        return float((preds == self._train_label_indices).to(dtype=torch.float32).mean().item())

    def _select_components(self) -> None:
        ranked_components = self._all_components_with_metric()
        if not ranked_components:
            raise RuntimeError("RSE could not rank any components")

        if self.selection_strategy == "topk":
            self.selected_components = [
                component for component, _ in ranked_components[: min(self.top_k, len(ranked_components))]
            ]
        else:
            pool = [component for component, _ in ranked_components[: min(self.greedy_pool_size, len(ranked_components))]]
            self.selected_components = [pool[0]]
            current_accuracy = self._subset_train_accuracy(self.selected_components)

            while len(self.selected_components) < self.top_k:
                best_candidate = None
                best_accuracy = current_accuracy
                for component in pool:
                    if component in self.selected_components:
                        continue
                    trial_components = self.selected_components + [component]
                    trial_accuracy = self._subset_train_accuracy(trial_components)
                    if trial_accuracy > best_accuracy + 1e-9:
                        best_accuracy = trial_accuracy
                        best_candidate = component

                if best_candidate is None:
                    break
                self.selected_components.append(best_candidate)
                current_accuracy = best_accuracy

        self.selected_component_weights = self._compute_component_weights(self.selected_components)
        self.selection_train_accuracy = self._subset_train_accuracy(self.selected_components)

    def _resolve_component_margin_thresholds(self) -> None:
        self.selected_component_margin_thresholds = {}
        if self.routing_mode != "adaptive":
            return

        if self.adaptive_margin_threshold >= 0.0:
            self.selected_component_margin_thresholds = {
                component: self.adaptive_margin_threshold
                for component in self.selected_components
            }
            return

        quantile = min(max(self.adaptive_margin_quantile, 0.0), 1.0)
        score_tables = self._active_train_component_scores()
        for component in self.selected_components:
            scores = score_tables.get(component)
            if scores is None:
                continue
            margins = self._compute_margin(self._normalize_score_tensor(scores))
            self.selected_component_margin_thresholds[component] = float(torch.quantile(margins, q=quantile).item())

    def _resolve_fallback_threshold(self) -> None:
        self.resolved_fallback_threshold = None
        if self.zero_shot_fallback is None:
            return

        if self.fallback_margin_threshold >= 0.0:
            self.resolved_fallback_threshold = self.fallback_margin_threshold
            return

        if self.fallback_margin_quantile < 0.0 or self._train_label_indices is None:
            return

        _, ensemble_margins, best_component_margins = self._combine_scores_batch(
            self._active_train_component_scores(),
            self.selected_components,
        )
        source_margins = best_component_margins if self.fallback_margin_source == "best_component" else ensemble_margins
        quantile = min(max(self.fallback_margin_quantile, 0.0), 1.0)
        self.resolved_fallback_threshold = float(torch.quantile(source_margins, q=quantile).item())

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
        self._oracle_component_hits = []

    def fit(self, train_data: Sequence[Dict]) -> None:
        if not train_data:
            raise ValueError("RSE fit requires non-empty train_data")

        train_labels = [str(item["label"]) for item in train_data]
        unique_train_labels = {label for label in train_labels}
        self.class_labels = [label for label in self.label_space if label in unique_train_labels]
        self.label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
        label_indices = torch.tensor([self.label_to_index[label] for label in train_labels], dtype=torch.long)
        self._train_label_indices = label_indices

        feature_table = self._collect_feature_table(train_data, desc="RSE fit features")
        feature_table = self._fit_feature_transforms(feature_table)
        self.centroids = self._compute_centroids(feature_table, label_indices)
        self.fdr_scores = self._compute_fdr_scores(feature_table, label_indices)
        self.loo_accuracy_scores, self.train_component_loo_scores = self._compute_loo_stats(feature_table, label_indices)
        if self.selection_metric == "cv_accuracy":
            self.cv_accuracy_scores, self.train_component_cv_scores = self._compute_cv_stats(feature_table, label_indices)
        else:
            self.cv_accuracy_scores = {}
            self.train_component_cv_scores = {}
            self.effective_cv_folds = None
        self.metric_scores = {}
        self._select_components()
        self._resolve_component_margin_thresholds()
        self._resolve_fallback_threshold()
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
                query = self._transform_component_values(level, layer_idx, level_features[layer_idx])
                centroids = level_centroids[layer_idx]
                scores = self._score_queries_against_centroids(query, centroids)[0]
                all_scores[(level, layer_idx)] = scores.to(dtype=torch.float32)
        return all_scores

    def _record_eval(
        self,
        true_label: str,
        final_label: str,
        margin: float,
        all_scores: Dict[Tuple[str, int], torch.Tensor],
        fallback_used: bool,
        sample_id: str | None = None,
    ) -> None:
        if true_label not in self.label_to_index:
            return

        self._final_total += 1
        self._final_correct += int(final_label == true_label)
        self._fallback_used += int(fallback_used)
        self._margins.append(float(margin))

        true_idx = self.label_to_index[true_label]
        correct_components: List[str] = []
        for key, scores in all_scores.items():
            pred_idx = int(scores.argmax().item())
            is_correct = pred_idx == true_idx
            self._component_eval_stats[key]["correct"] += float(is_correct)
            self._component_eval_stats[key]["count"] += 1.0
            if is_correct:
                correct_components.append(f"{key[0]}@{key[1]}")

        self._oracle_component_hits.append(
            {
                "sample_id": sample_id,
                "true_label": true_label,
                "final_label": final_label,
                "final_correct": final_label == true_label,
                "oracle_correct": bool(correct_components),
                "correct_components": correct_components,
            }
        )

    def predict(self, sample: Dict) -> str:
        if not self._fitted:
            raise RuntimeError("RSE method is not fitted. Call fit(train_data) first.")

        sample_features = self._extract_multilevel_features(sample)
        all_scores = self._component_scores(sample_features)
        final_scores, ensemble_margin, best_component_margin = self._combine_scores_single(
            all_scores,
            self.selected_components,
        )

        pred_label = self.class_labels[int(final_scores.argmax().item())]
        chosen_margin = best_component_margin if self.fallback_margin_source == "best_component" else ensemble_margin
        fallback_used = False

        if (
            self.zero_shot_fallback is not None
            and self.resolved_fallback_threshold is not None
            and chosen_margin < self.resolved_fallback_threshold
        ):
            pred_label = self.zero_shot_fallback.predict(sample)
            fallback_used = True

        self._record_eval(
            str(sample.get("label", "")),
            pred_label,
            chosen_margin,
            all_scores,
            fallback_used,
            sample_id=str(sample.get("question_id", "")) or None,
        )
        return pred_label

    def export_diagnostics(self) -> Dict:
        if not self._fitted:
            return {}

        component_table = []
        selected_set = set(self.selected_components)
        for level in self.representation_levels:
            fdr_level = self.fdr_scores[level]
            loo_level = self.loo_accuracy_scores[level]
            cv_level = self.cv_accuracy_scores.get(level)
            for layer_idx in range(len(fdr_level)):
                key = (level, layer_idx)
                stats = self._component_eval_stats.get(key, {"correct": 0.0, "count": 0.0})
                count = float(stats["count"])
                component_table.append(
                    {
                        "level": level,
                        "layer_idx": layer_idx,
                        "selection_score": float(self.metric_scores.get(key, 0.0)),
                        "fdr": float(fdr_level[layer_idx].item()),
                        "loo_accuracy": float(loo_level[layer_idx].item()),
                        "cv_accuracy": None if cv_level is None else float(cv_level[layer_idx].item()),
                        "selected": key in selected_set,
                        "weight": float(self.selected_component_weights.get(key, 0.0)),
                        "adaptive_margin_threshold": self.selected_component_margin_thresholds.get(key),
                        "val_accuracy": (float(stats["correct"]) / count) if count > 0 else None,
                    }
                )

        component_table.sort(
            key=lambda row: (-row["selection_score"], -(row["loo_accuracy"] or 0.0), row["level"], row["layer_idx"])
        )
        selected_components = [
            {
                "level": level,
                "layer_idx": layer_idx,
                "selection_score": float(self.metric_scores[(level, layer_idx)]),
                "weight": float(self.selected_component_weights[(level, layer_idx)]),
                "fdr": float(self.fdr_scores[level][layer_idx].item()),
                "loo_accuracy": float(self.loo_accuracy_scores[level][layer_idx].item()),
                "cv_accuracy": (
                    None
                    if level not in self.cv_accuracy_scores
                    else float(self.cv_accuracy_scores[level][layer_idx].item())
                ),
                "adaptive_margin_threshold": self.selected_component_margin_thresholds.get((level, layer_idx)),
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

        best_by_loo = max(component_table, key=lambda row: row["loo_accuracy"])
        best_by_cv = (
            max(
                [row for row in component_table if row["cv_accuracy"] is not None],
                key=lambda row: row["cv_accuracy"],
            )
            if self.cv_accuracy_scores
            else None
        )
        best_by_val = max(
            [row for row in component_table if row["val_accuracy"] is not None],
            key=lambda row: row["val_accuracy"],
            default=None,
        )

        margin_mean = (sum(self._margins) / len(self._margins)) if self._margins else None
        margin_min = min(self._margins) if self._margins else None
        margin_max = max(self._margins) if self._margins else None
        oracle_correct = sum(1 for row in self._oracle_component_hits if bool(row["oracle_correct"]))

        return {
            "method": "rse",
            "selection_metric": self.selection_metric,
            "selection_strategy": self.selection_strategy,
            "ensemble_weighting": self.ensemble_weighting,
            "score_normalization": self.score_normalization,
            "routing_mode": self.routing_mode,
            "adaptive_margin_threshold": self.adaptive_margin_threshold,
            "adaptive_margin_quantile": self.adaptive_margin_quantile,
            "representation_levels": list(self.representation_levels),
            "num_layers": self.num_layers,
            "top_k": self.top_k,
            "greedy_pool_size": self.greedy_pool_size,
            "cv_folds": self.cv_folds,
            "effective_cv_folds": self.effective_cv_folds,
            "centroid_shrinkage": self.centroid_shrinkage,
            "shrinkage_alpha": self.shrinkage_alpha,
            "feature_reduction": self.feature_reduction,
            "pca_dim": self.pca_dim,
            "fallback_margin_threshold": self.resolved_fallback_threshold,
            "fallback_margin_quantile": self.fallback_margin_quantile,
            "fallback_margin_source": self.fallback_margin_source,
            "class_labels": list(self.class_labels),
            "selected_components": selected_components,
            "best_component_by_loo": best_by_loo,
            "best_component_by_cv": best_by_cv,
            "best_component_by_val": best_by_val,
            "component_table": component_table,
            "train_selection_summary": {
                "selection_train_accuracy": self.selection_train_accuracy,
            },
            "oracle_summary": {
                "oracle_accuracy": (oracle_correct / self._final_total) if self._final_total > 0 else None,
                "oracle_correct": oracle_correct,
                "oracle_total": self._final_total,
            },
            "oracle_samples": self._oracle_component_hits,
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
