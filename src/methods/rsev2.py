from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from .rse import RSEMethod


@dataclass
class _MahalanobisComponentState:
    centroids: torch.Tensor
    residual_projection: torch.Tensor
    woodbury_inv: torch.Tensor
    base_scale: float
    shrinkage_alpha: float
    covariance_trace_mean: float


class RSEV2Method(RSEMethod):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        representation_levels: Sequence[str] = ("head", "layer", "attn", "mlp"),
        covariance_shrinkage: str = "auto",
        shrinkage_alpha: float = 0.0,
        min_shrinkage_alpha: float = 1e-4,
        covariance_floor: float = 1e-6,
        confidence_floor: float = 0.0,
        progress_bar: bool = True,
    ):
        super().__init__(
            model=model,
            dataset_name=dataset_name,
            label_space=label_space,
            top_k=1,
            representation_levels=representation_levels,
            normalize_features=False,
            selection_metric="fdr",
            selection_strategy="topk",
            ensemble_weighting="uniform",
            score_normalization="none",
            routing_mode="none",
            progress_bar=progress_bar,
        )
        self.covariance_shrinkage = str(covariance_shrinkage).strip().lower()
        self.shrinkage_alpha = float(shrinkage_alpha)
        self.min_shrinkage_alpha = float(min_shrinkage_alpha)
        self.covariance_floor = float(covariance_floor)
        self.confidence_floor = float(confidence_floor)
        if self.covariance_shrinkage not in {"auto", "fixed"}:
            raise ValueError("RSEv2 covariance_shrinkage must be 'auto' or 'fixed'")
        if not (0.0 <= self.shrinkage_alpha <= 1.0):
            raise ValueError("RSEv2 shrinkage_alpha must be in [0, 1]")
        if not (0.0 < self.min_shrinkage_alpha <= 1.0):
            raise ValueError("RSEv2 min_shrinkage_alpha must be in (0, 1]")
        if self.covariance_floor <= 0.0:
            raise ValueError("RSEv2 covariance_floor must be positive")
        if self.confidence_floor < 0.0:
            raise ValueError("RSEv2 confidence_floor must be non-negative")

        self.component_models: Dict[Tuple[str, int], _MahalanobisComponentState] = {}
        self.component_vote_weight_sums: Dict[Tuple[str, int], float] = {}
        self.component_margin_sums: Dict[Tuple[str, int], float] = {}
        self.selection_train_accuracy = None

    @staticmethod
    def _compute_margin_vector(scores: torch.Tensor) -> torch.Tensor:
        return RSEMethod._compute_margin(scores)

    def _compute_oas_shrinkage(self, residuals: torch.Tensor) -> float:
        sample_count, dim = residuals.shape
        if sample_count <= 1 or dim <= 1:
            return 1.0

        gram = residuals @ residuals.transpose(0, 1)
        trace_cov = float(torch.trace(gram).item()) / float(sample_count)
        if trace_cov <= 0.0:
            return 1.0

        trace_cov_sq = float(torch.trace(gram @ gram).item()) / float(sample_count * sample_count)
        mu = trace_cov / float(dim)
        alpha = trace_cov_sq / float(dim * dim)
        denom = (float(sample_count) + 1.0) * (alpha - (mu * mu) / float(dim))
        if abs(denom) <= 1e-12:
            return 1.0
        shrinkage = (alpha + mu * mu) / denom
        return float(min(max(shrinkage, self.min_shrinkage_alpha), 1.0))

    def _fit_component_model(
        self,
        values: torch.Tensor,
        label_indices: torch.Tensor,
        num_classes: int,
    ) -> _MahalanobisComponentState:
        centroids, _ = self._compute_class_centroids(values, label_indices, num_classes, allow_missing=False)
        residuals = values.to(dtype=torch.float32) - centroids[label_indices].to(dtype=torch.float32)
        sample_count, dim = residuals.shape

        if self.covariance_shrinkage == "auto":
            shrinkage = self._compute_oas_shrinkage(residuals)
        else:
            shrinkage = float(min(max(self.shrinkage_alpha, self.min_shrinkage_alpha), 1.0))

        trace_rr = float(residuals.pow(2).sum().item())
        covariance_trace_mean = trace_rr / float(max(sample_count * dim, 1))
        base_scale = max(shrinkage * covariance_trace_mean, self.covariance_floor)
        residual_weight = max(1.0 - shrinkage, 0.0) / float(max(sample_count, 1))

        if residual_weight > 0.0 and sample_count > 0:
            residual_projection = residuals * residual_weight**0.5
            gram = residual_projection @ residual_projection.transpose(0, 1)
            woodbury_matrix = torch.eye(sample_count, dtype=torch.float32) + gram / base_scale
            try:
                woodbury_inv = torch.linalg.inv(woodbury_matrix)
            except RuntimeError:
                woodbury_inv = torch.linalg.pinv(woodbury_matrix)
        else:
            residual_projection = torch.zeros((0, dim), dtype=torch.float32)
            woodbury_inv = torch.zeros((0, 0), dtype=torch.float32)

        return _MahalanobisComponentState(
            centroids=centroids.to(dtype=torch.float32).cpu(),
            residual_projection=residual_projection.to(dtype=torch.float32).cpu(),
            woodbury_inv=woodbury_inv.to(dtype=torch.float32).cpu(),
            base_scale=float(base_scale),
            shrinkage_alpha=float(shrinkage),
            covariance_trace_mean=float(covariance_trace_mean),
        )

    def _score_component(self, component: Tuple[str, int], query: torch.Tensor) -> torch.Tensor:
        state = self.component_models[component]
        query = query.to(dtype=torch.float32)
        diffs = query.unsqueeze(0) - state.centroids
        base_precision = 1.0 / float(state.base_scale)
        squared_norm = diffs.pow(2).sum(dim=-1)
        distances = base_precision * squared_norm

        if state.residual_projection.numel() > 0:
            projections = state.residual_projection @ diffs.transpose(0, 1)
            corrected = state.woodbury_inv @ projections
            distances = distances - (base_precision * base_precision) * (projections * corrected).sum(dim=0)

        distances = distances.clamp_min(0.0)
        return (-0.5 * distances).to(dtype=torch.float32)

    def _combine_scores_single(
        self,
        component_scores: Dict[Tuple[str, int], torch.Tensor],
    ) -> tuple[torch.Tensor, float, Dict[Tuple[str, int], float], Dict[Tuple[str, int], float]]:
        components = list(component_scores)
        if not components:
            raise ValueError("RSEv2 requires at least one component")

        stacked_scores = torch.stack([component_scores[component] for component in components], dim=0)
        component_margins = self._compute_margin_vector(stacked_scores).clamp_min(0.0)
        raw_weights = torch.where(
            component_margins > self.confidence_floor,
            component_margins,
            torch.zeros_like(component_margins),
        )
        if float(raw_weights.sum().item()) <= 0.0:
            raw_weights = torch.ones_like(raw_weights)
        normalized_weights = raw_weights / raw_weights.sum().clamp_min(1e-6)
        final_scores = (stacked_scores * normalized_weights.unsqueeze(-1)).sum(dim=0)
        final_margin = float(self._compute_margin_vector(final_scores.unsqueeze(0))[0].item())
        margin_dict = {
            component: float(component_margins[idx].item())
            for idx, component in enumerate(components)
        }
        weight_dict = {
            component: float(normalized_weights[idx].item())
            for idx, component in enumerate(components)
        }
        return final_scores, final_margin, margin_dict, weight_dict

    def _reset_eval_diagnostics(self) -> None:
        super()._reset_eval_diagnostics()
        self.component_vote_weight_sums = {
            (level, layer_idx): 0.0
            for level in self.representation_levels
            for layer_idx in range(self.num_layers)
        }
        self.component_margin_sums = {
            (level, layer_idx): 0.0
            for level in self.representation_levels
            for layer_idx in range(self.num_layers)
        }

    def fit(self, train_data: Sequence[Dict]) -> None:
        if not train_data:
            raise ValueError("RSEv2 fit requires non-empty train_data")

        train_labels = [str(item["label"]) for item in train_data]
        unique_train_labels = {label for label in train_labels}
        self.class_labels = [label for label in self.label_space if label in unique_train_labels]
        self.label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
        label_indices = torch.tensor([self.label_to_index[label] for label in train_labels], dtype=torch.long)
        self._train_label_indices = label_indices

        feature_table = self._collect_feature_table(train_data, desc="RSEv2 fit features")
        self.component_models = {}
        self.metric_scores = {}
        self.selected_components = []
        self.selected_component_weights = {}
        self.selected_component_margin_thresholds = {}

        for level in self.representation_levels:
            table = feature_table[level]
            for layer_idx in range(table.shape[1]):
                component = (level, layer_idx)
                self.component_models[component] = self._fit_component_model(
                    table[:, layer_idx, :].to(dtype=torch.float32),
                    label_indices,
                    len(self.class_labels),
                )
                self.metric_scores[component] = 1.0
                self.selected_components.append(component)
                self.selected_component_weights[component] = 1.0 / float(
                    len(self.representation_levels) * self.num_layers
                )

        self._reset_eval_diagnostics()
        self._fitted = True

    def _component_scores(
        self,
        sample_features: Dict[str, torch.Tensor],
    ) -> Dict[Tuple[str, int], torch.Tensor]:
        all_scores: Dict[Tuple[str, int], torch.Tensor] = {}
        for level in self.representation_levels:
            level_features = sample_features[level]
            for layer_idx in range(level_features.shape[0]):
                query = level_features[layer_idx].to(dtype=torch.float32)
                all_scores[(level, layer_idx)] = self._score_component((level, layer_idx), query)
        return all_scores

    def predict(self, sample: Dict) -> str:
        if not self._fitted:
            raise RuntimeError("RSEv2 method is not fitted. Call fit(train_data) first.")

        sample_features = self._extract_multilevel_features(sample)
        all_scores = self._component_scores(sample_features)
        final_scores, final_margin, component_margins, component_weights = self._combine_scores_single(all_scores)
        pred_label = self.class_labels[int(final_scores.argmax().item())]
        self._record_eval(str(sample.get("label", "")), pred_label, final_margin, all_scores, fallback_used=False)
        for component, weight in component_weights.items():
            self.component_vote_weight_sums[component] += float(weight)
        for component, margin in component_margins.items():
            self.component_margin_sums[component] += float(margin)
        return pred_label

    def export_diagnostics(self) -> Dict:
        if not self._fitted:
            return {}

        component_table = []
        for level in self.representation_levels:
            for layer_idx in range(self.num_layers):
                component = (level, layer_idx)
                stats = self._component_eval_stats.get(component, {"correct": 0.0, "count": 0.0})
                count = float(stats["count"])
                state = self.component_models[component]
                mean_vote_weight = (
                    self.component_vote_weight_sums[component] / self._final_total if self._final_total > 0 else None
                )
                mean_margin = (
                    self.component_margin_sums[component] / self._final_total if self._final_total > 0 else None
                )
                component_table.append(
                    {
                        "level": level,
                        "layer_idx": layer_idx,
                        "selected": True,
                        "selection_score": None,
                        "fdr": None,
                        "loo_accuracy": None,
                        "cv_accuracy": None,
                        "weight": None,
                        "adaptive_margin_threshold": None,
                        "shrinkage_alpha": state.shrinkage_alpha,
                        "covariance_trace_mean": state.covariance_trace_mean,
                        "mean_vote_weight": mean_vote_weight,
                        "mean_margin": mean_margin,
                        "val_accuracy": (float(stats["correct"]) / count) if count > 0 else None,
                    }
                )

        component_table.sort(
            key=lambda row: (
                -(row["val_accuracy"] if row["val_accuracy"] is not None else -1.0),
                -(row["mean_vote_weight"] if row["mean_vote_weight"] is not None else -1.0),
                row["level"],
                row["layer_idx"],
            )
        )

        best_by_val = max(
            [row for row in component_table if row["val_accuracy"] is not None],
            key=lambda row: row["val_accuracy"],
            default=None,
        )
        margin_mean = (sum(self._margins) / len(self._margins)) if self._margins else None
        margin_min = min(self._margins) if self._margins else None
        margin_max = max(self._margins) if self._margins else None

        return {
            "method": "rsev2",
            "representation_levels": list(self.representation_levels),
            "num_layers": self.num_layers,
            "num_components": len(self.selected_components),
            "covariance_shrinkage": self.covariance_shrinkage,
            "shrinkage_alpha": self.shrinkage_alpha if self.covariance_shrinkage == "fixed" else None,
            "min_shrinkage_alpha": self.min_shrinkage_alpha,
            "covariance_floor": self.covariance_floor,
            "confidence_floor": self.confidence_floor,
            "class_labels": list(self.class_labels),
            "selected_components": component_table,
            "best_component_by_loo": None,
            "best_component_by_cv": None,
            "best_component_by_val": best_by_val,
            "component_table": component_table,
            "train_selection_summary": {
                "selection_train_accuracy": None,
            },
            "eval_summary": {
                "final_accuracy": (self._final_correct / self._final_total) if self._final_total > 0 else None,
                "final_correct": self._final_correct,
                "final_total": self._final_total,
                "fallback_used": 0,
                "margin_mean": margin_mean,
                "margin_min": margin_min,
                "margin_max": margin_max,
            },
        }
