from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .rse import RSEMethod


class _FrozenFeatureMethodBase(RSEMethod):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        feature_level: str = "layer",
        feature_index: int = -1,
        normalize_features: bool = True,
        progress_bar: bool = True,
    ):
        self.feature_level = str(feature_level).strip().lower()
        self.feature_index = int(feature_index)

        super().__init__(
            model=model,
            dataset_name=dataset_name,
            label_space=label_space,
            top_k=1,
            representation_levels=(self.feature_level,),
            normalize_features=normalize_features,
            selection_metric="fdr",
            selection_strategy="topk",
            ensemble_weighting="uniform",
            score_normalization="none",
            routing_mode="none",
            adaptive_margin_threshold=-1.0,
            adaptive_margin_quantile=0.5,
            greedy_pool_size=1,
            cv_folds=2,
            cv_seed=42,
            centroid_shrinkage="none",
            shrinkage_alpha=1.0,
            feature_reduction="none",
            pca_dim=128,
            fallback_margin_threshold=-1.0,
            fallback_margin_quantile=-1.0,
            fallback_margin_source="best_component",
            progress_bar=progress_bar,
        )

        self.class_labels = []
        self.label_to_index = {}
        self.selected_component_index: int | None = None
        self.feature_dim = 0
        self.train_sample_count = 0
        self.train_accuracy: float | None = None
        self._fitted = False

    def _prepare_class_labels(self, train_data: Sequence[Dict]) -> torch.Tensor:
        observed_labels = {str(item["label"]).strip() for item in train_data}
        self.class_labels = [label for label in self.label_space if label in observed_labels]
        remaining_labels = sorted(observed_labels.difference(set(self.class_labels)))
        self.class_labels.extend(remaining_labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.class_labels)}

        if not self.class_labels:
            raise RuntimeError("Frozen-feature baseline requires at least one class in the support set")

        return torch.tensor(
            [self.label_to_index[str(item["label"]).strip()] for item in train_data],
            dtype=torch.long,
        )

    def _resolve_component_index(self, component_count: int) -> int:
        resolved = self.feature_index if self.feature_index >= 0 else component_count + self.feature_index
        if resolved < 0 or resolved >= component_count:
            raise IndexError(
                f"{type(self).__name__} feature_index={self.feature_index} is invalid for "
                f"feature_level='{self.feature_level}' with {component_count} components"
            )
        return resolved

    def _select_train_features(self, train_data: Sequence[Dict], desc: str) -> torch.Tensor:
        feature_table = self._collect_feature_table(train_data, desc=desc)
        level_table = feature_table[self.feature_level].to(dtype=torch.float32)
        self.selected_component_index = self._resolve_component_index(int(level_table.shape[1]))
        selected = level_table[:, self.selected_component_index, :].cpu()
        self.train_sample_count = int(selected.shape[0])
        self.feature_dim = int(selected.shape[-1])
        return selected

    def _select_query_features(self, sample: Dict) -> torch.Tensor:
        sample_features = self._extract_multilevel_features(sample)
        level_features = sample_features[self.feature_level].to(dtype=torch.float32)
        component_index = self.selected_component_index
        if component_index is None:
            component_index = self._resolve_component_index(int(level_features.shape[0]))
        return level_features[component_index : component_index + 1].cpu()

    def fit(self, train_data: Sequence[Dict]) -> None:
        if not train_data:
            raise RuntimeError(f"{type(self).__name__} requires a non-empty support set")

        label_indices = self._prepare_class_labels(train_data)
        train_features = self._select_train_features(train_data, desc=f"Fitting {type(self).__name__}")
        self._fit_classifier(train_features, label_indices)
        train_scores = self._score_queries(train_features)
        train_predictions = train_scores.argmax(dim=1)
        self.train_accuracy = float((train_predictions == label_indices).float().mean().item())
        self._fitted = True

    def predict(self, sample: Dict) -> str:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call fit(train_data) first.")

        query_features = self._select_query_features(sample)
        scores = self._score_queries(query_features)
        pred_index = int(scores.argmax(dim=1).item())
        return self.class_labels[pred_index]

    def export_diagnostics(self) -> Dict:
        if not self._fitted:
            return {}
        payload = {
            "feature_level": self.feature_level,
            "feature_index": self.feature_index,
            "resolved_component_index": self.selected_component_index,
            "feature_dim": self.feature_dim,
            "train_sample_count": self.train_sample_count,
            "num_classes": len(self.class_labels),
            "train_accuracy": self.train_accuracy,
            "normalize_features": bool(self.normalize_features),
        }
        payload.update(self._extra_diagnostics())
        return payload

    def _fit_classifier(self, train_features: torch.Tensor, label_indices: torch.Tensor) -> None:
        raise NotImplementedError

    def _score_queries(self, query_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _extra_diagnostics(self) -> Dict:
        return {}


class WhitenedNCMMethod(_FrozenFeatureMethodBase):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        feature_level: str = "layer",
        feature_index: int = -1,
        normalize_features: bool = True,
        covariance_shrinkage: str = "auto",
        shrinkage_alpha: float = 0.1,
        covariance_floor: float = 1e-6,
        progress_bar: bool = True,
    ):
        super().__init__(
            model=model,
            dataset_name=dataset_name,
            label_space=label_space,
            feature_level=feature_level,
            feature_index=feature_index,
            normalize_features=normalize_features,
            progress_bar=progress_bar,
        )
        self.covariance_shrinkage = str(covariance_shrinkage).strip().lower()
        self.shrinkage_alpha = float(shrinkage_alpha)
        self.covariance_floor = float(covariance_floor)

        if self.covariance_shrinkage not in {"none", "fixed", "auto"}:
            raise ValueError("WhitenedNCM covariance_shrinkage must be one of: none, fixed, auto")
        if not (0.0 <= self.shrinkage_alpha <= 1.0):
            raise ValueError("WhitenedNCM shrinkage_alpha must be in [0, 1]")
        if self.covariance_floor <= 0.0:
            raise ValueError("WhitenedNCM covariance_floor must be positive")

        self.global_mean: torch.Tensor | None = None
        self.whitener: torch.Tensor | None = None
        self.class_centroids: torch.Tensor | None = None
        self.present_mask: torch.Tensor | None = None
        self.effective_shrinkage_alpha: float = 0.0

    def _resolve_shrinkage_alpha(self, sample_count: int, dim: int) -> float:
        if self.covariance_shrinkage == "none":
            return 0.0
        if self.covariance_shrinkage == "fixed":
            return self.shrinkage_alpha
        return float(dim) / float(max(sample_count, 1) + dim)

    def _fit_classifier(self, train_features: torch.Tensor, label_indices: torch.Tensor) -> None:
        features = train_features.to(dtype=torch.float32)
        self.global_mean = features.mean(dim=0)
        centered = features - self.global_mean.unsqueeze(0)

        dim = int(centered.shape[-1])
        sample_count = int(centered.shape[0])
        denom = max(sample_count - 1, 1)
        covariance = centered.transpose(0, 1) @ centered / float(denom)

        alpha = self._resolve_shrinkage_alpha(sample_count, dim)
        shrink_target = (torch.trace(covariance) / float(dim)) * torch.eye(dim, dtype=torch.float32)
        regularized_cov = (1.0 - alpha) * covariance + alpha * shrink_target
        regularized_cov = regularized_cov + self.covariance_floor * torch.eye(dim, dtype=torch.float32)
        self.effective_shrinkage_alpha = alpha

        eigenvalues, eigenvectors = torch.linalg.eigh(regularized_cov)
        inv_sqrt = eigenvalues.clamp_min(self.covariance_floor).rsqrt()
        self.whitener = (
            eigenvectors @ torch.diag(inv_sqrt) @ eigenvectors.transpose(0, 1)
        ).to(dtype=torch.float32).cpu()

        whitened = centered @ self.whitener
        if self.normalize_features:
            whitened = F.normalize(whitened, dim=-1)

        class_centroids, present_mask = self._compute_class_centroids(
            whitened,
            label_indices,
            len(self.class_labels),
        )
        if self.normalize_features:
            class_centroids = F.normalize(class_centroids, dim=-1)

        self.class_centroids = class_centroids.cpu()
        self.present_mask = present_mask.cpu()

    def _score_queries(self, query_features: torch.Tensor) -> torch.Tensor:
        if self.global_mean is None or self.whitener is None or self.class_centroids is None:
            raise RuntimeError("WhitenedNCM is missing fitted whitening statistics")

        centered = query_features.to(dtype=torch.float32) - self.global_mean.unsqueeze(0)
        whitened = centered @ self.whitener
        if self.normalize_features:
            whitened = F.normalize(whitened, dim=-1)
            scores = whitened @ self.class_centroids.transpose(0, 1)
        else:
            diffs = whitened.unsqueeze(1) - self.class_centroids.unsqueeze(0)
            scores = -diffs.pow(2).sum(dim=-1)

        if self.present_mask is not None and not bool(self.present_mask.all()):
            scores[:, ~self.present_mask] = -1e9
        return scores

    def _extra_diagnostics(self) -> Dict:
        return {
            "covariance_shrinkage": self.covariance_shrinkage,
            "shrinkage_alpha": self.shrinkage_alpha,
            "effective_shrinkage_alpha": self.effective_shrinkage_alpha,
            "covariance_floor": self.covariance_floor,
        }


class RidgeProbeMethod(_FrozenFeatureMethodBase):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        feature_level: str = "layer",
        feature_index: int = -1,
        normalize_features: bool = True,
        ridge_lambda: float = 1.0,
        fit_bias: bool = True,
        progress_bar: bool = True,
    ):
        super().__init__(
            model=model,
            dataset_name=dataset_name,
            label_space=label_space,
            feature_level=feature_level,
            feature_index=feature_index,
            normalize_features=normalize_features,
            progress_bar=progress_bar,
        )
        self.ridge_lambda = float(ridge_lambda)
        self.fit_bias = bool(fit_bias)

        if self.ridge_lambda < 0.0:
            raise ValueError("RidgeProbe ridge_lambda must be non-negative")

        self.feature_mean: torch.Tensor | None = None
        self.weight_matrix: torch.Tensor | None = None

    def _augment_bias(self, values: torch.Tensor) -> torch.Tensor:
        if not self.fit_bias:
            return values
        ones = torch.ones((values.shape[0], 1), dtype=values.dtype)
        return torch.cat([values, ones], dim=1)

    def _fit_classifier(self, train_features: torch.Tensor, label_indices: torch.Tensor) -> None:
        features = train_features.to(dtype=torch.float32)
        self.feature_mean = features.mean(dim=0)
        centered = features - self.feature_mean.unsqueeze(0)
        if self.normalize_features:
            centered = F.normalize(centered, dim=-1)

        design = self._augment_bias(centered)
        targets = F.one_hot(label_indices, num_classes=len(self.class_labels)).to(dtype=torch.float32)

        reg_diag = torch.full((design.shape[1],), self.ridge_lambda, dtype=torch.float32)
        if self.fit_bias:
            reg_diag[-1] = 0.0
        lhs = design.transpose(0, 1) @ design + torch.diag(reg_diag)
        rhs = design.transpose(0, 1) @ targets

        try:
            weights = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            weights = torch.linalg.pinv(lhs) @ rhs

        self.weight_matrix = weights.cpu()

    def _score_queries(self, query_features: torch.Tensor) -> torch.Tensor:
        if self.feature_mean is None or self.weight_matrix is None:
            raise RuntimeError("RidgeProbe is missing fitted linear weights")

        centered = query_features.to(dtype=torch.float32) - self.feature_mean.unsqueeze(0)
        if self.normalize_features:
            centered = F.normalize(centered, dim=-1)

        design = self._augment_bias(centered)
        return design @ self.weight_matrix

    def _extra_diagnostics(self) -> Dict:
        return {
            "ridge_lambda": self.ridge_lambda,
            "fit_bias": self.fit_bias,
        }
