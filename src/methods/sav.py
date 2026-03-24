from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..data import build_prompt
from .base import MethodBase


class SAVMethod(MethodBase):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        num_heads: int = 20,
        selection_strategy: str = "topk",
        selection_seed: int = 42,
        prototype_mode: str = "mean",
        class_bank_size: int = 4,
        vote_weighting: str = "uniform",
        progress_bar: bool = True,
    ):
        super().__init__(model=model, dataset_name=dataset_name, label_space=label_space)
        self.num_heads = int(num_heads)
        self.selection_strategy = str(selection_strategy).strip().lower()
        self.selection_seed = int(selection_seed)
        self.prototype_mode = str(prototype_mode).strip().lower()
        self.class_bank_size = int(class_bank_size)
        self.vote_weighting = str(vote_weighting).strip().lower()
        self.progress_bar = bool(progress_bar)

        if self.vote_weighting not in {"head_accuracy", "uniform"}:
            raise ValueError("SAV vote_weighting must be 'uniform' or 'head_accuracy'")

        self.attn_hook_names = self._infer_attention_module_names()
        self.n_layers = len(self.attn_hook_names)
        self.n_heads = self._infer_num_heads()
        self.all_heads = [
            (layer_idx, head_idx, -1)
            for layer_idx in range(self.n_layers)
            for head_idx in range(self.n_heads)
        ]

        self.class_activations: torch.Tensor | None = None
        self.top_heads: List[Tuple[int, int, int]] = []
        self.int_to_str: Dict[int, str] = {}
        self.class_banks: Dict[str, torch.Tensor] = {}
        self.train_bank_activations: torch.Tensor | None = None
        self.train_bank_labels: List[str] = []
        self.selected_head_weights: torch.Tensor | None = None
        self._fitted = False

    def _iter_with_progress(self, data: Sequence[Dict], desc: str):
        if self.progress_bar:
            return tqdm(data, desc=desc)
        return data

    def _infer_attention_module_names(self) -> List[str]:
        names = []
        for name, _ in self.model.model.named_modules():
            if name.endswith("self_attn.o_proj") or name.endswith("self_attn.out_proj"):
                names.append(name)

        if not names:
            raise RuntimeError(
                "Could not find attention projection modules for SAV extraction. "
                "Expected module names ending with 'self_attn.o_proj' or 'self_attn.out_proj'."
            )

        def sort_key(module_name: str):
            nums = [int(v) for v in re.findall(r"\d+", module_name)]
            return tuple(nums) if nums else (10**9,)

        return sorted(names, key=sort_key)

    def _infer_num_heads(self) -> int:
        config = self.model.model.config
        text_config = getattr(config, "text_config", config)
        for attr_name in ["num_attention_heads", "n_heads", "num_heads"]:
            if hasattr(text_config, attr_name):
                return int(getattr(text_config, attr_name))
        raise RuntimeError("Unable to infer number of attention heads from model config")

    def _extract_all_head_activations(self, sample: Dict) -> torch.Tensor:
        images = self._load_images(sample)
        prompt = build_prompt(self.dataset_name, sample)

        captured: Dict[str, torch.Tensor] = {}

        def hook_fn(module, args, output, module_name=None):
            if not args:
                return
            maybe_tensor = args[0]
            if torch.is_tensor(maybe_tensor):
                captured[module_name] = maybe_tensor.detach()

        handles = self.model.register_forward_hook(self.attn_hook_names, hook_fn)
        if not isinstance(handles, list):
            handles = [handles]

        with torch.no_grad():
            _ = self.model.forward(images, [prompt])

        for handle in handles:
            handle.remove()

        missing = [name for name in self.attn_hook_names if name not in captured]
        if missing:
            missing_preview = ", ".join(missing[:3])
            raise RuntimeError(
                f"Missing attention activations for {len(missing)} modules, e.g. {missing_preview}"
            )

        layer_acts = []
        for module_name in self.attn_hook_names:
            activation = captured[module_name]
            if activation.ndim < 3:
                raise RuntimeError(
                    f"Unexpected activation shape {tuple(activation.shape)} for module {module_name}"
                )

            last_token = activation[0, -1, :]
            head_dim = last_token.shape[-1] // self.n_heads
            if head_dim * self.n_heads != last_token.shape[-1]:
                raise RuntimeError(
                    f"Hidden dim {last_token.shape[-1]} is not divisible by n_heads={self.n_heads}"
                )
            layer_acts.append(last_token.reshape(self.n_heads, head_dim))

        return torch.stack(layer_acts, dim=0)

    def _extract_selected_head_activations(
        self,
        sample: Dict,
        selected_heads: Sequence[Tuple[int, int, int]],
    ) -> torch.Tensor:
        all_head_acts = self._extract_all_head_activations(sample)
        selected = [all_head_acts[layer_idx, head_idx] for layer_idx, head_idx, _ in selected_heads]
        return torch.stack(selected, dim=0).to(dtype=torch.float32)

    def _compute_class_activations(
        self,
        train_data: Sequence[Dict],
        selected_heads: Sequence[Tuple[int, int, int]],
    ) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
        str_to_int: Dict[str, int] = {}
        int_to_str: Dict[int, str] = {}

        class_sums: Dict[str, torch.Tensor] = {}
        class_counts: Dict[str, int] = {}

        for item in self._iter_with_progress(train_data, desc="SAV class activations"):
            label = str(item["label"])
            head_act = self._extract_selected_head_activations(item, selected_heads)

            if label not in str_to_int:
                idx = len(str_to_int)
                str_to_int[label] = idx
                int_to_str[idx] = label
                class_sums[label] = head_act.clone()
                class_counts[label] = 1
            else:
                class_sums[label] = class_sums[label] + head_act
                class_counts[label] += 1

        ordered_labels = [int_to_str[idx] for idx in range(len(int_to_str))]
        class_activations = []
        for label in ordered_labels:
            class_activations.append(class_sums[label] / class_counts[label])

        return torch.stack(class_activations, dim=0), str_to_int, int_to_str

    def _collect_selected_activations(
        self,
        train_data: Sequence[Dict],
        selected_heads: Sequence[Tuple[int, int, int]],
    ) -> List[Tuple[str, torch.Tensor]]:
        collected: List[Tuple[str, torch.Tensor]] = []
        for item in self._iter_with_progress(train_data, desc="SAV selected activations"):
            label = str(item["label"])
            head_act = self._extract_selected_head_activations(item, selected_heads)
            collected.append((label, head_act))
        return collected

    def _compute_class_activations_from_collected(
        self,
        collected: Sequence[Tuple[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
        str_to_int: Dict[str, int] = {}
        int_to_str: Dict[int, str] = {}
        class_sums: Dict[str, torch.Tensor] = {}
        class_counts: Dict[str, int] = {}

        for label, head_act in collected:
            if label not in str_to_int:
                idx = len(str_to_int)
                str_to_int[label] = idx
                int_to_str[idx] = label
                class_sums[label] = head_act.clone()
                class_counts[label] = 1
            else:
                class_sums[label] = class_sums[label] + head_act
                class_counts[label] += 1

        ordered_labels = [int_to_str[idx] for idx in range(len(int_to_str))]
        class_activations = []
        for label in ordered_labels:
            class_activations.append(class_sums[label] / class_counts[label])

        return torch.stack(class_activations, dim=0), str_to_int, int_to_str

    def _select_diverse_prototypes(
        self,
        samples: torch.Tensor,
        bank_size: int,
    ) -> torch.Tensor:
        if samples.ndim != 3:
            raise ValueError(f"Expected class samples with shape [N, H, D], got {tuple(samples.shape)}")

        if bank_size <= 0:
            raise ValueError("class_bank_size must be positive")

        if samples.shape[0] <= bank_size:
            return samples

        flattened = samples.reshape(samples.shape[0], -1)
        flattened = F.normalize(flattened, dim=-1)
        class_mean = F.normalize(flattened.mean(dim=0, keepdim=True), dim=-1)
        mean_scores = F.cosine_similarity(flattened, class_mean, dim=-1)
        selected = [int(mean_scores.argmax().item())]

        min_dist = 1.0 - F.cosine_similarity(flattened, flattened[selected[0]].unsqueeze(0), dim=-1)
        while len(selected) < bank_size:
            min_dist[selected] = -1.0
            next_idx = int(min_dist.argmax().item())
            selected.append(next_idx)
            next_dist = 1.0 - F.cosine_similarity(flattened, flattened[next_idx].unsqueeze(0), dim=-1)
            min_dist = torch.minimum(min_dist, next_dist)

        return samples[selected]

    def _build_class_banks(
        self,
        collected: Sequence[Tuple[str, torch.Tensor]],
    ) -> tuple[Dict[str, torch.Tensor], Dict[int, str]]:
        grouped: Dict[str, List[torch.Tensor]] = {}
        int_to_str: Dict[int, str] = {}

        for label, head_act in collected:
            grouped.setdefault(label, []).append(head_act)

        class_banks: Dict[str, torch.Tensor] = {}
        for idx, label in enumerate(sorted(grouped.keys())):
            int_to_str[idx] = label
            samples = torch.stack(grouped[label], dim=0)
            class_banks[label] = self._select_diverse_prototypes(samples, self.class_bank_size)

        return class_banks, int_to_str

    def _select_heads(
        self,
        train_data: Sequence[Dict],
        all_class_acts: torch.Tensor,
        str_to_int: Dict[str, int],
    ) -> List[Tuple[int, int, int]]:
        total_heads = len(self.all_heads)
        if total_heads == 0:
            raise RuntimeError("SAV requires at least one attention head")

        if self.selection_strategy == "all":
            self.selected_head_weights = torch.ones(total_heads, dtype=torch.float32)
            return list(self.all_heads)

        k = min(self.num_heads, total_heads)
        if k <= 0:
            raise ValueError("SAV num_heads must be positive")

        if self.selection_strategy == "firstk":
            self.selected_head_weights = torch.ones(k, dtype=torch.float32)
            return list(self.all_heads[:k])

        if self.selection_strategy == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.selection_seed)
            indices = torch.randperm(total_heads, generator=generator)[:k].tolist()
            self.selected_head_weights = torch.ones(len(indices), dtype=torch.float32)
            return [self.all_heads[idx] for idx in indices]

        success_count = torch.zeros(len(self.all_heads), dtype=torch.int32, device=all_class_acts.device)

        iterator = self._iter_with_progress(train_data, desc="SAV head selection")
        for item in iterator:
            query = self._extract_selected_head_activations(item, self.all_heads).to(all_class_acts.device)
            label_idx = str_to_int[str(item["label"])]

            similarities = F.cosine_similarity(all_class_acts, query.unsqueeze(0), dim=-1)
            best_per_head = similarities.argmax(dim=0)
            success_count += (best_per_head == label_idx).to(dtype=torch.int32)

        if self.selection_strategy == "topk":
            selected_indices = torch.topk(success_count, k=k).indices.tolist()
            self.selected_head_weights = (success_count[selected_indices].to(dtype=torch.float32) / max(len(train_data), 1)).cpu()
            return [self.all_heads[idx] for idx in selected_indices]

        if self.selection_strategy == "bottomk":
            selected_indices = torch.topk(success_count, k=k, largest=False).indices.tolist()
            self.selected_head_weights = (success_count[selected_indices].to(dtype=torch.float32) / max(len(train_data), 1)).cpu()
            return [self.all_heads[idx] for idx in selected_indices]

        supported = ", ".join(["all", "bottomk", "firstk", "random", "topk"])
        raise ValueError(
            f"Unsupported SAV selection_strategy '{self.selection_strategy}'. "
            f"Supported strategies: {supported}"
        )

    def fit(self, train_data: Sequence[Dict]) -> None:
        if len(train_data) == 0:
            raise ValueError("SAV fit requires non-empty training data")

        all_class_acts, str_to_int, int_to_str = self._compute_class_activations(
            train_data, self.all_heads
        )

        self.top_heads = self._select_heads(train_data, all_class_acts, str_to_int)
        self.class_activations = None
        self.class_banks = {}
        self.train_bank_activations = None
        self.train_bank_labels = []
        self.selected_head_weights = None

        if self.prototype_mode == "mean" and len(self.top_heads) == len(self.all_heads):
            self.class_activations = all_class_acts
            self.int_to_str = int_to_str
            self._fitted = True
            return

        collected = self._collect_selected_activations(train_data, self.top_heads)

        if self.prototype_mode == "mean":
            top_class_acts, _, top_int_to_str = self._compute_class_activations_from_collected(collected)
            self.class_activations = top_class_acts
            self.int_to_str = top_int_to_str
        elif self.prototype_mode == "support_nn":
            self.train_bank_activations = torch.stack([head_act for _, head_act in collected], dim=0)
            self.train_bank_labels = [label for label, _ in collected]
            unique_labels = list(dict.fromkeys(self.train_bank_labels))
            self.int_to_str = {idx: label for idx, label in enumerate(unique_labels)}
        elif self.prototype_mode == "class_bank":
            self.class_banks, self.int_to_str = self._build_class_banks(collected)
        else:
            supported = ", ".join(["class_bank", "mean", "support_nn"])
            raise ValueError(
                f"Unsupported SAV prototype_mode '{self.prototype_mode}'. "
                f"Supported modes: {supported}"
            )
        self._fitted = True

    def predict_with_counts(self, sample: Dict) -> Tuple[str, Dict[str, float]]:
        if not self._fitted:
            raise RuntimeError("SAV method is not fitted. Call fit(train_data) first.")

        query = self._extract_selected_head_activations(sample, self.top_heads)

        if self.prototype_mode == "mean":
            if self.class_activations is None:
                raise RuntimeError("SAV mean prototype mode is missing class activations")
            query = query.to(self.class_activations.device)
            similarities = F.cosine_similarity(self.class_activations, query.unsqueeze(0), dim=-1)
            vote_indices = similarities.argmax(dim=0)
        elif self.prototype_mode == "support_nn":
            if self.train_bank_activations is None or not self.train_bank_labels:
                raise RuntimeError("SAV support_nn mode is missing support bank activations")
            query = query.to(self.train_bank_activations.device)
            similarities = F.cosine_similarity(self.train_bank_activations, query.unsqueeze(0), dim=-1)
            best_support_indices = similarities.argmax(dim=0).tolist()
            label_to_index = {label: idx for idx, label in self.int_to_str.items()}
            vote_indices = torch.tensor(
                [label_to_index[self.train_bank_labels[idx]] for idx in best_support_indices],
                device=query.device,
            )
        elif self.prototype_mode == "class_bank":
            if not self.class_banks:
                raise RuntimeError("SAV class_bank mode is missing class bank activations")
            ordered_labels = [self.int_to_str[idx] for idx in range(len(self.int_to_str))]
            class_scores = []
            for label in ordered_labels:
                bank = self.class_banks[label].to(query.device)
                similarities = F.cosine_similarity(bank, query.unsqueeze(0), dim=-1)
                class_scores.append(similarities.max(dim=0).values)
            vote_indices = torch.stack(class_scores, dim=0).argmax(dim=0)
        else:
            raise RuntimeError(f"Unsupported SAV prototype_mode '{self.prototype_mode}' at prediction time")

        if self.vote_weighting == "head_accuracy" and self.selected_head_weights is not None:
            weights = self.selected_head_weights.to(device=vote_indices.device, dtype=torch.float32)
            if int(weights.numel()) != int(vote_indices.numel()):
                weights = torch.ones_like(vote_indices, dtype=torch.float32)
        else:
            weights = torch.ones_like(vote_indices, dtype=torch.float32)

        counts = torch.zeros(len(self.int_to_str), dtype=torch.float32, device=vote_indices.device)
        counts.scatter_add_(0, vote_indices, weights)
        pred_index = int(counts.argmax().item())
        pred_label = self.int_to_str[pred_index]

        vote_counts = {
            self.int_to_str[index]: float(count.item())
            for index, count in enumerate(counts)
            if float(count.item()) > 0.0
        }
        return pred_label, vote_counts

    def predict(self, sample: Dict) -> str:
        pred_label, _ = self.predict_with_counts(sample)
        return pred_label
