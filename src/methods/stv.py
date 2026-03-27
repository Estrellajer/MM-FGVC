from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..data import build_prompt
from .base import MethodBase


@dataclass(frozen=True)
class _ProjectionMeta:
    attn_name: str
    proj_name: str
    num_heads: int
    head_dim: int


class _STVProjectionHook:
    def __init__(self, model_wrapper, attn_module_names: Sequence[str]):
        self.model_wrapper = model_wrapper

        named_modules = dict(self.model_wrapper.model.named_modules())
        self.layer_metas: List[_ProjectionMeta] = []
        self.proj_to_layer: Dict[str, int] = {}

        for layer_idx, attn_name in enumerate(attn_module_names):
            attn_module = named_modules.get(attn_name)
            if attn_module is None:
                raise ValueError(f"Attention module '{attn_name}' not found")

            proj_name = None
            for candidate in [f"{attn_name}.o_proj", f"{attn_name}.out_proj"]:
                if candidate in named_modules:
                    proj_name = candidate
                    break
            if proj_name is None:
                raise ValueError(f"Attention module '{attn_name}' is missing o_proj/out_proj")

            num_heads = self._infer_num_heads(attn_module)
            head_dim = self._infer_head_dim(attn_module, num_heads)

            self.layer_metas.append(
                _ProjectionMeta(
                    attn_name=attn_name,
                    proj_name=proj_name,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
            )
            self.proj_to_layer[proj_name] = layer_idx

        self.capture_enabled = False
        self.capture_token_index = -1
        self.captured: Dict[int, torch.Tensor] = {}

        self.intervention_enabled = False
        self.intervention_token_index = -1
        self.intervention_locations_by_layer: Dict[int, List[int]] = {}
        self.avg_activations: torch.Tensor | None = None

        handles = self.model_wrapper.register_forward_pre_hook(list(self.proj_to_layer), self._pre_hook)
        self._handles = handles if isinstance(handles, list) else [handles]

    def _infer_num_heads(self, attn_module) -> int:
        config = getattr(attn_module, "config", getattr(self.model_wrapper.model, "config", None))
        text_config = getattr(config, "text_config", config)

        for attr_name in ["num_heads", "num_attention_heads", "n_heads"]:
            if hasattr(attn_module, attr_name):
                return int(getattr(attn_module, attr_name))
            if text_config is not None and hasattr(text_config, attr_name):
                return int(getattr(text_config, attr_name))

        raise RuntimeError("Unable to infer number of attention heads for STV")

    def _infer_head_dim(self, attn_module, num_heads: int) -> int:
        head_dim = getattr(attn_module, "head_dim", None)
        if head_dim is not None:
            return int(head_dim)

        q_proj = getattr(attn_module, "q_proj", None)
        if q_proj is not None and hasattr(q_proj, "out_features"):
            return int(q_proj.out_features) // int(num_heads)

        hidden_size = getattr(attn_module, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size) // int(num_heads)

        config = getattr(attn_module, "config", getattr(self.model_wrapper.model, "config", None))
        text_config = getattr(config, "text_config", config)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            return int(text_config.hidden_size) // int(num_heads)

        raise RuntimeError("Unable to infer head dimension for STV")

    def _resolve_token_index(self, seq_len: int, token_index: int) -> int:
        resolved = token_index if token_index >= 0 else seq_len + token_index
        return max(0, min(seq_len - 1, resolved))

    def _pre_hook(self, module, args, module_name=None):
        if not args:
            return None

        hidden_states = args[0]
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
            return None

        layer_idx = self.proj_to_layer.get(module_name)
        if layer_idx is None:
            return None

        meta = self.layer_metas[layer_idx]
        reshaped = hidden_states.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            meta.num_heads,
            meta.head_dim,
        )

        if self.capture_enabled:
            token_idx = self._resolve_token_index(reshaped.shape[1], self.capture_token_index)
            self.captured[layer_idx] = reshaped[:, token_idx : token_idx + 1].detach().to(
                device="cpu",
                dtype=torch.float32,
            )

        if self.intervention_enabled and layer_idx in self.intervention_locations_by_layer:
            if self.avg_activations is None:
                raise RuntimeError("STV intervention was enabled without avg_activations")
            token_idx = self._resolve_token_index(reshaped.shape[1], self.intervention_token_index)
            patched = reshaped.clone()
            for head_idx in self.intervention_locations_by_layer[layer_idx]:
                patched[:, token_idx, head_idx] = self.avg_activations[layer_idx, head_idx, 0].to(
                    device=patched.device,
                    dtype=patched.dtype,
                )
            return (patched.view_as(hidden_states),)

        return None

    def capture(self, inputs, token_index: int) -> torch.Tensor:
        self.captured = {}
        self.capture_enabled = True
        self.capture_token_index = int(token_index)
        self.intervention_enabled = False

        try:
            with torch.no_grad():
                _ = self.model_wrapper.model(**inputs)
        finally:
            self.capture_enabled = False

        missing_layers = [layer_idx for layer_idx in range(len(self.layer_metas)) if layer_idx not in self.captured]
        if missing_layers:
            raise RuntimeError(f"Missing STV captures for {len(missing_layers)} layers")

        ordered = []
        for layer_idx in range(len(self.layer_metas)):
            layer_capture = self.captured[layer_idx]
            ordered.append(layer_capture.mean(dim=0).permute(1, 0, 2).contiguous())
        return torch.stack(ordered, dim=0)

    def enable_intervention(
        self,
        locations: Sequence[Tuple[int, int, int]],
        avg_activations: torch.Tensor,
        token_index: int,
    ) -> None:
        locations_by_layer: Dict[int, List[int]] = {}
        for layer_idx, head_idx, _ in locations:
            locations_by_layer.setdefault(int(layer_idx), []).append(int(head_idx))

        self.intervention_locations_by_layer = locations_by_layer
        self.avg_activations = avg_activations.to(device="cpu", dtype=torch.float32)
        self.intervention_token_index = int(token_index)
        self.intervention_enabled = True

    def disable_intervention(self) -> None:
        self.intervention_enabled = False
        self.intervention_locations_by_layer = {}
        self.avg_activations = None


class STVMethod(MethodBase):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        num_shots: int = 8,
        num_examples: int = 100,
        topk: int = 64,
        num_clusters: int = 32,
        kmeans_iters: int = 25,
        selection_epochs: int = 80,
        selection_samples_per_epoch: int = 8,
        selection_train_size: int = 16,
        selection_eval_size: int = 16,
        selection_lr: float = 0.1,
        final_selection_trials: int = 8,
        head_selection_mode: str = "sensitivity",
        cluster_selection_mode: str = "rl",
        support_strategy: str = "balanced",
        support_seed: int = 42,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 0.0,
        progress_bar: bool = True,
    ):
        super().__init__(model=model, dataset_name=dataset_name, label_space=label_space)

        self.num_shots = int(num_shots)
        self.num_examples = int(num_examples)
        self.topk = int(topk)
        self.num_clusters = int(num_clusters)
        self.kmeans_iters = int(kmeans_iters)
        self.selection_epochs = int(selection_epochs)
        self.selection_samples_per_epoch = int(selection_samples_per_epoch)
        self.selection_train_size = int(selection_train_size)
        self.selection_eval_size = int(selection_eval_size)
        self.selection_lr = float(selection_lr)
        self.final_selection_trials = int(final_selection_trials)
        self.head_selection_mode = str(head_selection_mode).strip().lower()
        self.cluster_selection_mode = str(cluster_selection_mode).strip().lower()
        self.support_strategy = str(support_strategy).strip().lower()
        self.support_seed = int(support_seed)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.progress_bar = bool(progress_bar)

        if self.num_shots <= 0:
            raise ValueError("STV num_shots must be positive")
        if self.num_examples <= 0:
            raise ValueError("STV num_examples must be positive")
        if self.topk <= 0:
            raise ValueError("STV topk must be positive")
        if self.num_clusters <= 0:
            raise ValueError("STV num_clusters must be positive")
        if self.selection_train_size <= 0:
            raise ValueError("STV selection_train_size must be positive")
        if self.head_selection_mode not in {"sav_accuracy", "sensitivity"}:
            raise ValueError("STV head_selection_mode must be 'sensitivity' or 'sav_accuracy'")
        if self.cluster_selection_mode not in {"query_adaptive", "rl"}:
            raise ValueError("STV cluster_selection_mode must be 'rl' or 'query_adaptive'")

        self.model_family = self._infer_model_family()
        self.attn_module_names = self._infer_attention_module_names()
        self.hook_controller = _STVProjectionHook(self.model, self.attn_module_names)

        first_meta = self.hook_controller.layer_metas[0]
        self.n_layers = len(self.hook_controller.layer_metas)
        self.n_heads = first_meta.num_heads
        self.head_dim = first_meta.head_dim
        self.all_heads = [
            (layer_idx, head_idx, -1)
            for layer_idx in range(self.n_layers)
            for head_idx in range(self.n_heads)
        ]

        self._label_to_indices: Dict[str, List[int]] = {}
        self.avg_diff: torch.Tensor | None = None
        self.intervention_locations: List[Tuple[int, int, int]] = []
        self.cluster_centers: torch.Tensor | None = None
        self.avg_activations: torch.Tensor | None = None
        self.last_query_cluster_index: int | None = None
        self._fitted = False

    def _infer_model_family(self) -> str:
        names = " ".join(
            [
                type(self.model).__name__.lower(),
                type(self.model.model).__name__.lower(),
                type(self.model.processor).__name__.lower(),
            ]
        )
        if "qwen2" in names:
            return "qwen2"
        if "qwen3" in names:
            return "qwen3"
        if "idefics3" in names:
            return "idefics3"
        raise ValueError("STV baseline currently supports qwen2_vl, qwen3_vl, and idefics3 style models only")

    def _infer_attention_module_names(self) -> List[str]:
        names = []
        for name, module in self.model.model.named_modules():
            if not name.endswith("self_attn"):
                continue
            if any(hasattr(module, attr) for attr in ["o_proj", "out_proj"]):
                names.append(name)
        names = self._filter_text_backbone_module_names(names)

        if not names:
            raise RuntimeError("Could not find decoder self-attention modules for STV")

        def sort_key(module_name: str):
            nums = [int(v) for v in re.findall(r"\d+", module_name)]
            return tuple(nums) if nums else (10**9,)

        return sorted(names, key=sort_key)

    def _iter_with_progress(self, data, desc: str):
        if self.progress_bar:
            return tqdm(data, desc=desc)
        return data

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip().lower())
        return re.sub(r"[^\w\s-]", "", text)

    def _match_label(self, output: str) -> str:
        output_norm = self._normalize_text(output)
        label_map = {self._normalize_text(label): label for label in self.label_space}

        if output_norm in label_map:
            return label_map[output_norm]

        for norm_label, raw_label in label_map.items():
            if output_norm.startswith(norm_label) or norm_label in output_norm:
                return raw_label

        if self.label_space:
            first_token = output_norm.split(" ")[0] if output_norm else ""
            short_map = {
                self._normalize_text(label).split(" ")[0]: label
                for label in self.label_space
                if label.strip()
            }
            if first_token in short_map:
                return short_map[first_token]

        return output.strip()

    def _move_inputs(self, inputs):
        return inputs.to(self.model.device)

    def _build_user_message(self, sample: Dict) -> tuple[Dict, List[Image.Image]]:
        prompt = build_prompt(self.dataset_name, sample)
        if self.model_family == "idefics3":
            prompt = self._sanitize_prompt_text_for_explicit_images(prompt) or "Describe this image."
        images = self._load_images(sample)
        if self.model_family in {"qwen2", "qwen3"}:
            content = [{"type": "image", "image": image} for image in images]
        else:
            content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        return {"role": "user", "content": content}, images

    def _build_conversation(
        self,
        demos: Sequence[Dict],
        query: Dict,
        include_answer: bool,
    ) -> tuple[List[Dict], List[Image.Image]]:
        conversation: List[Dict] = []
        flat_images: List[Image.Image] = []

        for demo in demos:
            user_message, images = self._build_user_message(demo)
            conversation.append(user_message)
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": str(demo["label"])}]})
            flat_images.extend(images)

        query_user, query_images = self._build_user_message(query)
        conversation.append(query_user)
        flat_images.extend(query_images)
        if include_answer:
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": str(query["label"])}]})

        return conversation, flat_images

    def _prepare_inputs(
        self,
        conversation: Sequence[Dict],
        flat_images: Sequence[Image.Image],
        *,
        add_generation_prompt: bool,
    ):
        if self.model_family in {"qwen2", "qwen3"}:
            inputs = self.model.processor.apply_chat_template(
                [list(conversation)],
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            return self._move_inputs(inputs)

        prompt = self.model.processor.apply_chat_template(
            list(conversation),
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        inputs = self.model.processor(
            text=[prompt],
            images=[list(flat_images)],
            return_tensors="pt",
            padding=True,
        )
        return self._move_inputs(inputs)

    def _build_answer_only_labels(self, full_inputs, prompt_inputs) -> torch.Tensor:
        input_ids = full_inputs["input_ids"]
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        if prompt_len >= int(input_ids.shape[-1]):
            prompt_len = max(0, int(input_ids.shape[-1]) - 1)
        labels = torch.full_like(input_ids, fill_value=-100)
        labels[:, prompt_len:] = input_ids[:, prompt_len:]
        return labels

    def _build_label_index(self, train_data: Sequence[Dict]) -> None:
        self._label_to_indices = {}
        for idx, item in enumerate(train_data):
            self._label_to_indices.setdefault(str(item["label"]), []).append(idx)

    def _sample_demo_indices(self, train_data: Sequence[Dict], query_index: int, rng: random.Random) -> List[int]:
        candidate_indices = [idx for idx in range(len(train_data)) if idx != query_index]
        if not candidate_indices:
            candidate_indices = [query_index]

        if self.support_strategy == "random":
            if len(candidate_indices) >= self.num_shots:
                return rng.sample(candidate_indices, self.num_shots)
            return [rng.choice(candidate_indices) for _ in range(self.num_shots)]

        if self.support_strategy != "balanced":
            raise ValueError("STV support_strategy must be 'balanced' or 'random'")

        labels = list(self._label_to_indices.keys())
        rng.shuffle(labels)
        selected: List[int] = []

        while len(selected) < self.num_shots:
            added = False
            for label in labels:
                pool = [idx for idx in self._label_to_indices[label] if idx != query_index]
                if not pool:
                    continue
                selected.append(rng.choice(pool))
                added = True
                if len(selected) >= self.num_shots:
                    break
            if not added:
                break

        while len(selected) < self.num_shots:
            selected.append(rng.choice(candidate_indices))

        return selected

    def _sample_subset_indices(self, total_size: int, target_size: int, rng: random.Random) -> List[int]:
        all_indices = list(range(total_size))
        rng.shuffle(all_indices)
        return all_indices[: min(total_size, target_size)]

    def _capture_prompt_activations(self, sample: Dict, demos: Sequence[Dict]) -> torch.Tensor:
        conversation, flat_images = self._build_conversation(demos, sample, include_answer=False)
        inputs = self._prepare_inputs(conversation, flat_images, add_generation_prompt=True)
        token_index = int(inputs["input_ids"].shape[-1]) - 1
        return self.hook_controller.capture(inputs, token_index=token_index)

    def _estimate_avg_diff(
        self,
        train_data: Sequence[Dict],
        activation_indices: Sequence[int],
        rng: random.Random,
    ) -> torch.Tensor:
        diffs = []
        iterator = self._iter_with_progress(activation_indices, desc="STV diff estimation")
        for query_index in iterator:
            sample = train_data[query_index]
            demo_indices = self._sample_demo_indices(train_data, query_index, rng)
            demos = [train_data[idx] for idx in demo_indices]
            query_only = self._capture_prompt_activations(sample, demos=[])
            query_with_context = self._capture_prompt_activations(sample, demos=demos)
            diffs.append(query_with_context - query_only)

        if not diffs:
            raise RuntimeError("STV failed to collect any activation differences")
        return torch.stack(diffs, dim=0).mean(dim=0)

    def _select_topk_locations(self, avg_diff: torch.Tensor) -> List[Tuple[int, int, int]]:
        impact = avg_diff.norm(dim=-1).squeeze(-1)
        return self._select_topk_locations_from_scores(impact)

    def _select_topk_locations_from_scores(self, scores: torch.Tensor) -> List[Tuple[int, int, int]]:
        flat = scores.flatten()
        k = min(self.topk, int(flat.numel()))
        topk_indices = torch.topk(flat, k=k).indices.tolist()

        intervention_locations: List[Tuple[int, int, int]] = []
        for index in topk_indices:
            layer_idx = int(index) // scores.shape[1]
            head_idx = int(index) % scores.shape[1]
            intervention_locations.append((layer_idx, head_idx, -1))
        return intervention_locations

    def _collect_query_only_head_activations(
        self,
        train_data: Sequence[Dict],
    ) -> List[Tuple[str, torch.Tensor]]:
        collected: List[Tuple[str, torch.Tensor]] = []
        iterator = self._iter_with_progress(train_data, desc="STV SAV activations")
        for item in iterator:
            activations = self._capture_prompt_activations(item, demos=[]).squeeze(2).to(device="cpu", dtype=torch.float32)
            collected.append((str(item["label"]), activations))
        return collected

    def _select_topk_locations_sav_accuracy(self, train_data: Sequence[Dict]) -> List[Tuple[int, int, int]]:
        collected = self._collect_query_only_head_activations(train_data)
        if not collected:
            raise RuntimeError("STV SAV-style head selection requires non-empty train data")

        str_to_int: Dict[str, int] = {}
        int_to_str: Dict[int, str] = {}
        class_sums: Dict[str, torch.Tensor] = {}
        class_counts: Dict[str, int] = {}

        for label, activations in collected:
            if label not in str_to_int:
                label_idx = len(str_to_int)
                str_to_int[label] = label_idx
                int_to_str[label_idx] = label
                class_sums[label] = activations.clone()
                class_counts[label] = 1
            else:
                class_sums[label] = class_sums[label] + activations
                class_counts[label] += 1

        ordered_labels = [int_to_str[idx] for idx in range(len(int_to_str))]
        class_activations = torch.stack(
            [class_sums[label] / class_counts[label] for label in ordered_labels],
            dim=0,
        )

        success_count = torch.zeros(self.n_layers, self.n_heads, dtype=torch.int32)
        iterator = self._iter_with_progress(collected, desc="STV SAV head scoring")
        for label, activations in iterator:
            similarities = F.cosine_similarity(class_activations, activations.unsqueeze(0), dim=-1)
            best_per_head = similarities.argmax(dim=0)
            success_count += (best_per_head == str_to_int[label]).to(dtype=torch.int32)

        return self._select_topk_locations_from_scores(success_count.to(dtype=torch.float32))

    def _collect_cluster_samples(
        self,
        train_data: Sequence[Dict],
        activation_indices: Sequence[int],
        rng: random.Random,
    ) -> torch.Tensor:
        samples = []
        iterator = self._iter_with_progress(activation_indices, desc="STV cluster samples")
        for query_index in iterator:
            sample = train_data[query_index]
            demo_indices = self._sample_demo_indices(train_data, query_index, rng)
            demos = [train_data[idx] for idx in demo_indices]
            samples.append(self._capture_prompt_activations(sample, demos=demos))

        if not samples:
            raise RuntimeError("STV failed to collect cluster activations")
        return torch.stack(samples, dim=0)

    def _run_kmeans(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if data.ndim != 2:
            raise ValueError(f"Expected 2D tensor for kmeans, got {tuple(data.shape)}")

        data = data.to(device="cpu", dtype=torch.float32)
        num_samples = int(data.shape[0])
        num_clusters = min(self.num_clusters, num_samples)
        if num_clusters <= 0:
            raise ValueError("STV num_clusters must be positive after clamping")
        if num_clusters == 1:
            return data[:1].clone(), torch.zeros(num_samples, dtype=torch.long)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.support_seed)
        init_indices = torch.randperm(num_samples, generator=generator)[:num_clusters]
        centers = data[init_indices].clone()

        for _ in range(max(1, self.kmeans_iters)):
            distances = torch.cdist(data, centers)
            assignments = distances.argmin(dim=1)
            new_centers = centers.clone()
            for cluster_idx in range(num_clusters):
                mask = assignments == cluster_idx
                if mask.any():
                    new_centers[cluster_idx] = data[mask].mean(dim=0)
                else:
                    replacement = int(torch.randint(0, num_samples, (1,), generator=generator).item())
                    new_centers[cluster_idx] = data[replacement]

            if torch.allclose(new_centers, centers, atol=1e-4, rtol=1e-4):
                centers = new_centers
                break
            centers = new_centers

        final_distances = torch.cdist(data, centers)
        return centers, final_distances.argmin(dim=1)

    def _build_cluster_centers(self, contextual_samples: torch.Tensor) -> torch.Tensor:
        num_samples = contextual_samples.shape[0]
        flattened = contextual_samples.reshape(num_samples, -1)
        centers, _ = self._run_kmeans(flattened)
        return centers.view(centers.shape[0], self.n_layers, self.n_heads, 1, self.head_dim)

    def _build_selection_record(self, sample: Dict) -> Dict[str, torch.Tensor | int]:
        prompt_conversation, prompt_images = self._build_conversation([], sample, include_answer=False)
        full_conversation, full_images = self._build_conversation([], sample, include_answer=True)

        prompt_inputs = self._prepare_inputs(prompt_conversation, prompt_images, add_generation_prompt=True)
        full_inputs = self._prepare_inputs(full_conversation, full_images, add_generation_prompt=False)
        labels = self._build_answer_only_labels(full_inputs, prompt_inputs)

        return {
            "full_inputs": full_inputs,
            "labels": labels,
            "prompt_token_index": int(prompt_inputs["input_ids"].shape[-1]) - 1,
        }

    def _build_selection_records(self, samples: Sequence[Dict]) -> List[Dict[str, torch.Tensor | int]]:
        return [self._build_selection_record(sample) for sample in samples]

    def _build_avg_activation_tensor(self, cluster_centers: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        avg_tensor = torch.zeros(
            self.n_layers,
            self.n_heads,
            1,
            self.head_dim,
            dtype=torch.float32,
        )
        for assign_idx, (layer_idx, head_idx, _) in zip(assignment.tolist(), self.intervention_locations):
            avg_tensor[layer_idx, head_idx, 0] = cluster_centers[int(assign_idx), layer_idx, head_idx, 0]
        return avg_tensor

    def _gather_location_features(self, activations: torch.Tensor) -> torch.Tensor:
        if not self.intervention_locations:
            if activations.ndim == 4:
                return activations.reshape(-1).to(dtype=torch.float32)
            if activations.ndim == 5:
                return activations.reshape(activations.shape[0], -1).to(dtype=torch.float32)
            raise ValueError(f"Unsupported activation shape for STV feature gathering: {tuple(activations.shape)}")

        if activations.ndim == 4:
            gathered = [activations[layer_idx, head_idx, 0] for layer_idx, head_idx, _ in self.intervention_locations]
            return torch.cat(gathered, dim=-1).to(dtype=torch.float32)

        if activations.ndim == 5:
            gathered = [activations[:, layer_idx, head_idx, 0] for layer_idx, head_idx, _ in self.intervention_locations]
            return torch.cat(gathered, dim=-1).to(dtype=torch.float32)

        raise ValueError(f"Unsupported activation shape for STV feature gathering: {tuple(activations.shape)}")

    def _build_query_adaptive_avg_activations(self, sample: Dict) -> torch.Tensor:
        if self.cluster_centers is None:
            raise RuntimeError("STV query-adaptive mode requires fitted cluster_centers")

        if not self.intervention_locations:
            return torch.zeros(self.n_layers, self.n_heads, 1, self.head_dim, dtype=torch.float32)

        query_activations = self._capture_prompt_activations(sample, demos=[])
        query_features = F.normalize(self._gather_location_features(query_activations).unsqueeze(0), dim=-1)
        center_features = F.normalize(self._gather_location_features(self.cluster_centers), dim=-1)
        similarities = F.cosine_similarity(center_features, query_features, dim=-1)
        best_cluster_index = int(similarities.argmax().item())
        self.last_query_cluster_index = best_cluster_index

        assignment = torch.full(
            (len(self.intervention_locations),),
            fill_value=best_cluster_index,
            dtype=torch.long,
        )
        return self._build_avg_activation_tensor(self.cluster_centers, assignment)

    def _resolve_avg_activations(self, sample: Dict) -> torch.Tensor:
        if self.cluster_selection_mode == "query_adaptive":
            return self._build_query_adaptive_avg_activations(sample)

        if self.avg_activations is None:
            raise RuntimeError("STV is missing fitted avg_activations")
        return self.avg_activations

    def _evaluate_assignment_loss(
        self,
        records: Sequence[Dict[str, torch.Tensor | int]],
        cluster_centers: torch.Tensor,
        assignment: torch.Tensor,
    ) -> float:
        if not records:
            return float("inf")

        avg_activations = self._build_avg_activation_tensor(cluster_centers, assignment)
        losses = []

        for record in records:
            prompt_token_index = int(record["prompt_token_index"])
            full_inputs = record["full_inputs"]
            labels = record["labels"]

            self.hook_controller.enable_intervention(
                self.intervention_locations,
                avg_activations=avg_activations,
                token_index=prompt_token_index,
            )
            try:
                with torch.no_grad():
                    outputs = self.model.model(**full_inputs, labels=labels)
            finally:
                self.hook_controller.disable_intervention()

            losses.append(outputs.loss.detach().to(device="cpu", dtype=torch.float32))

        return float(torch.stack(losses).mean().item())

    def _optimize_cluster_choices(
        self,
        cluster_centers: torch.Tensor,
        train_records: Sequence[Dict[str, torch.Tensor | int]],
        eval_records: Sequence[Dict[str, torch.Tensor | int]],
        rng: random.Random,
    ) -> torch.Tensor:
        if not self.intervention_locations:
            return torch.zeros(self.n_layers, self.n_heads, 1, self.head_dim, dtype=torch.float32)

        num_choices = int(cluster_centers.shape[0])
        if num_choices == 1:
            zero_assignment = torch.zeros(len(self.intervention_locations), dtype=torch.long)
            return self._build_avg_activation_tensor(cluster_centers, zero_assignment)

        logits = torch.zeros(len(self.intervention_locations), num_choices, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=self.selection_lr)
        train_records = list(train_records)
        eval_records = list(eval_records) if eval_records else list(train_records)

        iterator = self._iter_with_progress(range(max(1, self.selection_epochs)), desc="STV cluster selection")
        for _ in iterator:
            record = [rng.choice(train_records)]
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            sample_log_probs = []
            sample_rewards = []
            for _ in range(max(1, self.selection_samples_per_epoch)):
                assignment = dist.sample()
                log_prob = dist.log_prob(assignment).sum()
                loss = self._evaluate_assignment_loss(record, cluster_centers, assignment)
                sample_log_probs.append(log_prob)
                sample_rewards.append(-loss)

            reward_tensor = torch.tensor(sample_rewards, dtype=torch.float32)
            if reward_tensor.numel() > 1:
                reward_tensor = (reward_tensor - reward_tensor.mean()) / reward_tensor.std().clamp_min(1e-6)

            policy_loss = -(
                torch.stack(sample_log_probs) * reward_tensor.to(device=logits.device, dtype=logits.dtype)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            optimizer.step()

        candidate_assignments = {
            tuple(torch.argmax(logits.detach(), dim=-1).tolist())
        }
        probs = torch.softmax(logits.detach(), dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        for _ in range(max(1, self.final_selection_trials)):
            candidate_assignments.add(tuple(dist.sample().tolist()))

        best_loss = float("inf")
        best_assignment = None
        for candidate in candidate_assignments:
            assignment = torch.tensor(candidate, dtype=torch.long)
            loss = self._evaluate_assignment_loss(eval_records, cluster_centers, assignment)
            if loss < best_loss:
                best_loss = loss
                best_assignment = assignment

        if best_assignment is None:
            raise RuntimeError("STV failed to choose a final cluster assignment")
        return self._build_avg_activation_tensor(cluster_centers, best_assignment)

    def fit(self, train_data: Sequence[Dict]) -> None:
        if len(train_data) == 0:
            raise ValueError("STV fit requires non-empty training data")

        self.model.model.eval()
        self._build_label_index(train_data)
        rng = random.Random(self.support_seed)

        activation_indices = self._sample_subset_indices(len(train_data), self.num_examples, rng)
        if self.head_selection_mode == "sav_accuracy":
            self.avg_diff = None
            self.intervention_locations = self._select_topk_locations_sav_accuracy(train_data)
        else:
            self.avg_diff = self._estimate_avg_diff(train_data, activation_indices, rng)
            self.intervention_locations = self._select_topk_locations(self.avg_diff)

        contextual_samples = self._collect_cluster_samples(train_data, activation_indices, rng)
        self.cluster_centers = self._build_cluster_centers(contextual_samples)

        if self.cluster_selection_mode == "query_adaptive":
            self.avg_activations = None
            self.last_query_cluster_index = None
            self._fitted = True
            return

        selection_indices = self._sample_subset_indices(
            len(train_data),
            self.selection_train_size + self.selection_eval_size,
            rng,
        )
        train_cutoff = min(len(selection_indices), self.selection_train_size)
        selection_train_items = [train_data[idx] for idx in selection_indices[:train_cutoff]]
        selection_eval_items = [train_data[idx] for idx in selection_indices[train_cutoff:]]
        if not selection_eval_items:
            selection_eval_items = list(selection_train_items)

        selection_train_records = self._build_selection_records(selection_train_items)
        selection_eval_records = self._build_selection_records(selection_eval_items)

        self.avg_activations = self._optimize_cluster_choices(
            self.cluster_centers,
            train_records=selection_train_records,
            eval_records=selection_eval_records,
            rng=rng,
        )
        self._fitted = True

    def _generate_with_intervention(self, sample: Dict) -> str:
        avg_activations = self._resolve_avg_activations(sample)

        conversation, flat_images = self._build_conversation([], sample, include_answer=False)
        inputs = self._prepare_inputs(conversation, flat_images, add_generation_prompt=True)
        prompt_len = int(inputs["input_ids"].shape[-1])

        self.hook_controller.enable_intervention(
            self.intervention_locations,
            avg_activations=avg_activations,
            token_index=prompt_len - 1,
        )
        try:
            generated_ids = self.model.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                use_cache=False,
            )
        finally:
            self.hook_controller.disable_intervention()

        generated_ids = generated_ids[:, prompt_len:]
        outputs = self.model.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs[0] if outputs else ""

    def predict(self, sample: Dict) -> str:
        if not self._fitted:
            raise RuntimeError("STV method is not fitted. Call fit(train_data) first.")

        raw = self._generate_with_intervention(sample)
        return self._match_label(raw)
