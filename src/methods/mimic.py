from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..data import build_prompt
from .base import MethodBase


@dataclass(frozen=True)
class _LayerMeta:
    name: str
    num_heads: int
    head_dim: int
    num_key_value_heads: int
    num_key_value_groups: int
    device: torch.device


class _MimICShiftAdapter(nn.Module):
    def __init__(
        self,
        model_wrapper,
        attn_module_names: Sequence[str],
        shift_init_std: float = 1e-3,
        score_chunk_size: int = 16,
    ):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.active = False
        self.score_chunk_size = max(1, int(score_chunk_size))

        named_modules = dict(self.model_wrapper.model.named_modules())
        self.layer_metas: List[_LayerMeta] = []
        self.layer_index_by_name: Dict[str, int] = {}
        self.q_proj_to_layer: Dict[str, int] = {}
        self.k_proj_to_layer: Dict[str, int] = {}
        self.o_proj_to_layer: Dict[str, int] = {}
        self.attn_modules: Dict[int, nn.Module] = {}

        self.shift = nn.ParameterList()
        self.gate_weight = nn.ParameterList()
        self.gate_bias = nn.ParameterList()

        for layer_idx, name in enumerate(attn_module_names):
            module = named_modules.get(name)
            if module is None:
                raise ValueError(f"Attention module '{name}' not found")
            if not all(hasattr(module, attr) for attr in ["q_proj", "k_proj", "o_proj"]):
                raise ValueError(
                    f"Attention module '{name}' is missing q_proj/k_proj/o_proj and cannot be used by MimIC"
                )

            meta = self._infer_layer_meta(name, module)
            self.layer_metas.append(meta)
            self.layer_index_by_name[name] = layer_idx
            self.q_proj_to_layer[f"{name}.q_proj"] = layer_idx
            self.k_proj_to_layer[f"{name}.k_proj"] = layer_idx
            self.o_proj_to_layer[f"{name}.o_proj"] = layer_idx
            self.attn_modules[layer_idx] = module

            param_device = meta.device
            self.shift.append(
                nn.Parameter(
                    torch.empty(meta.num_heads, meta.head_dim, device=param_device, dtype=torch.float32).normal_(
                        mean=0.0,
                        std=shift_init_std,
                    )
                )
            )
            self.gate_weight.append(
                nn.Parameter(
                    torch.empty(meta.num_heads, meta.head_dim, device=param_device, dtype=torch.float32).normal_(
                        mean=0.0,
                        std=0.02,
                    )
                )
            )
            self.gate_bias.append(
                nn.Parameter(torch.zeros(meta.num_heads, device=param_device, dtype=torch.float32))
            )

        self._q_cache: Dict[int, torch.Tensor] = {}
        self._k_cache: Dict[int, torch.Tensor] = {}
        self._handles = self._register_runtime_hooks()

    def _infer_layer_meta(self, name: str, module: nn.Module) -> _LayerMeta:
        config = getattr(module, "config", getattr(self.model_wrapper.model, "config", None))

        num_heads = getattr(module, "num_heads", None)
        if num_heads is None and config is not None:
            num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is None:
            head_dim = getattr(module, "head_dim", None)
            if head_dim is None:
                raise RuntimeError(f"Unable to infer num_heads for attention module '{name}'")
            num_heads = module.q_proj.out_features // int(head_dim)
        num_heads = int(num_heads)

        head_dim = getattr(module, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(module, "hidden_size", None)
            if hidden_size is None and config is not None:
                hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = module.q_proj.out_features
            head_dim = int(hidden_size) // num_heads
        head_dim = int(head_dim)

        num_key_value_heads = getattr(module, "num_key_value_heads", None)
        if num_key_value_heads is None and config is not None:
            num_key_value_heads = getattr(config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        num_key_value_heads = int(num_key_value_heads)

        num_key_value_groups = getattr(module, "num_key_value_groups", None)
        if num_key_value_groups is None:
            num_key_value_groups = max(1, num_heads // num_key_value_heads)
        num_key_value_groups = int(num_key_value_groups)

        return _LayerMeta(
            name=name,
            num_heads=num_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            num_key_value_groups=num_key_value_groups,
            device=module.q_proj.weight.device,
        )

    def _register_runtime_hooks(self) -> List:
        handles = []
        handles.extend(self._ensure_list(self.model_wrapper.register_forward_hook(list(self.q_proj_to_layer), self._q_hook)))
        handles.extend(self._ensure_list(self.model_wrapper.register_forward_hook(list(self.k_proj_to_layer), self._k_hook)))
        handles.extend(
            self._ensure_list(
                self.model_wrapper.register_forward_pre_hook(list(self.o_proj_to_layer), self._o_proj_pre_hook)
            )
        )
        return handles

    @staticmethod
    def _ensure_list(handles):
        return handles if isinstance(handles, list) else [handles]

    def clear_runtime_cache(self) -> None:
        self._q_cache.clear()
        self._k_cache.clear()

    def enable(self) -> None:
        self.clear_runtime_cache()
        self.active = True

    def disable(self) -> None:
        self.active = False
        self.clear_runtime_cache()

    def _reshape_q(self, layer_idx: int, tensor: torch.Tensor) -> torch.Tensor:
        meta = self.layer_metas[layer_idx]
        reshaped = tensor.view(tensor.shape[0], tensor.shape[1], meta.num_heads, meta.head_dim)
        attn_module = self.attn_modules[layer_idx]
        q_norm = getattr(attn_module, "q_norm", None)
        if q_norm is not None:
            reshaped = q_norm(reshaped)
        return reshaped

    def _reshape_k(self, layer_idx: int, tensor: torch.Tensor) -> torch.Tensor:
        meta = self.layer_metas[layer_idx]
        reshaped = tensor.view(tensor.shape[0], tensor.shape[1], meta.num_key_value_heads, meta.head_dim)
        attn_module = self.attn_modules[layer_idx]
        k_norm = getattr(attn_module, "k_norm", None)
        if k_norm is not None:
            reshaped = k_norm(reshaped)
        if meta.num_key_value_groups > 1:
            reshaped = reshaped.repeat_interleave(meta.num_key_value_groups, dim=2)
        return reshaped

    def _q_hook(self, module, args, output, module_name=None):
        if not self.active or not torch.is_tensor(output) or module_name not in self.q_proj_to_layer:
            return None
        layer_idx = self.q_proj_to_layer[module_name]
        self._q_cache[layer_idx] = self._reshape_q(layer_idx, output)
        return None

    def _k_hook(self, module, args, output, module_name=None):
        if not self.active or not torch.is_tensor(output) or module_name not in self.k_proj_to_layer:
            return None
        layer_idx = self.k_proj_to_layer[module_name]
        self._k_cache[layer_idx] = self._reshape_k(layer_idx, output)
        return None

    def _o_proj_pre_hook(self, module, args, module_name=None):
        if not self.active or not args or module_name not in self.o_proj_to_layer:
            return None

        attn_input = args[0]
        if not torch.is_tensor(attn_input) or attn_input.ndim != 3:
            return None

        layer_idx = self.o_proj_to_layer[module_name]
        if layer_idx not in self._q_cache or layer_idx not in self._k_cache:
            return None

        meta = self.layer_metas[layer_idx]
        q_states = self._q_cache[layer_idx].to(torch.float32)
        k_states = self._k_cache[layer_idx].to(torch.float32)
        attn_output = attn_input.view(attn_input.shape[0], attn_input.shape[1], meta.num_heads, meta.head_dim).to(
            torch.float32
        )

        scale = max(meta.head_dim**0.5, 1.0)
        source_len = int(k_states.shape[1])
        chunk_size = max(1, min(self.score_chunk_size, source_len))
        while True:
            try:
                log_z2 = None
                for start in range(0, source_len, chunk_size):
                    stop = min(start + chunk_size, source_len)
                    score_chunk = torch.einsum("btnd,bsnd->btns", q_states, k_states[:, start:stop, :, :]) / scale
                    chunk_log_z2 = torch.logsumexp(score_chunk, dim=-1)
                    log_z2 = chunk_log_z2 if log_z2 is None else torch.logaddexp(log_z2, chunk_log_z2)
                break
            except torch.OutOfMemoryError:
                if chunk_size == 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    q_cpu = q_states.detach().to(device="cpu")
                    k_cpu = k_states.detach().to(device="cpu")
                    log_z2 = None
                    cpu_chunk_size = max(1, min(self.score_chunk_size, source_len))
                    for start in range(0, source_len, cpu_chunk_size):
                        stop = min(start + cpu_chunk_size, source_len)
                        score_chunk = torch.einsum("btnd,bsnd->btns", q_cpu, k_cpu[:, start:stop, :, :]) / scale
                        chunk_log_z2 = torch.logsumexp(score_chunk, dim=-1)
                        log_z2 = chunk_log_z2 if log_z2 is None else torch.logaddexp(log_z2, chunk_log_z2)
                    log_z2 = log_z2.to(device=q_states.device, dtype=q_states.dtype)
                    break
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)

        if log_z2 is None:
            log_z2 = torch.zeros(
                q_states.shape[0],
                q_states.shape[1],
                q_states.shape[2],
                device=q_states.device,
                dtype=q_states.dtype,
            )

        gate_weight = self.gate_weight[layer_idx]
        gate_bias = self.gate_bias[layer_idx]
        log_z1 = torch.einsum("btnd,nd->btn", q_states, gate_weight) + gate_bias
        mu = torch.exp(log_z1 - torch.logaddexp(log_z1, log_z2)).unsqueeze(-1)

        shift = self.shift[layer_idx].unsqueeze(0).unsqueeze(0)
        shifted = attn_output + mu * shift

        original_norm = attn_output.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        shifted_norm = shifted.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        shifted = shifted / shifted_norm * original_norm
        shifted = shifted.reshape(attn_input.shape).to(dtype=attn_input.dtype)
        return (shifted,)


class MimICMethod(MethodBase):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        num_shots: int = 4,
        epochs: int = 1,
        max_steps: int = 0,
        learning_rate: float = 5e-3,
        weight_decay: float = 0.0,
        lm_loss_weight: float = 1.0,
        align_loss_weight: float = 1.0,
        support_strategy: str = "balanced",
        support_seed: int = 42,
        shift_init_std: float = 1e-3,
        score_chunk_size: int = 16,
        alignment_window_tokens: int = 64,
        max_adapt_layers: int = 4,
        gradient_clip_norm: float = 1.0,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 0.0,
        progress_bar: bool = True,
    ):
        super().__init__(model=model, dataset_name=dataset_name, label_space=label_space)

        self.num_shots = int(num_shots)
        self.epochs = int(epochs)
        self.max_steps = int(max_steps)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.lm_loss_weight = float(lm_loss_weight)
        self.align_loss_weight = float(align_loss_weight)
        self.support_strategy = str(support_strategy).strip().lower()
        self.support_seed = int(support_seed)
        self.score_chunk_size = int(score_chunk_size)
        self.alignment_window_tokens = int(alignment_window_tokens)
        self.max_adapt_layers = int(max_adapt_layers)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.progress_bar = bool(progress_bar)

        if self.num_shots <= 0:
            raise ValueError("MimIC num_shots must be positive")
        if self.epochs <= 0:
            raise ValueError("MimIC epochs must be positive")
        if self.lm_loss_weight <= 0.0 and self.align_loss_weight <= 0.0:
            raise ValueError("At least one of lm_loss_weight or align_loss_weight must be positive")

        self.model_family = self._infer_model_family()
        self.attn_module_names = self._infer_attention_module_names()
        for param in self.model.model.parameters():
            param.requires_grad_(False)
        if hasattr(self.model.model, "config") and hasattr(self.model.model.config, "use_cache"):
            self.model.model.config.use_cache = False
        self.adapter = _MimICShiftAdapter(
            model_wrapper=self.model,
            attn_module_names=self.attn_module_names,
            shift_init_std=float(shift_init_std),
            score_chunk_size=self.score_chunk_size,
        )

        self._label_to_indices: Dict[str, List[int]] = {}
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
        raise ValueError(
            "MimIC baseline currently supports qwen2_vl, qwen3_vl, and idefics3 style models only"
        )

    def _infer_attention_module_names(self) -> List[str]:
        names = []
        for name, module in self.model.model.named_modules():
            if not name.endswith("self_attn"):
                continue
            if all(hasattr(module, attr) for attr in ["q_proj", "k_proj", "o_proj"]):
                names.append(name)
        names = self._filter_text_backbone_module_names(names)

        if not names:
            raise RuntimeError("Could not find decoder self-attention modules for MimIC extraction")

        def sort_key(module_name: str):
            nums = [int(v) for v in re.findall(r"\d+", module_name)]
            return tuple(nums) if nums else (10**9,)

        names = sorted(names, key=sort_key)
        if self.max_adapt_layers > 0 and len(names) > self.max_adapt_layers:
            names = names[-self.max_adapt_layers :]
        return names

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

    def _build_user_message(
        self,
        sample: Dict,
    ) -> tuple[Dict, List[Image.Image]]:
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

    def _record_attention_outputs(self, inputs, *, labels=None, shift_active: bool, no_grad: bool, capture_tail_tokens: int):
        recorded: Dict[str, torch.Tensor] = {}

        def hook_fn(module, args, output, module_name=None):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden_states):
                if capture_tail_tokens > 0 and hidden_states.ndim >= 3:
                    hidden_states = hidden_states[:, -capture_tail_tokens:, :]
                recorded[module_name] = hidden_states.detach().to(device="cpu") if no_grad else hidden_states

        handles = self.model.register_forward_hook(self.attn_module_names, hook_fn)
        handle_list = handles if isinstance(handles, list) else [handles]

        self.adapter.enable() if shift_active else self.adapter.disable()
        try:
            context = torch.no_grad() if no_grad else torch.enable_grad()
            with context:
                outputs = (
                    self.model.model(**inputs, labels=labels, use_cache=False)
                    if labels is not None
                    else self.model.model(**inputs, use_cache=False)
                )
        finally:
            self.adapter.disable()
            for handle in handle_list:
                handle.remove()

        missing = [name for name in self.attn_module_names if name not in recorded]
        if missing:
            raise RuntimeError(f"Failed to capture MimIC attention outputs for modules: {', '.join(missing[:3])}")
        ordered_outputs = [recorded[name] for name in self.attn_module_names]
        return outputs, ordered_outputs

    def _compute_alignment_loss(
        self,
        shifted_outputs: Sequence[torch.Tensor],
        full_outputs: Sequence[torch.Tensor],
        reference_device: torch.device,
    ) -> torch.Tensor:
        losses = []
        for shifted, full in zip(shifted_outputs, full_outputs):
            shared_len = min(int(shifted.shape[1]), int(full.shape[1]))
            if shared_len <= 0:
                continue
            shifted_slice = shifted[:, -shared_len:, :].to(torch.float32)
            full_slice = full[:, -shared_len:, :].to(device=shifted_slice.device, dtype=torch.float32)
            losses.append(F.mse_loss(shifted_slice, full_slice, reduction="mean").to(reference_device))

        if not losses:
            return torch.zeros((), device=reference_device, dtype=torch.float32)
        return torch.stack(losses).mean()

    def _build_label_index(self, train_data: Sequence[Dict]) -> None:
        self._label_to_indices = {}
        for idx, item in enumerate(train_data):
            self._label_to_indices.setdefault(str(item["label"]), []).append(idx)

    def _sample_demo_indices(
        self,
        train_data: Sequence[Dict],
        query_index: int,
        rng: random.Random,
    ) -> List[int]:
        candidate_indices = [idx for idx in range(len(train_data)) if idx != query_index]
        if not candidate_indices:
            candidate_indices = [query_index]

        if self.support_strategy == "random":
            if len(candidate_indices) >= self.num_shots:
                return rng.sample(candidate_indices, self.num_shots)
            return [rng.choice(candidate_indices) for _ in range(self.num_shots)]

        if self.support_strategy != "balanced":
            raise ValueError("MimIC support_strategy must be 'balanced' or 'random'")

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

    def fit(self, train_data: Sequence[Dict]) -> None:
        if len(train_data) == 0:
            raise ValueError("MimIC fit requires non-empty training data")

        self._build_label_index(train_data)
        self.model.model.eval()
        self.adapter.train()

        optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        rng = random.Random(self.support_seed)

        step_budget = self.max_steps if self.max_steps > 0 else self.epochs * len(train_data)
        step_count = 0
        epoch_iter = range(self.epochs)

        for epoch in self._iter_with_progress(epoch_iter, desc="MimIC epochs"):
            if step_count >= step_budget:
                break

            indices = list(range(len(train_data)))
            rng.shuffle(indices)
            sample_iter = self._iter_with_progress(indices, desc=f"MimIC train epoch {epoch + 1}")

            for query_index in sample_iter:
                if step_count >= step_budget:
                    break

                query_item = train_data[query_index]
                demo_indices = self._sample_demo_indices(train_data, query_index, rng)
                demos = [train_data[idx] for idx in demo_indices]

                full_conversation, full_images = self._build_conversation(demos, query_item, include_answer=True)
                query_conversation, query_images = self._build_conversation([], query_item, include_answer=True)
                query_prompt_conversation, query_prompt_images = self._build_conversation([], query_item, include_answer=False)

                full_inputs = self._prepare_inputs(full_conversation, full_images, add_generation_prompt=False)
                query_inputs = self._prepare_inputs(query_conversation, query_images, add_generation_prompt=False)
                query_prompt_inputs = self._prepare_inputs(
                    query_prompt_conversation,
                    query_prompt_images,
                    add_generation_prompt=True,
                )
                labels = self._build_answer_only_labels(query_inputs, query_prompt_inputs)

                optimizer.zero_grad(set_to_none=True)

                capture_tail_tokens = self.alignment_window_tokens if self.align_loss_weight > 0.0 else 0
                if self.align_loss_weight > 0.0:
                    _, full_attn_outputs = self._record_attention_outputs(
                        full_inputs,
                        shift_active=False,
                        no_grad=True,
                        capture_tail_tokens=capture_tail_tokens,
                    )
                else:
                    full_attn_outputs = []
                query_outputs, shifted_attn_outputs = self._record_attention_outputs(
                    query_inputs,
                    labels=labels,
                    shift_active=True,
                    no_grad=False,
                    capture_tail_tokens=capture_tail_tokens,
                )

                loss_terms: List[torch.Tensor] = []
                reference_device = query_outputs.logits.device

                if self.lm_loss_weight > 0.0:
                    ce_loss = query_outputs.loss.to(reference_device)
                    loss_terms.append(self.lm_loss_weight * ce_loss)

                if self.align_loss_weight > 0.0:
                    align_loss = self._compute_alignment_loss(
                        shifted_attn_outputs,
                        full_attn_outputs,
                        reference_device=reference_device,
                    )
                    loss_terms.append(self.align_loss_weight * align_loss)

                loss = torch.stack([term.to(reference_device) for term in loss_terms]).sum()
                loss.backward()

                if self.gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.gradient_clip_norm)
                optimizer.step()
                step_count += 1

        self.adapter.eval()
        self._fitted = True

    def _generate_from_conversation(self, conversation: Sequence[Dict], flat_images: Sequence[Image.Image]) -> str:
        inputs = self._prepare_inputs(conversation, flat_images, add_generation_prompt=True)
        prompt_len = int(inputs["input_ids"].shape[-1])

        self.adapter.enable()
        try:
            generated_ids = self.model.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                use_cache=False,
            )
        finally:
            self.adapter.disable()

        generated_ids = generated_ids[:, prompt_len:]
        outputs = self.model.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs[0] if outputs else ""

    def predict(self, sample: Dict) -> str:
        if not self._fitted:
            raise RuntimeError("MimIC method is not fitted. Call fit(train_data) first.")

        conversation, flat_images = self._build_conversation([], sample, include_answer=False)
        raw = self._generate_from_conversation(conversation, flat_images)
        return self._match_label(raw)
