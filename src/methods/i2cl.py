from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..data import build_prompt
from .base import MethodBase
from .write_failure import WriteFailureRecorder


@dataclass(frozen=True)
class _LayerSpec:
    attn_name: str
    mlp_name: str
    hidden_name: str
    hidden_size: int
    device: torch.device


class _I2CLInjector(nn.Module):
    def __init__(
        self,
        model_wrapper,
        layer_specs: Sequence[_LayerSpec],
        inject_layers: Sequence[int],
        modules: Sequence[str],
        *,
        inject_method: str,
        inject_pos: str,
        init_value,
        add_noise: bool,
        noise_scale: float,
    ):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.layer_specs = list(layer_specs)
        self.inject_layers = [int(layer_idx) for layer_idx in inject_layers]
        self.modules = [str(module_name).strip().lower() for module_name in modules]
        self.inject_method = str(inject_method).strip().lower()
        self.inject_pos = str(inject_pos).strip().lower()
        self.add_noise = bool(add_noise)
        self.noise_scale = float(noise_scale)

        if self.inject_method not in {"add", "linear", "balance"}:
            raise ValueError("I2CL inject_method must be one of: add, linear, balance")
        if self.inject_pos not in {"all", "last", "first", "random"}:
            raise ValueError("I2CL inject_pos must be one of: all, last, first, random")
        if not self.modules:
            raise ValueError("I2CL modules must be non-empty")
        if any(module_name not in {"attn", "mlp", "hidden"} for module_name in self.modules):
            raise ValueError("I2CL modules must be chosen from: attn, mlp, hidden")

        self.active = False
        self.train_mode = False
        self.context_vector_dict: Dict[int, Dict[str, torch.Tensor]] = {}

        self.module_name_to_slot: Dict[str, int] = {}
        self.slot_to_layer: List[int] = []
        self.slot_to_module: List[str] = []
        self.strength_params = nn.ParameterList()

        init_values = self._normalize_init_value(init_value)
        param_dim = 1 if self.inject_method == "add" else 2
        if len(init_values) != param_dim:
            raise ValueError(
                f"I2CL inject_method '{self.inject_method}' expects init_value length {param_dim}, got {len(init_values)}"
            )

        for layer_idx in self.inject_layers:
            spec = self.layer_specs[layer_idx]
            for module_name in self.modules:
                module_path = getattr(spec, f"{module_name}_name")
                slot_idx = len(self.slot_to_layer)
                self.module_name_to_slot[module_path] = slot_idx
                self.slot_to_layer.append(layer_idx)
                self.slot_to_module.append(module_name)
                self.strength_params.append(
                    nn.Parameter(torch.tensor(init_values, device=spec.device, dtype=torch.float32))
                )

        handles = self.model_wrapper.register_forward_hook(list(self.module_name_to_slot), self._inject_hook)
        self._handles = handles if isinstance(handles, list) else [handles]

    def _normalize_init_value(self, init_value) -> List[float]:
        if isinstance(init_value, (int, float)):
            return [float(init_value)]
        if isinstance(init_value, Sequence):
            return [float(value) for value in init_value]
        raise TypeError(f"Unsupported I2CL init_value type: {type(init_value).__name__}")

    def enable(self, context_vector_dict: Dict[int, Dict[str, torch.Tensor]], *, train_mode: bool) -> None:
        stored: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx, module_dict in context_vector_dict.items():
            stored[int(layer_idx)] = {
                str(module_name): tensor.detach().to(device="cpu", dtype=torch.float32)
                for module_name, tensor in module_dict.items()
            }
        self.context_vector_dict = stored
        self.train_mode = bool(train_mode)
        self.active = True

    def disable(self) -> None:
        self.active = False
        self.train_mode = False
        self.context_vector_dict = {}

    def _resolve_token_index(self, seq_len: int) -> int:
        if self.inject_pos == "first":
            return 0
        if self.inject_pos == "last":
            return seq_len - 1
        if self.inject_pos == "random":
            return random.randint(0, max(0, seq_len - 1))
        raise ValueError(f"I2CL token position resolution does not support '{self.inject_pos}'")

    def _inject_tensor(
        self,
        output: torch.Tensor,
        context_vector: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        base = output.to(torch.float32)
        context = context_vector.to(device=base.device, dtype=torch.float32).view(1, 1, -1)
        context = context.expand(base.shape[0], base.shape[1], -1)

        if self.inject_method == "add":
            modified = base + F.relu(strength[0]) * context
        elif self.inject_method == "linear":
            if self.inject_pos == "all":
                modified = strength[1] * base + strength[0] * context
            else:
                token_idx = self._resolve_token_index(base.shape[1])
                updated_token = (
                    strength[1] * base[:, token_idx : token_idx + 1, :]
                    + strength[0] * context[:, token_idx : token_idx + 1, :]
                )
                modified = torch.cat(
                    [
                        base[:, :token_idx, :],
                        updated_token,
                        base[:, token_idx + 1 :, :],
                    ],
                    dim=1,
                )
        elif self.inject_method == "balance":
            modified = ((1.0 - strength[0]) * base + strength[0] * context) * strength[1]
        else:
            raise ValueError(f"Unsupported I2CL inject_method '{self.inject_method}'")

        if self.add_noise and self.train_mode:
            output_norm = torch.norm(modified, p=2, dim=-1, keepdim=True).detach()
            noise = torch.randn_like(modified).detach()
            modified = modified + noise * output_norm * self.noise_scale

        return modified.to(dtype=output.dtype)

    def _inject_hook(self, module, args, output, module_name=None):
        if not self.active or module_name not in self.module_name_to_slot:
            return None

        tensor = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(tensor) or tensor.ndim != 3:
            return None

        slot_idx = self.module_name_to_slot[module_name]
        layer_idx = self.slot_to_layer[slot_idx]
        module_key = self.slot_to_module[slot_idx]
        if layer_idx not in self.context_vector_dict or module_key not in self.context_vector_dict[layer_idx]:
            return None

        context_vector = self.context_vector_dict[layer_idx][module_key]
        strength = self.strength_params[slot_idx].to(device=tensor.device, dtype=torch.float32)
        modified = self._inject_tensor(tensor, context_vector, strength)

        if isinstance(output, tuple):
            return (modified,) + tuple(output[1:])
        return modified


class I2CLMethod(MethodBase):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        num_shots: int = 8,
        support_strategy: str = "balanced",
        support_seed: int = 42,
        layer_selection="all",
        modules: Sequence[str] = ("mlp", "attn"),
        tok_pos: str = "last",
        inject_method: str = "linear",
        inject_pos: str = "all",
        init_value: Sequence[float] = (0.1, 1.0),
        context_init: str = "context",
        post_fuse_method: str = "mean",
        epochs: int = 30,
        max_steps: int = 0,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-3,
        add_noise: bool = True,
        noise_scale: float = 1e-3,
        gradient_clip_norm: float = 1.0,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 0.0,
        progress_bar: bool = True,
        write_failure_dump_dir: str | None = None,
        write_failure_max_samples: int = 0,
        write_failure_heatmap_samples: int = 2,
        write_failure_query_last_k: int = 3,
        write_failure_answer_source: str = "label",
    ):
        super().__init__(model=model, dataset_name=dataset_name, label_space=label_space)

        self.num_shots = int(num_shots)
        self.support_strategy = str(support_strategy).strip().lower()
        self.support_seed = int(support_seed)
        self.layer_selection = layer_selection
        self.modules = self._normalize_modules(modules)
        self.tok_pos = str(tok_pos).strip().lower()
        self.inject_method = str(inject_method).strip().lower()
        self.inject_pos = str(inject_pos).strip().lower()
        self.init_value = self._normalize_init_value(init_value)
        self.context_init = str(context_init).strip().lower()
        self.post_fuse_method = str(post_fuse_method).strip().lower()
        self.epochs = int(epochs)
        self.max_steps = int(max_steps)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.add_noise = bool(add_noise)
        self.noise_scale = float(noise_scale)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.progress_bar = bool(progress_bar)

        if self.num_shots <= 0:
            raise ValueError("I2CL num_shots must be positive")
        if self.support_strategy not in {"balanced", "random"}:
            raise ValueError("I2CL support_strategy must be 'balanced' or 'random'")
        if self.tok_pos not in {"last", "first", "random"}:
            raise ValueError("I2CL tok_pos must be one of: last, first, random")
        if self.context_init not in {"context", "noise"}:
            raise ValueError("I2CL context_init must be 'context' or 'noise'")
        if self.post_fuse_method not in {"mean", "pca"}:
            raise ValueError("I2CL post_fuse_method must be 'mean' or 'pca'")
        if self.epochs <= 0:
            raise ValueError("I2CL epochs must be positive")

        self.model_family = self._infer_model_family()
        self.layer_specs = self._infer_layer_specs()
        self.inject_layers = self._resolve_inject_layers(self.layer_selection)
        self.injector = _I2CLInjector(
            model_wrapper=self.model,
            layer_specs=self.layer_specs,
            inject_layers=self.inject_layers,
            modules=self.modules,
            inject_method=self.inject_method,
            inject_pos=self.inject_pos,
            init_value=self.init_value,
            add_noise=self.add_noise,
            noise_scale=self.noise_scale,
        )

        self._label_to_indices: Dict[str, List[int]] = {}
        self.demo_indices: List[int] = []
        self.context_vector_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self._fitted = False
        self._predict_calls = 0
        self.write_failure_recorder = WriteFailureRecorder(
            method_name="i2cl",
            dump_dir=write_failure_dump_dir,
            max_samples=write_failure_max_samples,
            heatmap_samples=write_failure_heatmap_samples,
            query_last_k=write_failure_query_last_k,
            answer_source=write_failure_answer_source,
        )

    @staticmethod
    def _normalize_modules(modules) -> List[str]:
        if isinstance(modules, str):
            return [modules.strip().lower()]
        return [str(module_name).strip().lower() for module_name in modules]

    @staticmethod
    def _normalize_init_value(init_value) -> List[float]:
        if isinstance(init_value, (int, float)):
            return [float(init_value)]
        return [float(value) for value in init_value]

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
        raise ValueError("I2CL baseline currently supports qwen2_vl, qwen3_vl, and idefics3 style models only")

    def _infer_layer_specs(self) -> List[_LayerSpec]:
        named_modules = dict(self.model.model.named_modules())
        attn_names = []
        for name, module in named_modules.items():
            if not name.endswith("self_attn"):
                continue
            if hasattr(module, "o_proj") or hasattr(module, "out_proj"):
                attn_names.append(name)
        attn_names = self._filter_text_backbone_module_names(attn_names)

        if not attn_names:
            raise RuntimeError("Could not find decoder self-attention modules for I2CL")

        def sort_key(module_name: str):
            nums = [int(v) for v in re.findall(r"\d+", module_name)]
            return tuple(nums) if nums else (10**9,)

        layer_specs: List[_LayerSpec] = []
        for attn_name in sorted(attn_names, key=sort_key):
            layer_prefix = attn_name.rsplit(".", 1)[0]
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
                raise RuntimeError(f"Could not uniquely determine the MLP module for layer '{layer_prefix}'")

            attn_module = named_modules[attn_name]
            layer_specs.append(
                _LayerSpec(
                    attn_name=attn_name,
                    mlp_name=mlp_name,
                    hidden_name=layer_prefix,
                    hidden_size=self._infer_hidden_size(attn_module),
                    device=self._infer_module_device(attn_module),
                )
            )
        return layer_specs

    def _infer_hidden_size(self, attn_module) -> int:
        for proj_name in ["o_proj", "out_proj", "q_proj"]:
            proj = getattr(attn_module, proj_name, None)
            if proj is not None:
                for attr_name in ["out_features", "in_features"]:
                    if hasattr(proj, attr_name):
                        return int(getattr(proj, attr_name))

        hidden_size = getattr(attn_module, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)

        config = getattr(attn_module, "config", getattr(self.model.model, "config", None))
        text_config = getattr(config, "text_config", config)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            return int(text_config.hidden_size)

        raise RuntimeError("Unable to infer hidden size for I2CL")

    @staticmethod
    def _infer_module_device(module) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
        return torch.device("cpu")

    def _resolve_inject_layers(self, layer_selection) -> List[int]:
        total_layers = len(self.layer_specs)
        if isinstance(layer_selection, str):
            key = layer_selection.strip().lower()
            if key == "all":
                layers = list(range(total_layers))
            elif key == "early":
                layers = list(range(max(1, total_layers // 3)))
            elif key == "mid":
                start = total_layers // 3
                end = max(start + 1, (total_layers * 2) // 3)
                layers = list(range(start, min(end, total_layers)))
            elif key == "late":
                start = (total_layers * 2) // 3
                layers = list(range(start, total_layers))
            else:
                raise ValueError("I2CL layer_selection must be one of: all, early, mid, late, or a list of indices")
        elif isinstance(layer_selection, Sequence):
            layers = [int(layer_idx) for layer_idx in layer_selection]
        else:
            raise TypeError(f"Unsupported I2CL layer_selection type: {type(layer_selection).__name__}")

        if not layers:
            raise ValueError("I2CL resolved zero injection layers")
        if min(layers) < 0 or max(layers) >= total_layers:
            raise ValueError(f"I2CL layer_selection must be within [0, {total_layers - 1}]")
        return sorted(dict.fromkeys(layers))

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
        answer_text: str | None = None,
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
            answer_value = str(query["label"]) if answer_text is None else str(answer_text)
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": answer_value}]})

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

    def _sample_demo_indices(self, train_data: Sequence[Dict], rng: random.Random) -> List[int]:
        candidate_indices = list(range(len(train_data)))
        if not candidate_indices:
            raise ValueError("I2CL demo sampling requires non-empty train data")

        if self.support_strategy == "random":
            if len(candidate_indices) >= self.num_shots:
                return rng.sample(candidate_indices, self.num_shots)
            return [rng.choice(candidate_indices) for _ in range(self.num_shots)]

        labels = list(self._label_to_indices.keys())
        rng.shuffle(labels)
        selected: List[int] = []
        used: set[int] = set()

        while len(selected) < min(self.num_shots, len(candidate_indices)):
            added = False
            for label in labels:
                pool = [idx for idx in self._label_to_indices[label] if idx not in used]
                if not pool:
                    continue
                chosen = rng.choice(pool)
                selected.append(chosen)
                used.add(chosen)
                added = True
                if len(selected) >= min(self.num_shots, len(candidate_indices)):
                    break
            if not added:
                break

        while len(selected) < self.num_shots:
            selected.append(rng.choice(candidate_indices))

        return selected

    def _get_slot_mapping(self) -> Dict[str, tuple[int, str]]:
        mapping: Dict[str, tuple[int, str]] = {}
        for module_path, slot_idx in self.injector.module_name_to_slot.items():
            mapping[module_path] = (
                self.injector.slot_to_layer[slot_idx],
                self.injector.slot_to_module[slot_idx],
            )
        return mapping

    def _select_token_feature(self, tensor: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if tensor.ndim != 3 or int(tensor.shape[0]) != 1:
            raise ValueError(f"I2CL expected hidden states with shape [1, T, D], got {tuple(tensor.shape)}")

        seq_len = int(tensor.shape[1])
        if self.tok_pos == "last":
            token_idx = seq_len - 1
        elif self.tok_pos == "first":
            token_idx = 0
        elif self.tok_pos == "random":
            token_idx = rng.randint(0, max(0, seq_len - 1))
        else:
            raise ValueError(f"Unsupported I2CL tok_pos '{self.tok_pos}'")

        return tensor[0, token_idx, :].detach().to(device="cpu", dtype=torch.float32)

    def _capture_demo_latent(self, sample: Dict, rng: random.Random) -> Dict[int, Dict[str, torch.Tensor]]:
        conversation, flat_images = self._build_conversation([], sample, include_answer=True)
        inputs = self._prepare_inputs(conversation, flat_images, add_generation_prompt=False)
        slot_mapping = self._get_slot_mapping()
        recorded: Dict[str, torch.Tensor] = {}

        def hook_fn(module, args, output, module_name=None):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                recorded[module_name] = tensor.detach().to(device="cpu", dtype=torch.float32)

        handles = self.model.register_forward_hook(list(slot_mapping), hook_fn)
        handle_list = handles if isinstance(handles, list) else [handles]
        try:
            with torch.no_grad():
                _ = self.model.model(**inputs, use_cache=False)
        finally:
            for handle in handle_list:
                handle.remove()

        missing = [module_name for module_name in slot_mapping if module_name not in recorded]
        if missing:
            raise RuntimeError(f"I2CL failed to capture demo latents for modules: {', '.join(missing[:3])}")

        output_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        for module_name, (layer_idx, module_key) in slot_mapping.items():
            output_dict.setdefault(layer_idx, {})[module_key] = self._select_token_feature(recorded[module_name], rng)
        return output_dict

    def _merge_demo_latents_mean(self, demo_latents: Sequence[Dict[int, Dict[str, torch.Tensor]]]) -> Dict[int, Dict[str, torch.Tensor]]:
        merged: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx in self.inject_layers:
            merged[layer_idx] = {}
            for module_name in self.modules:
                stacked = torch.stack(
                    [latent[layer_idx][module_name].to(dtype=torch.float32) for latent in demo_latents],
                    dim=0,
                )
                merged[layer_idx][module_name] = stacked.mean(dim=0).detach().to(device="cpu", dtype=torch.float32)
        return merged

    def _merge_demo_latents_pca(self, demo_latents: Sequence[Dict[int, Dict[str, torch.Tensor]]]) -> Dict[int, Dict[str, torch.Tensor]]:
        merged: Dict[int, Dict[str, torch.Tensor]] = {}
        for module_name in self.modules:
            stacked = torch.stack(
                [
                    torch.stack(
                        [latent[layer_idx][module_name].to(dtype=torch.float32) for layer_idx in self.inject_layers],
                        dim=0,
                    )
                    for latent in demo_latents
                ],
                dim=0,
            )
            num_demos, num_layers, hidden_size = stacked.shape
            if num_demos == 1:
                fused = stacked[0]
            else:
                flat = stacked.view(num_demos, num_layers * hidden_size)
                mean = flat.mean(dim=0, keepdim=True)
                centered = flat - mean
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                component = vh[0]
                pivot = int(torch.argmax(component.abs()).item())
                if component[pivot] < 0:
                    component = -component
                fused = (mean.squeeze(0) + component).view(num_layers, hidden_size)

            for local_idx, layer_idx in enumerate(self.inject_layers):
                merged.setdefault(layer_idx, {})[module_name] = fused[local_idx].detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
        return merged

    def _build_context_vector(self, demo_latents: Sequence[Dict[int, Dict[str, torch.Tensor]]]) -> Dict[int, Dict[str, torch.Tensor]]:
        if not demo_latents:
            raise RuntimeError("I2CL needs at least one demo latent to build a context vector")

        if self.post_fuse_method == "mean":
            context_vector = self._merge_demo_latents_mean(demo_latents)
        elif self.post_fuse_method == "pca":
            context_vector = self._merge_demo_latents_pca(demo_latents)
        else:
            raise ValueError(f"Unsupported I2CL post_fuse_method '{self.post_fuse_method}'")

        if self.context_init == "noise":
            for layer_idx, module_dict in context_vector.items():
                for module_name, tensor in module_dict.items():
                    context_vector[layer_idx][module_name] = torch.randn_like(tensor).detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    )
        return context_vector

    def _calibration_step(self, sample: Dict) -> torch.Tensor:
        prompt_conversation, prompt_images = self._build_conversation([], sample, include_answer=False)
        full_conversation, full_images = self._build_conversation([], sample, include_answer=True)

        prompt_inputs = self._prepare_inputs(prompt_conversation, prompt_images, add_generation_prompt=True)
        full_inputs = self._prepare_inputs(full_conversation, full_images, add_generation_prompt=False)
        labels = self._build_answer_only_labels(full_inputs, prompt_inputs)

        self.injector.enable(self.context_vector_dict, train_mode=True)
        try:
            outputs = self.model.model(**full_inputs, labels=labels, use_cache=False)
        finally:
            self.injector.disable()
        return outputs.loss

    def fit(self, train_data: Sequence[Dict]) -> None:
        if len(train_data) == 0:
            raise ValueError("I2CL fit requires non-empty training data")

        self.model.model.eval()
        self._build_label_index(train_data)
        rng = random.Random(self.support_seed)
        self.demo_indices = self._sample_demo_indices(train_data, rng)

        demo_latents = []
        iterator = self._iter_with_progress(self.demo_indices, desc="I2CL demo extraction")
        for demo_index in iterator:
            demo_latents.append(self._capture_demo_latent(train_data[demo_index], rng))
        self.context_vector_dict = self._build_context_vector(demo_latents)

        optimizer = torch.optim.AdamW(
            self.injector.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        calibration_items = [train_data[demo_index] for demo_index in self.demo_indices]
        step_budget = self.max_steps if self.max_steps > 0 else self.epochs * len(calibration_items)
        step_count = 0

        for epoch in self._iter_with_progress(range(self.epochs), desc="I2CL epochs"):
            if step_count >= step_budget:
                break

            order = list(range(len(calibration_items)))
            rng.shuffle(order)
            epoch_iter = self._iter_with_progress(order, desc=f"I2CL calibrate epoch {epoch + 1}")
            for item_idx in epoch_iter:
                if step_count >= step_budget:
                    break

                optimizer.zero_grad(set_to_none=True)
                loss = self._calibration_step(calibration_items[item_idx])
                loss.backward()
                if self.gradient_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.injector.parameters(), self.gradient_clip_norm)
                optimizer.step()
                step_count += 1

        self.injector.eval()
        self._fitted = True

    def _generate_with_context(self, sample: Dict) -> str:
        conversation, flat_images = self._build_conversation([], sample, include_answer=False)
        inputs = self._prepare_inputs(conversation, flat_images, add_generation_prompt=True)
        prompt_len = int(inputs["input_ids"].shape[-1])

        self.injector.enable(self.context_vector_dict, train_mode=False)
        try:
            generated_ids = self.model.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                use_cache=False,
            )
        finally:
            self.injector.disable()

        generated_ids = generated_ids[:, prompt_len:]
        outputs = self.model.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs[0] if outputs else ""

    def _generate_without_context(self, sample: Dict) -> str:
        conversation, flat_images = self._build_conversation([], sample, include_answer=False)
        inputs = self._prepare_inputs(conversation, flat_images, add_generation_prompt=True)
        prompt_len = int(inputs["input_ids"].shape[-1])

        generated_ids = self.model.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            use_cache=False,
        )
        generated_ids = generated_ids[:, prompt_len:]
        outputs = self.model.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs[0] if outputs else ""

    def _prepare_write_failure_inputs(
        self,
        sample: Dict,
        answer_text: str,
    ):
        prompt_conversation, prompt_images = self._build_conversation([], sample, include_answer=False)
        full_conversation, full_images = self._build_conversation(
            [],
            sample,
            include_answer=True,
            answer_text=answer_text,
        )
        prompt_inputs = self._prepare_inputs(prompt_conversation, prompt_images, add_generation_prompt=True)
        full_inputs = self._prepare_inputs(full_conversation, full_images, add_generation_prompt=False)
        return prompt_inputs, full_inputs

    def _run_write_failure_forward(self, full_inputs, *, steered: bool):
        if steered:
            self.injector.enable(self.context_vector_dict, train_mode=False)
        try:
            with torch.no_grad():
                outputs = self.model.model(
                    **full_inputs,
                    use_cache=False,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            if steered:
                self.injector.disable()

        if (
            outputs.attentions is None
            or not outputs.attentions
            or outputs.attentions[0] is None
        ):
            raise RuntimeError(
                "I2CL write-failure analysis requires real attention tensors. "
                "Run with model.model_args.attn_implementation=eager."
            )
        return outputs

    def _get_write_failure_task_vector(self) -> Dict[str, Any]:
        return {
            "context_vector": {
                int(layer_idx): {
                    str(module_name): tensor.detach().to(device="cpu", dtype=torch.float32)
                    for module_name, tensor in module_dict.items()
                }
                for layer_idx, module_dict in self.context_vector_dict.items()
            },
            "strength_params": [
                parameter.detach().to(device="cpu", dtype=torch.float32)
                for parameter in self.injector.strength_params
            ],
        }

    def _maybe_record_write_failure(
        self,
        sample: Dict,
        *,
        sample_index: int,
        steered_raw_output: str,
        steered_prediction: str,
    ) -> None:
        if not self.write_failure_recorder.should_record():
            return

        normal_raw_output = self._generate_without_context(sample)
        normal_prediction = self._match_label(normal_raw_output)
        answer_text = self.write_failure_recorder.choose_answer_text(
            sample,
            normal_prediction=normal_prediction,
            steered_prediction=steered_prediction,
        )
        prompt_inputs, full_inputs = self._prepare_write_failure_inputs(sample, answer_text)
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        normal_outputs = self._run_write_failure_forward(full_inputs, steered=False)
        steered_outputs = self._run_write_failure_forward(full_inputs, steered=True)

        self.write_failure_recorder.record_pair(
            sample=sample,
            sample_index=sample_index,
            full_inputs=full_inputs,
            prompt_len=prompt_len,
            normal_outputs=normal_outputs,
            steered_outputs=steered_outputs,
            task_vector_obj=self._get_write_failure_task_vector(),
            analysis_answer_text=answer_text,
            normal_raw_output=normal_raw_output,
            steered_raw_output=steered_raw_output,
            normal_prediction=normal_prediction,
            steered_prediction=steered_prediction,
            processor=self.model.processor,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, sample: Dict) -> str:
        if not self._fitted:
            raise RuntimeError("I2CL method is not fitted. Call fit(train_data) first.")

        sample_index = self._predict_calls
        self._predict_calls += 1

        raw = self._generate_with_context(sample)
        prediction = self._match_label(raw)
        self._maybe_record_write_failure(
            sample,
            sample_index=sample_index,
            steered_raw_output=raw,
            steered_prediction=prediction,
        )
        return prediction

    def export_diagnostics(self) -> Dict:
        return {
            "write_failure_analysis": self.write_failure_recorder.export(),
        }
