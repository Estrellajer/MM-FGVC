"""
Offline analysis for "write" failure in LMMs.

This script compares paired normal / steered forward dumps and measures:

1. Visual Attention Ratio drop
2. Attention entropy change and distribution drift
3. Hidden-state corruption on prediction tokens
4. Task-vector / hidden-state norm mismatch
5. Optional qualitative attention heatmaps over image tokens

The original version of this file was only a placeholder and had a few
scientific issues:

- it reported raw attention mass instead of a true ratio
- it silently assumed a single tensor shape for attentions
- it approximated "steering norm" with final-token hidden-state drift
  instead of the actual injected vector
- it only looked at one token / one layer and could not produce heatmaps

This version is an executable CLI for offline analysis.

Recommended bundle format (.pt/.pth/.npz):
{
    "hidden_states": <tensor or HF-style tuple of tensors>,
    "attentions": <tensor or HF-style tuple of tensors>,
    "task_vector": <optional tensor / dict / nested structure>,
}

Example:
python swap/paper/scripts/analyze_write_failure.py \
  --normal-bundle baseline_dump.pt \
  --steered-bundle stv_dump.pt \
  --image-token-start 32 \
  --image-token-end 608 \
  --query-token-indices=-1,-2,-3 \
  --output-json swap/paper/outputs/write_failure_analysis.json \
  --heatmap-path swap/paper/figures/write_failure_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-9


def _to_python_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _json_default(obj: Any):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _load_raw(path: str):
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".pt", ".pth", ".bin"}:
        return torch.load(file_path, map_location="cpu")
    if suffix == ".npy":
        loaded = np.load(file_path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
            return loaded.item()
        return loaded
    if suffix == ".npz":
        with np.load(file_path, allow_pickle=True) as loaded:
            return {key: loaded[key] for key in loaded.files}

    raise ValueError(f"Unsupported file type for '{path}'")


def _maybe_tensor(obj: Any) -> bool:
    return torch.is_tensor(obj) or isinstance(obj, np.ndarray)


def _to_tensor(obj: Any) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj.detach().to(device="cpu", dtype=torch.float32)
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.asarray(obj)).to(dtype=torch.float32)
    if isinstance(obj, (float, int, bool, np.number)):
        return torch.tensor(obj, dtype=torch.float32)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return torch.as_tensor(obj, dtype=torch.float32)
    raise TypeError(f"Cannot convert object of type {type(obj).__name__} to tensor")


def _load_component(
    explicit_path: str | None,
    explicit_key: str | None,
    bundle_path: str | None,
    default_keys: Sequence[str],
    component_name: str,
    *,
    required: bool,
):
    source = explicit_path or bundle_path
    if source is None:
        if required:
            raise ValueError(
                f"Missing required input for '{component_name}'. Provide either an explicit file "
                f"or a bundle containing one of: {', '.join(default_keys)}"
            )
        return None

    payload = _load_raw(source)
    if _maybe_tensor(payload) or isinstance(payload, (list, tuple)):
        return payload

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected '{component_name}' source '{source}' to be a tensor, list/tuple, or dict bundle; "
            f"got {type(payload).__name__}"
        )

    candidate_keys = [explicit_key] if explicit_key else list(default_keys)
    for key in candidate_keys:
        if key and key in payload:
            return payload[key]

    if required:
        raise KeyError(
            f"Could not find '{component_name}' in bundle '{source}'. "
            f"Looked for keys: {', '.join(candidate_keys)}"
        )
    return None


def _normalize_hidden_states(obj: Any, layout: str) -> torch.Tensor:
    if isinstance(obj, (list, tuple)):
        layers = [_to_tensor(item) for item in obj]
        if not layers:
            raise ValueError("Hidden-state sequence is empty")
        normalized_layers = []
        for layer in layers:
            if layer.ndim == 2:
                normalized_layers.append(layer.unsqueeze(0))
            elif layer.ndim == 3:
                normalized_layers.append(layer)
            else:
                raise ValueError(
                    "Expected each hidden-state layer tensor to have shape [batch, seq, dim] or [seq, dim], "
                    f"got {tuple(layer.shape)}"
                )
        return torch.stack(normalized_layers, dim=0)

    tensor = _to_tensor(obj)
    if tensor.ndim == 4:
        if layout == "batch_first":
            return tensor.permute(1, 0, 2, 3).contiguous()
        return tensor
    if tensor.ndim == 3:
        if layout == "batch_first":
            return tensor.unsqueeze(0)
        return tensor.unsqueeze(1)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)

    raise ValueError(
        "Hidden states must have shape [layers, batch, seq, dim], [layers, seq, dim], "
        "[batch, seq, dim], or a list/tuple of [batch, seq, dim] tensors"
    )


def _normalize_attentions(obj: Any, layout: str) -> torch.Tensor:
    if isinstance(obj, (list, tuple)):
        layers = [_to_tensor(item) for item in obj]
        if not layers:
            raise ValueError("Attention sequence is empty")
        normalized_layers = []
        for layer in layers:
            if layer.ndim == 4:
                normalized_layers.append(layer)
            elif layer.ndim == 3:
                normalized_layers.append(layer.unsqueeze(0))
            elif layer.ndim == 2:
                normalized_layers.append(layer.unsqueeze(0).unsqueeze(0))
            elif layer.ndim == 1:
                normalized_layers.append(layer.unsqueeze(0).unsqueeze(0).unsqueeze(0))
            else:
                raise ValueError(
                    "Expected each attention layer tensor to have shape [batch, heads, q, k], "
                    "[heads, q, k], [q, k], or [k], "
                    f"got {tuple(layer.shape)}"
                )
        return torch.stack(normalized_layers, dim=0)

    tensor = _to_tensor(obj)
    if tensor.ndim == 5:
        if layout == "batch_first":
            return tensor.permute(1, 0, 2, 3, 4).contiguous()
        return tensor
    if tensor.ndim == 4:
        if layout == "batch_first":
            return tensor.unsqueeze(0)
        return tensor.unsqueeze(1)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    raise ValueError(
        "Attentions must have shape [layers, batch, heads, q, k], [layers, heads, q, k], "
        "[heads, q, k], [q, k], [k], or a list/tuple of per-layer tensors"
    )


def _trim_pair(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != b.ndim:
        raise ValueError(f"Cannot align tensors with different ranks: {a.ndim} vs {b.ndim}")

    slices = tuple(slice(0, min(int(sa), int(sb))) for sa, sb in zip(a.shape, b.shape))
    return a[slices].contiguous(), b[slices].contiguous()


def _parse_indices(spec: str | None) -> List[int] | None:
    if spec is None:
        return None
    values = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def _resolve_relative_indices(length: int, indices: Sequence[int] | None, *, default_last: bool) -> List[int]:
    if length <= 0:
        raise ValueError("Cannot resolve indices for an empty dimension")
    if not indices:
        indices = [-1] if default_last else [0]

    resolved = []
    for idx in indices:
        resolved_idx = idx if idx >= 0 else length + idx
        if resolved_idx < 0 or resolved_idx >= length:
            raise IndexError(f"Index {idx} resolves to {resolved_idx}, which is outside length {length}")
        resolved.append(int(resolved_idx))
    return resolved


def _build_mask(
    length: int,
    *,
    start: int | None,
    end: int | None,
    indices: Sequence[int] | None,
) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)

    if start is not None or end is not None:
        if start is None or end is None:
            raise ValueError("image-token-start and image-token-end must be provided together")
        start = max(0, int(start))
        end = min(length, int(end))
        if end <= start:
            raise ValueError(
                f"Invalid image token range [{start}, {end}); end must be greater than start and within [0, {length}]"
            )
        mask[start:end] = True

    if indices:
        for idx in indices:
            resolved_idx = idx if idx >= 0 else length + idx
            if resolved_idx < 0 or resolved_idx >= length:
                raise IndexError(f"Image token index {idx} resolves outside length {length}")
            mask[resolved_idx] = True

    if not mask.any():
        raise ValueError("Image token mask is empty. Provide a valid range or explicit indices.")
    return mask


def _entropy(prob: torch.Tensor) -> torch.Tensor:
    return -(prob * torch.log(prob.clamp_min(EPS))).sum(dim=-1)


def _normalized_entropy(prob: torch.Tensor) -> torch.Tensor:
    num_classes = int(prob.shape[-1])
    if num_classes <= 1:
        return torch.zeros(prob.shape[:-1], dtype=prob.dtype)
    return _entropy(prob) / math.log(num_classes)


def _safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return _to_python_float(values.mean())


def _summarize_vector_norms(values: torch.Tensor) -> Dict[str, float]:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": _to_python_float(flat.mean()),
        "std": _to_python_float(flat.std(unbiased=False)) if flat.numel() > 1 else 0.0,
        "min": _to_python_float(flat.min()),
        "max": _to_python_float(flat.max()),
    }


def _flatten_tensor_collection(obj: Any, prefix: str = "root") -> List[Tuple[str, torch.Tensor]]:
    leaves: List[Tuple[str, torch.Tensor]] = []

    if _maybe_tensor(obj):
        leaves.append((prefix, _to_tensor(obj)))
        return leaves

    if isinstance(obj, dict):
        for key, value in obj.items():
            leaves.extend(_flatten_tensor_collection(value, f"{prefix}.{key}"))
        return leaves

    if isinstance(obj, (list, tuple)):
        if obj and all(_maybe_tensor(item) for item in obj):
            try:
                leaves.append((prefix, _to_tensor(obj)))
                return leaves
            except Exception:
                pass
        for idx, value in enumerate(obj):
            leaves.extend(_flatten_tensor_collection(value, f"{prefix}[{idx}]"))
        return leaves

    return leaves


def _collect_leaf_vector_norms(obj: Any) -> Tuple[List[str], torch.Tensor]:
    leaves = _flatten_tensor_collection(obj)
    names: List[str] = []
    norms: List[torch.Tensor] = []

    for name, tensor in leaves:
        if tensor.numel() == 0:
            continue
        names.append(name)
        if tensor.ndim == 0:
            norms.append(tensor.abs().reshape(1))
        elif tensor.ndim == 1:
            norms.append(tensor.norm(p=2).reshape(1))
        else:
            flat = tensor.reshape(-1, tensor.shape[-1])
            norms.append(flat.norm(p=2, dim=-1))

    if not norms:
        return names, torch.zeros(0, dtype=torch.float32)
    return names, torch.cat([norm.reshape(-1) for norm in norms], dim=0)


def compute_attention_metrics(
    attention_normal: torch.Tensor,
    attention_steered: torch.Tensor,
    *,
    image_mask: torch.Tensor,
    query_indices: Sequence[int] | None,
) -> Dict[str, Any]:
    attention_normal, attention_steered = _trim_pair(attention_normal, attention_steered)
    key_len = int(attention_normal.shape[-1])
    image_mask = image_mask[:key_len]

    query_len = int(attention_normal.shape[-2])
    resolved_queries = _resolve_relative_indices(query_len, query_indices, default_last=True)

    selected_normal = attention_normal[:, :, :, resolved_queries, :]
    selected_steered = attention_steered[:, :, :, resolved_queries, :]

    prob_normal = selected_normal / selected_normal.sum(dim=-1, keepdim=True).clamp_min(EPS)
    prob_steered = selected_steered / selected_steered.sum(dim=-1, keepdim=True).clamp_min(EPS)

    image_prob_normal = prob_normal[..., image_mask]
    image_prob_steered = prob_steered[..., image_mask]

    visual_ratio_normal = image_prob_normal.sum(dim=-1)
    visual_ratio_steered = image_prob_steered.sum(dim=-1)

    entropy_normal = _entropy(prob_normal)
    entropy_steered = _entropy(prob_steered)
    normalized_entropy_normal = _normalized_entropy(prob_normal)
    normalized_entropy_steered = _normalized_entropy(prob_steered)

    image_mass_normal = image_prob_normal.sum(dim=-1, keepdim=True)
    image_mass_steered = image_prob_steered.sum(dim=-1, keepdim=True)
    conditional_image_prob_normal = image_prob_normal / image_mass_normal.clamp_min(EPS)
    conditional_image_prob_steered = image_prob_steered / image_mass_steered.clamp_min(EPS)
    conditional_image_entropy_normal = _normalized_entropy(conditional_image_prob_normal)
    conditional_image_entropy_steered = _normalized_entropy(conditional_image_prob_steered)

    midpoint = 0.5 * (prob_normal + prob_steered)
    js_divergence = 0.5 * (
        (prob_normal * (prob_normal.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())).sum(dim=-1)
        + (prob_steered * (prob_steered.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())).sum(dim=-1)
    )
    total_variation = 0.5 * (prob_normal - prob_steered).abs().sum(dim=-1)

    per_layer_visual_ratio_normal = visual_ratio_normal.mean(dim=(1, 2, 3))
    per_layer_visual_ratio_steered = visual_ratio_steered.mean(dim=(1, 2, 3))

    image_profile_normal = image_prob_normal.mean(dim=(0, 1, 2, 3))
    image_profile_steered = image_prob_steered.mean(dim=(0, 1, 2, 3))

    normal_ratio_mean = _safe_mean(visual_ratio_normal)
    steered_ratio_mean = _safe_mean(visual_ratio_steered)
    ratio_drop = normal_ratio_mean - steered_ratio_mean
    ratio_drop_pct = 100.0 * ratio_drop / max(normal_ratio_mean, EPS)

    return {
        "query_token_indices": list(resolved_queries),
        "num_layers_compared": int(attention_normal.shape[0]),
        "num_heads_compared": int(attention_normal.shape[2]),
        "num_image_tokens": int(image_mask.sum().item()),
        "visual_attention_ratio_normal": normal_ratio_mean,
        "visual_attention_ratio_steered": steered_ratio_mean,
        "visual_attention_ratio_drop": ratio_drop,
        "visual_attention_ratio_drop_percent": ratio_drop_pct,
        "attention_entropy_normal": _safe_mean(entropy_normal),
        "attention_entropy_steered": _safe_mean(entropy_steered),
        "normalized_attention_entropy_normal": _safe_mean(normalized_entropy_normal),
        "normalized_attention_entropy_steered": _safe_mean(normalized_entropy_steered),
        "conditional_image_entropy_normal": _safe_mean(conditional_image_entropy_normal),
        "conditional_image_entropy_steered": _safe_mean(conditional_image_entropy_steered),
        "attention_js_divergence": _safe_mean(js_divergence),
        "attention_total_variation": _safe_mean(total_variation),
        "per_layer_visual_attention_ratio_normal": per_layer_visual_ratio_normal.tolist(),
        "per_layer_visual_attention_ratio_steered": per_layer_visual_ratio_steered.tolist(),
        "image_attention_profile_normal": image_profile_normal.tolist(),
        "image_attention_profile_steered": image_profile_steered.tolist(),
        "image_attention_profile_delta": (image_profile_steered - image_profile_normal).tolist(),
    }


def compute_hidden_state_metrics(
    hidden_normal: torch.Tensor,
    hidden_steered: torch.Tensor,
    *,
    image_mask: torch.Tensor,
    query_indices: Sequence[int] | None,
) -> Dict[str, Any]:
    hidden_normal, hidden_steered = _trim_pair(hidden_normal, hidden_steered)
    seq_len = int(hidden_normal.shape[-2])
    dim = int(hidden_normal.shape[-1])
    image_mask = image_mask[:seq_len]
    resolved_queries = _resolve_relative_indices(seq_len, query_indices, default_last=True)

    query_normal = hidden_normal[:, :, resolved_queries, :]
    query_steered = hidden_steered[:, :, resolved_queries, :]
    delta = hidden_steered - hidden_normal

    query_cosine = F.cosine_similarity(query_normal, query_steered, dim=-1)
    query_l2 = (query_steered - query_normal).norm(p=2, dim=-1)
    query_base_norm = query_normal.norm(p=2, dim=-1).clamp_min(EPS)
    query_l2_ratio = query_l2 / query_base_norm

    token_norm_normal = hidden_normal.norm(p=2, dim=-1)
    token_norm_steered = hidden_steered.norm(p=2, dim=-1)
    token_delta_norm = delta.norm(p=2, dim=-1)

    metrics: Dict[str, Any] = {
        "hidden_size": dim,
        "query_token_indices": list(resolved_queries),
        "representation_cosine_similarity": _safe_mean(query_cosine),
        "query_hidden_l2_distance": _safe_mean(query_l2),
        "query_hidden_l2_ratio": _safe_mean(query_l2_ratio),
        "token_norm_normal": _summarize_vector_norms(token_norm_normal),
        "token_norm_steered": _summarize_vector_norms(token_norm_steered),
        "token_hidden_drift_norm": _summarize_vector_norms(token_delta_norm),
    }

    if image_mask.any():
        image_norm_normal = token_norm_normal[..., image_mask]
        image_norm_steered = token_norm_steered[..., image_mask]
        image_delta_norm = token_delta_norm[..., image_mask]
        metrics.update(
            {
                "image_token_norm_normal": _summarize_vector_norms(image_norm_normal),
                "image_token_norm_steered": _summarize_vector_norms(image_norm_steered),
                "image_token_hidden_drift_ratio": _to_python_float(
                    image_delta_norm.mean() / image_norm_normal.mean().clamp_min(EPS)
                ),
            }
        )

    text_mask = ~image_mask
    if text_mask.any():
        text_norm_normal = token_norm_normal[..., text_mask]
        text_norm_steered = token_norm_steered[..., text_mask]
        text_delta_norm = token_delta_norm[..., text_mask]
        metrics.update(
            {
                "text_token_norm_normal": _summarize_vector_norms(text_norm_normal),
                "text_token_norm_steered": _summarize_vector_norms(text_norm_steered),
                "text_token_hidden_drift_ratio": _to_python_float(
                    text_delta_norm.mean() / text_norm_normal.mean().clamp_min(EPS)
                ),
            }
        )

    return metrics


def compute_norm_mismatch(
    task_vector_obj: Any | None,
    hidden_normal: torch.Tensor,
    *,
    image_mask: torch.Tensor,
) -> Dict[str, Any]:
    token_norm_normal = hidden_normal.norm(p=2, dim=-1)
    seq_len = int(hidden_normal.shape[-2])
    hidden_size = int(hidden_normal.shape[-1])
    image_mask = image_mask[:seq_len]
    text_mask = ~image_mask

    base = {
        "hidden_token_norm_mean": _to_python_float(token_norm_normal.mean()),
        "hidden_token_norm_std": _to_python_float(token_norm_normal.std(unbiased=False)),
        "image_token_norm_mean": _to_python_float(token_norm_normal[..., image_mask].mean())
        if image_mask.any()
        else None,
        "text_token_norm_mean": _to_python_float(token_norm_normal[..., text_mask].mean())
        if text_mask.any()
        else None,
    }

    if task_vector_obj is None:
        base["task_vector_available"] = False
        return base

    leaf_names, leaf_norms = _collect_leaf_vector_norms(task_vector_obj)
    result = dict(base)
    result.update(
        {
            "task_vector_available": True,
            "task_vector_num_leaf_tensors": len(leaf_names),
            "task_vector_num_vectors": int(leaf_norms.numel()),
            "task_vector_norm": _summarize_vector_norms(leaf_norms),
        }
    )

    if leaf_norms.numel() == 0:
        result["task_vector_comparable_to_hidden"] = False
        result["task_vector_comparison_note"] = "Task vector exists but no numeric tensor leaves were found."
        return result

    result["task_vector_to_hidden_norm_ratio"] = _to_python_float(leaf_norms.mean() / token_norm_normal.mean().clamp_min(EPS))

    comparable_dims = []
    for _, leaf in _flatten_tensor_collection(task_vector_obj):
        if leaf.numel() == 0:
            continue
        if leaf.ndim == 0:
            continue
        comparable_dims.append(int(leaf.shape[-1]))

    if comparable_dims and all(dim == hidden_size for dim in comparable_dims):
        result["task_vector_comparable_to_hidden"] = True
        if image_mask.any():
            result["task_vector_to_image_token_norm_ratio"] = _to_python_float(
                leaf_norms.mean() / token_norm_normal[..., image_mask].mean().clamp_min(EPS)
            )
        if text_mask.any():
            result["task_vector_to_text_token_norm_ratio"] = _to_python_float(
                leaf_norms.mean() / token_norm_normal[..., text_mask].mean().clamp_min(EPS)
            )
    else:
        result["task_vector_comparable_to_hidden"] = False
        result["task_vector_comparison_note"] = (
            "Task-vector last dimension does not consistently match hidden size, "
            "so cross-space norm ratios should be treated as approximate."
        )

    return result


def _infer_grid(num_tokens: int, height: int | None, width: int | None) -> Tuple[int, int]:
    if height is not None and width is not None:
        if height * width != num_tokens:
            raise ValueError(f"Provided grid {height}x{width} does not match {num_tokens} image tokens")
        return int(height), int(width)

    if height is not None:
        if num_tokens % height != 0:
            raise ValueError(f"Cannot infer grid width for {num_tokens} tokens from height {height}")
        return int(height), int(num_tokens // height)

    if width is not None:
        if num_tokens % width != 0:
            raise ValueError(f"Cannot infer grid height for {num_tokens} tokens from width {width}")
        return int(num_tokens // width), int(width)

    side = int(round(math.sqrt(num_tokens)))
    if side * side == num_tokens:
        return side, side
    return 1, num_tokens


def save_heatmap(
    metrics: Dict[str, Any],
    path: str,
    *,
    grid_height: int | None,
    grid_width: int | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal = torch.tensor(metrics["attention"]["image_attention_profile_normal"], dtype=torch.float32)
    steered = torch.tensor(metrics["attention"]["image_attention_profile_steered"], dtype=torch.float32)
    delta = steered - normal

    rows, cols = _infer_grid(int(normal.numel()), grid_height, grid_width)
    normal_grid = normal.view(rows, cols)
    steered_grid = steered.view(rows, cols)
    delta_grid = delta.view(rows, cols)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    vmax = max(_to_python_float(normal_grid.max()), _to_python_float(steered_grid.max()), EPS)
    vmax_delta = max(abs(_to_python_float(delta_grid.min())), abs(_to_python_float(delta_grid.max())), EPS)

    panels = [
        ("Normal", normal_grid, "viridis", 0.0, vmax),
        ("Steered", steered_grid, "viridis", 0.0, vmax),
        ("Delta", delta_grid, "coolwarm", -vmax_delta, vmax_delta),
    ]

    for ax, (title, grid, cmap, vmin, vmax_panel) in zip(axes, panels):
        im = ax.imshow(grid.numpy(), cmap=cmap, vmin=vmin, vmax=vmax_panel, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Prediction-token attention over image tokens", fontsize=12)
    fig.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def analyze_write_failure(
    *,
    hidden_normal_obj: Any,
    hidden_steered_obj: Any,
    attention_normal_obj: Any,
    attention_steered_obj: Any,
    task_vector_obj: Any | None,
    image_token_start: int | None,
    image_token_end: int | None,
    image_token_indices: Sequence[int] | None,
    query_token_indices: Sequence[int] | None,
    hidden_layout: str,
    attention_layout: str,
) -> Dict[str, Any]:
    hidden_normal = _normalize_hidden_states(hidden_normal_obj, layout=hidden_layout)
    hidden_steered = _normalize_hidden_states(hidden_steered_obj, layout=hidden_layout)
    attention_normal = _normalize_attentions(attention_normal_obj, layout=attention_layout)
    attention_steered = _normalize_attentions(attention_steered_obj, layout=attention_layout)

    hidden_normal, hidden_steered = _trim_pair(hidden_normal, hidden_steered)
    attention_normal, attention_steered = _trim_pair(attention_normal, attention_steered)

    shared_key_len = min(int(attention_normal.shape[-1]), int(hidden_normal.shape[-2]))
    image_mask = _build_mask(
        shared_key_len,
        start=image_token_start,
        end=image_token_end,
        indices=image_token_indices,
    )

    attention_metrics = compute_attention_metrics(
        attention_normal,
        attention_steered,
        image_mask=image_mask[: int(attention_normal.shape[-1])],
        query_indices=query_token_indices,
    )
    hidden_metrics = compute_hidden_state_metrics(
        hidden_normal,
        hidden_steered,
        image_mask=image_mask[: int(hidden_normal.shape[-2])],
        query_indices=query_token_indices,
    )
    norm_mismatch = compute_norm_mismatch(
        task_vector_obj,
        hidden_normal,
        image_mask=image_mask[: int(hidden_normal.shape[-2])],
    )

    result = {
        "summary": {
            "supports_write_failure_claim": (
                attention_metrics["visual_attention_ratio_steered"]
                < attention_metrics["visual_attention_ratio_normal"]
            ),
            "interpretation": (
                "Steered prediction tokens allocate less attention mass to image tokens than the normal run."
                if attention_metrics["visual_attention_ratio_steered"]
                < attention_metrics["visual_attention_ratio_normal"]
                else "No visual-attention drop was detected in the provided tensors."
            ),
        },
        "attention": attention_metrics,
        "hidden_state": hidden_metrics,
        "norm_mismatch": norm_mismatch,
    }
    return result


def build_self_test_inputs() -> Dict[str, Any]:
    generator = torch.Generator().manual_seed(7)

    layers = 4
    batch = 1
    heads = 3
    query_len = 6
    seq_len = 20
    hidden_dim = 16

    base_attn = torch.rand(layers, batch, heads, query_len, seq_len, generator=generator)
    base_attn = base_attn / base_attn.sum(dim=-1, keepdim=True)

    image_start = 8
    image_end = 16
    image_slice = slice(image_start, image_end)

    for token_idx in [4, 5]:
        base_attn[:, :, :, token_idx, image_slice] += 3.0
    base_attn = base_attn / base_attn.sum(dim=-1, keepdim=True)

    steered_attn = base_attn.clone()
    steered_attn[:, :, :, 4:, image_slice] *= 0.18
    steered_attn[:, :, :, 4:, :image_start] += 0.45
    steered_attn[:, :, :, 4:, image_end:] += 0.35
    steered_attn = steered_attn / steered_attn.sum(dim=-1, keepdim=True)

    base_hidden = torch.randn(layers, batch, seq_len, hidden_dim, generator=generator)
    steered_hidden = base_hidden.clone()
    steered_hidden[:, :, -1, :] += 1.2
    steered_hidden[:, :, image_slice, :] += 0.15 * torch.randn(
        layers, batch, image_end - image_start, hidden_dim, generator=generator
    )

    task_vector = torch.randn(hidden_dim, generator=generator) * 1.6

    return {
        "hidden_normal": base_hidden,
        "hidden_steered": steered_hidden,
        "attention_normal": base_attn,
        "attention_steered": steered_attn,
        "task_vector": task_vector,
        "image_token_start": image_start,
        "image_token_end": image_end,
        "query_token_indices": [-1, -2],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze write-method failure from paired LMM forward dumps")

    parser.add_argument("--normal-bundle", type=str, default=None, help="Bundle containing normal hidden states / attentions")
    parser.add_argument("--steered-bundle", type=str, default=None, help="Bundle containing steered hidden states / attentions")

    parser.add_argument("--normal-hidden", type=str, default=None, help="Explicit file for normal hidden states")
    parser.add_argument("--steered-hidden", type=str, default=None, help="Explicit file for steered hidden states")
    parser.add_argument("--normal-attn", type=str, default=None, help="Explicit file for normal attentions")
    parser.add_argument("--steered-attn", type=str, default=None, help="Explicit file for steered attentions")
    parser.add_argument("--task-vector", type=str, default=None, help="Explicit file for task vector / intervention tensor")

    parser.add_argument("--normal-hidden-key", type=str, default=None)
    parser.add_argument("--steered-hidden-key", type=str, default=None)
    parser.add_argument("--normal-attn-key", type=str, default=None)
    parser.add_argument("--steered-attn-key", type=str, default=None)
    parser.add_argument("--task-vector-key", type=str, default=None)

    parser.add_argument("--hidden-layout", choices=["layers_first", "batch_first"], default="layers_first")
    parser.add_argument("--attention-layout", choices=["layers_first", "batch_first"], default="layers_first")

    parser.add_argument("--image-token-start", type=int, default=None, help="Inclusive image-token start index")
    parser.add_argument("--image-token-end", type=int, default=None, help="Exclusive image-token end index")
    parser.add_argument(
        "--image-token-indices",
        type=str,
        default=None,
        help="Comma-separated explicit image token indices, e.g. 32,33,34",
    )
    parser.add_argument(
        "--query-token-indices",
        type=str,
        default="-1",
        help="Comma-separated query token indices to analyze; defaults to the last generated token",
    )

    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save metrics JSON")
    parser.add_argument("--heatmap-path", type=str, default=None, help="Optional path to save a qualitative heatmap")
    parser.add_argument("--grid-height", type=int, default=None, help="Optional image-patch grid height for heatmap")
    parser.add_argument("--grid-width", type=int, default=None, help="Optional image-patch grid width for heatmap")
    parser.add_argument("--self-test", action="store_true", help="Run a deterministic synthetic smoke test")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.self_test:
        payload = build_self_test_inputs()
        results = analyze_write_failure(
            hidden_normal_obj=payload["hidden_normal"],
            hidden_steered_obj=payload["hidden_steered"],
            attention_normal_obj=payload["attention_normal"],
            attention_steered_obj=payload["attention_steered"],
            task_vector_obj=payload["task_vector"],
            image_token_start=payload["image_token_start"],
            image_token_end=payload["image_token_end"],
            image_token_indices=None,
            query_token_indices=payload["query_token_indices"],
            hidden_layout=args.hidden_layout,
            attention_layout=args.attention_layout,
        )
    else:
        query_token_indices = _parse_indices(args.query_token_indices)
        image_token_indices = _parse_indices(args.image_token_indices)

        hidden_normal_obj = _load_component(
            args.normal_hidden,
            args.normal_hidden_key,
            args.normal_bundle,
            default_keys=("hidden_states", "hidden", "hs"),
            component_name="normal_hidden_states",
            required=True,
        )
        hidden_steered_obj = _load_component(
            args.steered_hidden,
            args.steered_hidden_key,
            args.steered_bundle,
            default_keys=("hidden_states", "hidden", "hs"),
            component_name="steered_hidden_states",
            required=True,
        )
        attention_normal_obj = _load_component(
            args.normal_attn,
            args.normal_attn_key,
            args.normal_bundle,
            default_keys=("attentions", "attention", "attn"),
            component_name="normal_attentions",
            required=True,
        )
        attention_steered_obj = _load_component(
            args.steered_attn,
            args.steered_attn_key,
            args.steered_bundle,
            default_keys=("attentions", "attention", "attn"),
            component_name="steered_attentions",
            required=True,
        )
        task_vector_obj = _load_component(
            args.task_vector,
            args.task_vector_key,
            args.steered_bundle,
            default_keys=("task_vector", "context_vector", "intervention", "shift"),
            component_name="task_vector",
            required=False,
        )

        results = analyze_write_failure(
            hidden_normal_obj=hidden_normal_obj,
            hidden_steered_obj=hidden_steered_obj,
            attention_normal_obj=attention_normal_obj,
            attention_steered_obj=attention_steered_obj,
            task_vector_obj=task_vector_obj,
            image_token_start=args.image_token_start,
            image_token_end=args.image_token_end,
            image_token_indices=image_token_indices,
            query_token_indices=query_token_indices,
            hidden_layout=args.hidden_layout,
            attention_layout=args.attention_layout,
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=_json_default))

    if args.heatmap_path:
        save_heatmap(
            results,
            args.heatmap_path,
            grid_height=args.grid_height,
            grid_width=args.grid_width,
        )

    print(json.dumps(results, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
