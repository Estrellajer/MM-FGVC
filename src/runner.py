from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .data import collect_label_space, load_train_val
from .evaluate import build_evaluator
from .methods import build_method
from .models import get_model_class


def _to_plain_dict(cfg_part: Any) -> Dict[str, Any]:
    if cfg_part is None:
        return {}
    if isinstance(cfg_part, DictConfig):
        return OmegaConf.to_container(cfg_part, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_part, dict):
        return dict(cfg_part)
    return {}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(cfg: DictConfig):
    model_cls = get_model_class(cfg.model.name)
    model_root = hydra.utils.to_absolute_path(cfg.model.model_root)

    processor_args = _to_plain_dict(cfg.model.get("processor_args"))
    model_args = _to_plain_dict(cfg.model.get("model_args"))
    common_args = _to_plain_dict(cfg.model.get("common_args"))

    model = model_cls(
        model_root=model_root,
        processor_args=processor_args,
        model_args=model_args,
        **common_args,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def _resolve_evaluator_name(cfg: DictConfig) -> str:
    evaluator_name = str(cfg.evaluator.name)
    if evaluator_name != "auto":
        return evaluator_name

    default_eval = cfg.dataset.get("default_evaluator")
    if default_eval is None:
        raise ValueError(
            "evaluator.name is set to 'auto' but dataset.default_evaluator is missing"
        )
    return str(default_eval)


def _save_results(
    cfg: DictConfig,
    metrics: Dict[str, Any],
    predictions: List[str],
    labels: List[str],
    val_data: List[Dict[str, Any]],
    timings: Dict[str, Any],
    diagnostics: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    output_dir = Path(hydra.utils.to_absolute_path(cfg.run.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg.run.get("run_name")
    if not run_name:
        run_name = f"{cfg.model.name}_{cfg.dataset.name}_{cfg.method.name}_{cfg.evaluator.name}"

    metrics_path = output_dir / f"{run_name}.metrics.json"
    payload = {
        "model": str(cfg.model.name),
        "dataset": str(cfg.dataset.name),
        "method": str(cfg.method.name),
        "evaluator": str(cfg.evaluator.name),
        "metrics": metrics,
        "timings": timings,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_path = output_dir / f"{run_name}.predictions.jsonl"
    if bool(cfg.run.get("save_predictions", True)):
        with prediction_path.open("w", encoding="utf-8") as fp:
            for sample, pred, label in zip(val_data, predictions, labels):
                row = {
                    "question_id": sample.get("question_id"),
                    "image": sample.get("image"),
                    "images": sample.get("images"),
                    "question": sample.get("question"),
                    "prediction": pred,
                    "label": label,
                    "correct": pred == label,
                }
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    saved = {
        "metrics": str(metrics_path),
        "predictions": str(prediction_path),
    }
    if diagnostics:
        diagnostics_path = output_dir / f"{run_name}.diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved["diagnostics"] = str(diagnostics_path)

    return saved


def run_experiment(cfg: DictConfig) -> Dict[str, Any]:
    _set_seed(int(cfg.run.seed))

    train_path = hydra.utils.to_absolute_path(cfg.dataset.train_path)
    val_path = hydra.utils.to_absolute_path(cfg.dataset.val_path)
    train_data, val_data = load_train_val(cfg.dataset.name, train_path, val_path)

    label_space = collect_label_space(train_data, val_data)
    model = _build_model(cfg)

    method_params = _to_plain_dict(cfg.method.get("params"))
    method = build_method(
        cfg.method.name,
        model=model,
        dataset_name=cfg.dataset.name,
        label_space=label_space,
        **method_params,
    )

    timings: Dict[str, Any] = {}
    fit_started = time.perf_counter()
    if bool(cfg.method.get("fit_on_train", False)):
        method.fit(train_data)
    fit_time_sec = time.perf_counter() - fit_started

    labels = [str(item["label"]).strip() for item in val_data]
    iterator = tqdm(val_data, desc="Evaluating") if bool(cfg.run.get("progress_bar", True)) else val_data
    predict_started = time.perf_counter()
    predictions = [method.predict(item) for item in iterator]
    predict_time_sec = time.perf_counter() - predict_started

    timings = {
        "fit_time_sec": fit_time_sec,
        "predict_time_sec": predict_time_sec,
        "total_time_sec": fit_time_sec + predict_time_sec,
        "num_eval_samples": len(val_data),
        "avg_predict_time_sec": (predict_time_sec / len(val_data)) if val_data else 0.0,
    }

    evaluator_name = _resolve_evaluator_name(cfg)
    evaluator = build_evaluator(evaluator_name)
    metrics = evaluator.evaluate(predictions, labels, val_data)

    diagnostics: Dict[str, Any] = {}
    if hasattr(method, "export_diagnostics"):
        exported = method.export_diagnostics()
        if isinstance(exported, dict):
            diagnostics = exported
    diagnostics["timings"] = timings

    saved_paths = _save_results(cfg, metrics, predictions, labels, val_data, timings, diagnostics=diagnostics)

    return {
        "metrics": metrics,
        "timings": timings,
        "saved_paths": saved_paths,
    }
