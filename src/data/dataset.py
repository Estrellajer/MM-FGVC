from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from paths import remap_path


SUPPORTED_DATASETS = {
    "general",
    "natural_ret",
    "naturalbench_ret",
    "naturalbench_vqa",
    "sugarcrepe",
    "pets",
    "eurosat",
    "flowers",
    "cub",
    "tinyimage",
    "vizwiz",
    "vlguard",
    "mhalubench",
    "blink",
}

YES_NO_DATASETS = {
    "natural_ret",
    "naturalbench_ret",
    "naturalbench_vqa",
    "sugarcrepe",
}

CLASS_NAME_DATASETS = {
    "pets",
    "eurosat",
    "flowers",
    "cub",
    "tinyimage",
}


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in JSON file: {path}")
    return data


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no} in {path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"JSONL line {line_no} in {path} should be a JSON object"
                )
            records.append(item)
    return records


def _dataset_key(dataset_name: str) -> str:
    return str(dataset_name).strip().lower().replace("-", "_")


def _first_present_text(item: Dict[str, Any], keys: List[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _collect_images(item: Dict[str, Any]) -> List[str]:
    collected: List[str] = []

    images_value = item.get("images")
    if isinstance(images_value, (list, tuple)):
        for image_path in images_value:
            if image_path:
                collected.append(remap_path(str(image_path)))

    for key in ["image", "image_path"]:
        value = item.get(key)
        if value:
            collected.append(remap_path(str(value)))

    numbered_image_keys = sorted(
        [
            key
            for key in item.keys()
            if re.fullmatch(r"image_\d+", key)
        ],
        key=lambda key: int(key.split("_", maxsplit=1)[1]),
    )
    for key in numbered_image_keys:
        value = item.get(key)
        if value:
            collected.append(remap_path(str(value)))

    deduped: List[str] = []
    seen = set()
    for image_path in collected:
        if image_path not in seen:
            deduped.append(image_path)
            seen.add(image_path)
    return deduped


def _normalize_record(item: Dict[str, Any], dataset_name: str, index: int) -> Dict[str, Any]:
    question = _first_present_text(item, ["question", "instruction", "claim"])
    if question is None:
        raise ValueError(
            f"Missing question-like field at sample index {index} for dataset '{dataset_name}'. "
            "Expected one of: question, instruction, claim."
        )

    if "label" not in item:
        raise ValueError(
            f"Missing field ['label'] at sample index {index} for dataset '{dataset_name}'"
        )

    images = _collect_images(item)
    if not images:
        raise ValueError(
            f"Missing image-like field at sample index {index} for dataset '{dataset_name}'. "
            "Expected one of: image, image_path, images, image_1..."
        )

    normalized = {
        "image": images[0],
        "images": images,
        "question": question,
        "label": str(item["label"]).strip(),
        "question_id": item.get("question_id", index),
        "split": item.get("split"),
    }

    alias_keys = {"image", "images", "image_path", "question", "instruction", "claim"}
    alias_keys.update({key for key in item.keys() if re.fullmatch(r"image_\d+", key)})
    extra = {
        key: value
        for key, value in item.items()
        if key not in alias_keys | {"label", "question_id", "split"}
    }
    normalized["extra"] = extra
    return normalized


def build_prompt(dataset_name: str, item: Dict[str, Any]) -> str:
    question = str(item["question"]).strip()
    dataset_key = _dataset_key(dataset_name)

    if dataset_key in YES_NO_DATASETS:
        if "yes or no" in question.lower():
            return question
        return f"{question} Answer with Yes or No."
    if dataset_key in CLASS_NAME_DATASETS:
        return f"{question} Answer with the class name."
    if dataset_key == "vlguard":
        return (
            f"Instruction: {question}\n"
            "Classify the instruction as harmful or unharmful. "
            "Answer with harmful or unharmful."
        )
    if dataset_key == "mhalubench":
        return (
            f"Claim: {question}\n"
            "Decide whether the claim is hallucination or non-hallucination. "
            "Answer with hallucination or non-hallucination."
        )
    if dataset_key == "blink" or dataset_key.startswith("blink_"):
        return question
    return question


def load_dataset(dataset_name: str, path: str) -> List[Dict[str, Any]]:
    dataset_name = str(dataset_name).strip()
    if not dataset_name:
        raise ValueError("dataset_name must be a non-empty string")

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        records = _load_jsonl(dataset_path)
    elif suffix == ".json":
        records = _load_json(dataset_path)
    else:
        raise ValueError(
            f"Unsupported dataset file suffix '{suffix}' for {dataset_path}. Use .json or .jsonl"
        )

    return [
        _normalize_record(item, dataset_name=dataset_name, index=index)
        for index, item in enumerate(records)
    ]


def load_train_val(
    dataset_name: str,
    train_path: str,
    val_path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_data = load_dataset(dataset_name, train_path)
    val_data = load_dataset(dataset_name, val_path)
    return train_data, val_data


def collect_label_space(*datasets: Iterable[Dict[str, Any]]) -> List[str]:
    labels = []
    seen = set()
    for dataset in datasets:
        for item in dataset:
            label = str(item["label"]).strip()
            if label not in seen:
                labels.append(label)
                seen.add(label)
    return labels
