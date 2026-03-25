from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import math
import os
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

from .base import MethodBase


_CHOICE_LETTERS = string.ascii_uppercase
_CLASSIFICATION_DATASETS = {"pets", "eurosat", "flowers", "cub", "tinyimage"}


@dataclass
class _KecoRecord:
    sample_id: str
    label: str
    question: str
    image_path: str
    feature: torch.Tensor
    options: List[str]


class KecoMethod(MethodBase):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        mode: str = "offline",
        support_per_class: int = 2,
        num_shots: int = 2,
        num_choices: int = 4,
        selection_strategy: str = "cosine",
        sample_method: str = "random",
        target_select: str = "least_similarity",
        alpha: float = 0.2,
        offline_epochs: int = 10,
        offline_batch_size: int = 32,
        embedding_model_name: str = "openai/clip-vit-large-patch14-336",
        embedding_device: str = "auto",
        embedding_batch_size: int = 8,
        max_new_tokens: int = 8,
        do_sample: bool = False,
        temperature: float = 0.0,
        seed: int = 42,
        progress_bar: bool = True,
    ):
        super().__init__(model=model, dataset_name=dataset_name, label_space=label_space)
        self.mode = str(mode).strip().lower()
        self.support_per_class = int(support_per_class)
        self.num_shots = int(num_shots)
        self.num_choices = int(num_choices)
        self.selection_strategy = str(selection_strategy).strip().lower()
        self.sample_method = str(sample_method).strip().lower()
        self.target_select = str(target_select).strip().lower()
        self.alpha = float(alpha)
        self.offline_epochs = int(offline_epochs)
        self.offline_batch_size = int(offline_batch_size)
        self.embedding_model_name = str(embedding_model_name).strip()
        self.embedding_device = str(embedding_device).strip().lower()
        self.embedding_batch_size = int(embedding_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.progress_bar = bool(progress_bar)

        if self.mode not in {"fewshot", "offline", "online"}:
            raise ValueError("KeCO mode must be one of: fewshot, offline, online")
        if self.support_per_class <= 0:
            raise ValueError("KeCO support_per_class must be positive")
        if self.num_shots < 0:
            raise ValueError("KeCO num_shots must be non-negative")
        if self.num_choices < 2 or self.num_choices > len(_CHOICE_LETTERS):
            raise ValueError(f"KeCO num_choices must be in [2, {len(_CHOICE_LETTERS)}]")
        if self.selection_strategy not in {"cosine", "diversity", "random"}:
            raise ValueError("KeCO selection_strategy must be one of: cosine, diversity, random")
        if self.sample_method not in {"random", "k_center_greedy"}:
            raise ValueError("KeCO sample_method must be one of: random, k_center_greedy")
        if self.target_select not in {"least_similarity", "most_similarity", "random"}:
            raise ValueError("KeCO target_select must be one of: least_similarity, most_similarity, random")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("KeCO alpha must be in [0, 1]")
        if self.offline_epochs <= 0:
            raise ValueError("KeCO offline_epochs must be positive")
        if self.offline_batch_size <= 0:
            raise ValueError("KeCO offline_batch_size must be positive")
        if self.embedding_batch_size <= 0:
            raise ValueError("KeCO embedding_batch_size must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("KeCO max_new_tokens must be positive")

        self._image_cache: Dict[str, Image.Image] = {}
        self._embedding_model = None
        self._embedding_processor = None
        self._embedding_torch_device = None

        self.train_records: List[_KecoRecord] = []
        self.support_records: List[_KecoRecord] = []
        self.pool_records: List[_KecoRecord] = []
        self.support_indices_by_label: Dict[str, List[int]] = {}
        self.eval_count = 0
        self.parse_failures = 0
        self._fitted = False

    def _dataset_key(self) -> str:
        return str(self.dataset_name).strip().lower().replace("-", "_")

    def _check_dataset_support(self) -> None:
        if self._dataset_key() not in _CLASSIFICATION_DATASETS:
            supported = ", ".join(sorted(_CLASSIFICATION_DATASETS))
            raise ValueError(
                f"KeCO is currently wired as an image-classification baseline only. "
                f"Unsupported dataset '{self.dataset_name}'. Supported datasets: {supported}"
            )

    def _resolve_embedding_device(self) -> torch.device:
        if self.embedding_device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(self.embedding_device)

    @staticmethod
    @contextmanager
    def _without_socks_proxy():
        from huggingface_hub.utils._http import close_session

        removed: Dict[str, str] = {}
        for key in ("ALL_PROXY", "all_proxy"):
            value = os.environ.pop(key, None)
            if value is not None:
                removed[key] = value
        close_session()
        try:
            yield
        finally:
            close_session()
            for key, value in removed.items():
                os.environ[key] = value

    def _iter_with_progress(self, items: Sequence, desc: str):
        if not self.progress_bar:
            return items
        try:
            from tqdm import tqdm
        except Exception:
            return items
        return tqdm(items, desc=desc)

    def _stable_int(self, *parts: object) -> int:
        joined = "||".join(str(part) for part in parts)
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _load_image_path(self, image_path: str) -> Image.Image:
        cached = self._image_cache.get(image_path)
        if cached is not None:
            return cached.copy()

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as img:
            loaded = img.convert("RGB").copy()
        self._image_cache[image_path] = loaded
        return loaded.copy()

    def _ensure_embedding_components(self) -> None:
        if self._embedding_model is not None and self._embedding_processor is not None:
            return

        self._embedding_torch_device = self._resolve_embedding_device()
        try:
            with self._without_socks_proxy():
                try:
                    self._embedding_processor = AutoProcessor.from_pretrained(
                        self.embedding_model_name,
                        local_files_only=True,
                    )
                    self._embedding_model = CLIPModel.from_pretrained(
                        self.embedding_model_name,
                        local_files_only=True,
                        use_safetensors=False,
                    )
                except Exception:
                    self._embedding_processor = AutoProcessor.from_pretrained(self.embedding_model_name)
                    self._embedding_model = CLIPModel.from_pretrained(
                        self.embedding_model_name,
                        use_safetensors=False,
                    )
        except Exception as inner_exc:
            message = str(inner_exc)
            if "socksio" in message.lower():
                raise RuntimeError(
                    "KeCO failed to load the embedding model because the environment is using a "
                    "SOCKS proxy without the optional 'socksio' dependency, and no local cache "
                    "was available. Retry with ALL_PROXY= all_proxy=, install socksio, or point "
                    "embedding_model_name to a local checkpoint."
                ) from inner_exc
            raise
        self._embedding_model.eval()
        self._embedding_model.requires_grad_(False)
        self._embedding_model.to(self._embedding_torch_device)

    @staticmethod
    def _coerce_feature_tensor(features: torch.Tensor | object) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features

        pooler_output = getattr(features, "pooler_output", None)
        if isinstance(pooler_output, torch.Tensor):
            return pooler_output

        image_embeds = getattr(features, "image_embeds", None)
        if isinstance(image_embeds, torch.Tensor):
            return image_embeds

        raise TypeError(
            "KeCO expected CLIP image features to be a tensor or a model output with "
            "pooler_output/image_embeds."
        )

    def _extract_features(self, image_paths: Sequence[str]) -> torch.Tensor:
        self._ensure_embedding_components()
        assert self._embedding_processor is not None
        assert self._embedding_model is not None
        assert self._embedding_torch_device is not None

        all_features: List[torch.Tensor] = []
        for start in range(0, len(image_paths), self.embedding_batch_size):
            batch_paths = image_paths[start : start + self.embedding_batch_size]
            images = [self._load_image_path(path) for path in batch_paths]
            inputs = self._embedding_processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self._embedding_torch_device) for key, value in inputs.items()}
            with torch.no_grad():
                features = self._embedding_model.get_image_features(**inputs)
            features = self._coerce_feature_tensor(features)
            features = F.normalize(features.to(dtype=torch.float32), dim=-1).cpu()
            all_features.append(features)
        return torch.cat(all_features, dim=0) if all_features else torch.empty((0, 0), dtype=torch.float32)

    def _build_candidate_options(self, sample: Dict[str, str]) -> List[str]:
        extra_options = (sample.get("extra") or {}).get("options")
        label = str(sample["label"]).strip()
        distractor_pool = [candidate for candidate in self.label_space if candidate != label]

        if isinstance(extra_options, list):
            normalized = []
            seen = {label}
            for option in extra_options:
                option_str = str(option).strip()
                if not option_str or option_str in seen:
                    continue
                normalized.append(option_str)
                seen.add(option_str)
            if normalized:
                return [label] + normalized[: max(self.num_choices - 1, 0)]

        rng = random.Random(self.seed + self._stable_int(sample.get("question_id"), sample.get("image"), label))
        if len(distractor_pool) <= self.num_choices - 1:
            distractors = list(distractor_pool)
        else:
            distractors = rng.sample(distractor_pool, self.num_choices - 1)
        return [label] + distractors

    def _prepare_mc_prompt(
        self,
        question: str,
        options: Sequence[str],
        sample_id: str,
        include_answer: bool,
    ) -> tuple[str, str | None, Dict[str, str]]:
        if not options:
            raise ValueError("KeCO requires non-empty options")

        shuffle_rng = random.Random(self.seed + self._stable_int("prompt", sample_id))
        shuffle_indices = list(range(len(options)))
        shuffle_rng.shuffle(shuffle_indices)

        gold_letter = _CHOICE_LETTERS[shuffle_indices.index(0)]
        label2option = {
            _CHOICE_LETTERS[pos]: str(options[option_idx])
            for pos, option_idx in enumerate(shuffle_indices)
        }
        formatted_options = "\n".join(
            f"{letter}. {label2option[letter]}"
            for letter in _CHOICE_LETTERS[: len(shuffle_indices)]
        )
        base_prompt = (
            f"{question.strip()}\nChoices:\n{formatted_options}\n"
            "Answer with the letter from the given choices directly."
        )
        if include_answer:
            return f"{base_prompt}{gold_letter}.", gold_letter, label2option
        return base_prompt, None, label2option

    def _build_record(self, sample: Dict, feature: torch.Tensor) -> _KecoRecord:
        sample_id = str(sample.get("question_id", self._stable_int(sample.get("image"), sample["label"])))
        options = self._build_candidate_options(sample)
        return _KecoRecord(
            sample_id=sample_id,
            label=str(sample["label"]).strip(),
            question=str(sample["question"]).strip(),
            image_path=str(sample["image"]),
            feature=F.normalize(feature.to(dtype=torch.float32), dim=0).cpu(),
            options=options,
        )

    def _build_records(self, data: Sequence[Dict], desc: str) -> List[_KecoRecord]:
        image_paths = [str(item["image"]) for item in data]
        features = self._extract_features(image_paths)
        return [
            self._build_record(sample, feature)
            for sample, feature in zip(self._iter_with_progress(data, desc=desc), features)
        ]

    def _select_initial_support(self, records: Sequence[_KecoRecord], rng: random.Random) -> tuple[List[_KecoRecord], List[_KecoRecord]]:
        if not records:
            return [], []

        support_size = min(self.support_per_class, len(records))
        if support_size >= len(records):
            return list(records), []

        if self.sample_method == "random":
            selected_indices = sorted(rng.sample(range(len(records)), support_size))
        else:
            features = torch.stack([record.feature for record in records], dim=0)
            selected_indices = [0]
            min_distances = torch.cdist(features, features[[0]], p=2).squeeze(-1)
            while len(selected_indices) < support_size:
                masked = min_distances.clone()
                masked[selected_indices] = -1.0
                next_idx = int(masked.argmax().item())
                selected_indices.append(next_idx)
                candidate_distances = torch.cdist(features, features[[next_idx]], p=2).squeeze(-1)
                min_distances = torch.minimum(min_distances, candidate_distances)
            selected_indices = sorted(selected_indices)

        selected = [records[idx] for idx in selected_indices]
        selected_set = set(selected_indices)
        remaining = [record for idx, record in enumerate(records) if idx not in selected_set]
        return selected, remaining

    def _rebuild_support_index(self) -> None:
        support_indices_by_label: Dict[str, List[int]] = defaultdict(list)
        for idx, record in enumerate(self.support_records):
            support_indices_by_label[record.label].append(idx)
        self.support_indices_by_label = dict(support_indices_by_label)

    def _select_target_support_index(
        self,
        label: str,
        query_feature: torch.Tensor,
        rng: random.Random,
    ) -> int:
        candidate_indices = self.support_indices_by_label.get(label, [])
        if not candidate_indices:
            raise ValueError(f"KeCO found no support records for label '{label}'")

        if self.target_select == "random":
            return rng.choice(candidate_indices)

        support_features = torch.stack([self.support_records[idx].feature for idx in candidate_indices], dim=0)
        scores = support_features @ query_feature.unsqueeze(-1)
        scores = scores.squeeze(-1)
        offset = int(scores.argmin().item()) if self.target_select == "least_similarity" else int(scores.argmax().item())
        return candidate_indices[offset]

    def _update_online_support(self) -> None:
        if not self.pool_records:
            return

        rng = random.Random(self.seed + 11)
        pool = list(self.pool_records)
        rng.shuffle(pool)
        for record in self._iter_with_progress(pool, desc="KeCO online update"):
            target_idx = self._select_target_support_index(record.label, record.feature, rng)
            updated = (1.0 - self.alpha) * self.support_records[target_idx].feature + self.alpha * record.feature
            self.support_records[target_idx].feature = F.normalize(updated, dim=0).cpu()

    def _update_offline_support(self) -> None:
        if not self.pool_records:
            return

        rng = random.Random(self.seed + 23)
        pool = list(self.pool_records)
        rng.shuffle(pool)

        batches = [
            pool[start : start + self.offline_batch_size]
            for start in range(0, len(pool), self.offline_batch_size)
        ]
        for _ in self._iter_with_progress(range(self.offline_epochs), desc="KeCO offline epochs"):
            for batch in batches:
                grouped: Dict[int, List[torch.Tensor]] = defaultdict(list)
                for record in batch:
                    target_idx = self._select_target_support_index(record.label, record.feature, rng)
                    grouped[target_idx].append(record.feature)

                for target_idx, pool_features in grouped.items():
                    target_feature = self.support_records[target_idx].feature
                    stacked = torch.stack(pool_features, dim=0)
                    gradient = (target_feature.unsqueeze(0) - stacked).mean(dim=0)
                    updated = target_feature - self.alpha * gradient
                    self.support_records[target_idx].feature = F.normalize(updated, dim=0).cpu()

    def _select_demonstrations(self, query_record: _KecoRecord) -> List[_KecoRecord]:
        if self.num_shots <= 0 or not self.support_records:
            return []

        demo_count = min(self.num_shots, len(self.support_records))
        rng = random.Random(self.seed + self._stable_int("demo", query_record.sample_id))
        if self.selection_strategy == "random":
            indices = rng.sample(range(len(self.support_records)), demo_count)
            return [self.support_records[idx] for idx in indices]

        support_features = torch.stack([record.feature for record in self.support_records], dim=0)
        scores = support_features @ query_record.feature.unsqueeze(-1)
        scores = scores.squeeze(-1)
        largest = self.selection_strategy == "cosine"
        indices = torch.topk(scores, k=demo_count, largest=largest).indices.tolist()
        return [self.support_records[int(idx)] for idx in indices]

    def _generate_from_interleaved_content(self, content: List[Dict[str, object]]) -> str:
        if not hasattr(self.model.processor, "apply_chat_template"):
            raise RuntimeError("KeCO currently requires a processor with apply_chat_template support")

        messages = [{"role": "user", "content": content}]
        inputs = self.model.processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(device=self.model.device, dtype=self.model.model.dtype)
        generate_args = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            generate_args["temperature"] = self.temperature
        generated_ids = self.model.model.generate(**inputs, **generate_args)
        seq_len = inputs.input_ids.shape[-1]
        generated_ids = generated_ids[:, seq_len:]
        outputs = self.model.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0] if outputs else ""

    def _extract_choice_letter(self, raw_output: str, valid_letters: Iterable[str]) -> str | None:
        valid = set(valid_letters)
        for match in re.finditer(r"\b([A-Z])\b", raw_output.upper()):
            candidate = match.group(1)
            if candidate in valid:
                return candidate

        compact = "".join(ch for ch in raw_output.upper() if ch.isalpha())
        if compact[:1] in valid:
            return compact[:1]
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _match_output_to_label(self, raw_output: str, label2option: Dict[str, str]) -> str:
        letter = self._extract_choice_letter(raw_output, label2option.keys())
        if letter is not None:
            return label2option[letter]

        raw_norm = self._normalize_text(raw_output)
        option_map = {self._normalize_text(label): label for label in label2option.values()}
        if raw_norm in option_map:
            return option_map[raw_norm]
        for option_norm, option in option_map.items():
            if raw_norm.startswith(option_norm) or option_norm in raw_norm:
                return option

        self.parse_failures += 1
        return raw_output.strip()

    def fit(self, train_data: Sequence[Dict]) -> None:
        self._check_dataset_support()
        if not train_data:
            raise ValueError("KeCO fit requires non-empty train_data")

        self.eval_count = 0
        self.parse_failures = 0
        self.train_records = self._build_records(train_data, desc="KeCO train embeddings")

        by_label: Dict[str, List[_KecoRecord]] = defaultdict(list)
        for record in self.train_records:
            by_label[record.label].append(record)

        rng = random.Random(self.seed)
        self.support_records = []
        self.pool_records = []
        for label in [label for label in self.label_space if label in by_label]:
            support, pool = self._select_initial_support(by_label[label], rng)
            self.support_records.extend(support)
            self.pool_records.extend(pool)

        self._rebuild_support_index()
        if self.mode == "online":
            self._update_online_support()
        elif self.mode == "offline":
            self._update_offline_support()

        self._rebuild_support_index()
        self._fitted = True

    def predict(self, sample: Dict) -> str:
        if not self._fitted:
            raise RuntimeError("KeCO method is not fitted. Call fit(train_data) first.")

        feature = self._extract_features([str(sample["image"])])[0]
        query_record = self._build_record(sample, feature)
        demos = self._select_demonstrations(query_record)

        content: List[Dict[str, object]] = []
        for demo in demos:
            demo_prompt, _, _ = self._prepare_mc_prompt(
                question=demo.question,
                options=demo.options,
                sample_id=f"train::{demo.sample_id}",
                include_answer=True,
            )
            content.append({"type": "image", "image": self._load_image_path(demo.image_path)})
            content.append({"type": "text", "text": demo_prompt})

        query_prompt, _, label2option = self._prepare_mc_prompt(
            question=query_record.question,
            options=query_record.options,
            sample_id=f"val::{query_record.sample_id}",
            include_answer=False,
        )
        content.append({"type": "image", "image": self._load_image_path(query_record.image_path)})
        content.append({"type": "text", "text": query_prompt})

        raw_output = self._generate_from_interleaved_content(content)
        self.eval_count += 1
        return self._match_output_to_label(raw_output, label2option)

    def export_diagnostics(self) -> Dict:
        if not self._fitted:
            return {}

        per_class_support = {
            label: len(indices)
            for label, indices in sorted(self.support_indices_by_label.items())
        }
        return {
            "method": "keco",
            "mode": self.mode,
            "embedding_model_name": self.embedding_model_name,
            "embedding_device": str(self._embedding_torch_device),
            "support_per_class": self.support_per_class,
            "num_shots": self.num_shots,
            "num_choices": self.num_choices,
            "selection_strategy": self.selection_strategy,
            "sample_method": self.sample_method,
            "target_select": self.target_select,
            "alpha": self.alpha,
            "offline_epochs": self.offline_epochs,
            "offline_batch_size": self.offline_batch_size,
            "train_size": len(self.train_records),
            "support_size": len(self.support_records),
            "pool_size": len(self.pool_records),
            "per_class_support_counts": per_class_support,
            "eval_summary": {
                "predictions": self.eval_count,
                "parse_failures": self.parse_failures,
            },
        }
