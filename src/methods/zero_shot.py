from __future__ import annotations

import re
from typing import Dict, List, Sequence

from ..data import build_prompt
from .base import MethodBase


class ZeroShotMethod(MethodBase):
    def __init__(
        self,
        model,
        dataset_name: str,
        label_space: Sequence[str],
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 0.0,
    ):
        super().__init__(model=model, dataset_name=dataset_name, label_space=label_space)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _match_label(self, output: str) -> str:
        output_norm = self._normalize_text(output)
        label_map = {self._normalize_text(label): label for label in self.label_space}

        if output_norm in label_map:
            return label_map[output_norm]

        for norm_label, raw_label in label_map.items():
            if output_norm.startswith(norm_label) or norm_label in output_norm:
                return raw_label

        if self.label_space:
            short_map = {
                self._normalize_text(label).split(" ")[0]: label
                for label in self.label_space
                if label.strip()
            }
            first_token = output_norm.split(" ")[0] if output_norm else ""
            if first_token in short_map:
                return short_map[first_token]

        return output.strip()

    def predict(self, sample: Dict) -> str:
        images = self._load_images(sample)
        prompt = build_prompt(self.dataset_name, sample)
        outputs = self.model.generate(
            images,
            [prompt],
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )
        raw = outputs[0] if isinstance(outputs, list) else str(outputs)
        return self._match_label(raw)
