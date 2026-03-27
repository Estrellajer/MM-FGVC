from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Sequence

from PIL import Image

_INLINE_IMAGE_RE = re.compile(r"<image(?:_\d+)?>", re.IGNORECASE)


class MethodBase:
    def __init__(self, model, dataset_name: str, label_space: Sequence[str]):
        self.model = model
        self.dataset_name = dataset_name
        self.label_space = list(label_space)

    def fit(self, train_data: Sequence[Dict]) -> None:
        return None

    def _iter_image_paths(self, sample: Dict) -> List[str]:
        image_paths = sample.get("images")
        if isinstance(image_paths, list) and image_paths:
            return [str(image_path) for image_path in image_paths]

        single_image = sample.get("image")
        if single_image:
            return [str(single_image)]

        raise KeyError("Sample is missing 'image' or 'images'")

    def _load_images(self, sample: Dict) -> List[Image.Image]:
        images: List[Image.Image] = []
        for image_path in self._iter_image_paths(sample):
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            images.append(Image.open(path).convert("RGB"))
        return images

    @staticmethod
    def _sanitize_prompt_text_for_explicit_images(text: str) -> str:
        cleaned_lines = [
            line
            for line in str(text).splitlines()
            if not _INLINE_IMAGE_RE.fullmatch(line.strip())
        ]
        cleaned = "\n".join(cleaned_lines)
        cleaned = _INLINE_IMAGE_RE.sub("", cleaned)
        return cleaned.strip()

    @staticmethod
    def _is_vision_backbone_module_name(module_name: str) -> bool:
        lowered = str(module_name).lower()
        if lowered.startswith(("vision_model.", "vision_tower.", "vision_encoder.")):
            return True
        return any(
            token in lowered
            for token in (".vision_model.", ".vision_tower.", ".vision_encoder.")
        )

    def _filter_text_backbone_module_names(self, module_names: Sequence[str]) -> List[str]:
        filtered = [
            str(module_name)
            for module_name in module_names
            if not self._is_vision_backbone_module_name(str(module_name))
        ]
        return filtered if filtered else [str(module_name) for module_name in module_names]

    def predict(self, sample: Dict) -> str:
        raise NotImplementedError

    def predict_many(self, samples: Sequence[Dict]) -> List[str]:
        return [self.predict(sample) for sample in samples]
