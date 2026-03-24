from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image


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

    def predict(self, sample: Dict) -> str:
        raise NotImplementedError

    def predict_many(self, samples: Sequence[Dict]) -> List[str]:
        return [self.predict(sample) for sample in samples]
