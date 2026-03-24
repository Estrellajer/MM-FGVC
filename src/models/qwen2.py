from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from .model_base import ModelBase

HF_QWEN2_VL = ["Qwen2-VL-72B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2-VL-2B-Instruct"]


class Qwen2(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=Qwen2VLForConditionalGeneration,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        model_args = model_args or {}
        model_args.setdefault("dtype", "auto")
        model_args.setdefault("device_map", "auto")

        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=HF_QWEN2_VL,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

    def _build_messages(self, images: List[Image], text: Union[str, List[Dict]]) -> List[Dict]:
        content = [{"type": "image", "image": img} for img in images]
        if isinstance(text, str):
            content.append({"type": "text", "text": text})
        elif isinstance(text, list) and len(text) > 0:
            for c in text:
                if isinstance(c, dict) and c.get("type") == "text":
                    content.append(c)
                    break
            else:
                content.append({"type": "text", "text": "Describe this image."})
        else:
            content.append({"type": "text", "text": "Describe this image."})
        return [{"role": "user", "content": content}]

    def process_input(
        self,
        images: Union[List[Image], List[List[Image]]],
        text: Union[
            List[Union[str, Dict[str, Any]]],
            List[List[Union[str, Dict[str, Any]]]],
        ],
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(text, str):
            text = [text]
            images = [images] if isinstance(images[0], Image) else images
        if isinstance(text[0], dict):
            text = [text]
            if isinstance(images[0], Image):
                images = [images]
        if isinstance(images[0], Image):
            images = [images]

        batch_messages = []
        for t, imgs in zip(text, images):
            imgs = imgs if isinstance(imgs, list) else [imgs]
            if isinstance(t, list) and len(t) > 0 and isinstance(t[0], dict):
                content = [{"type": "image", "image": img} for img in imgs]
                for msg in t:
                    if msg.get("role") == "user" and "content" in msg:
                        inner = msg["content"] if isinstance(msg["content"], list) else [msg["content"]]
                        for c in inner:
                            if isinstance(c, dict) and c.get("type") == "text":
                                content.append(c)
                                break
                        else:
                            content.append({"type": "text", "text": "Describe this image."})
                        break
                batch_messages.append([{"role": "user", "content": content}])
            else:
                txt = t[0] if isinstance(t, list) else t
                batch_messages.append(self._build_messages(imgs, txt))

        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors=kwargs.pop("return_tensors", "pt"),
            padding=kwargs.pop("padding", True),
            **kwargs,
        )
        return inputs
