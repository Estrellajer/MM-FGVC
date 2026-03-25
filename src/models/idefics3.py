"""
Idefics3 视觉语言模型封装。
基于 ModelBase，使用 apply_chat_template + processor(text, images) 两段式处理。
官方文档: https://huggingface.co/docs/transformers/model_doc/idefics3
"""

import re
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image

from transformers import AutoModelForImageTextToText, AutoProcessor

from .model_base import ModelBase

HF_IDEFICS3 = ["Idefics3-8B-Llama3", "Idefics3-8B-Llama3.1"]
IMAGE_PLACEHOLDER_RE = re.compile(r"<image(?:_\d+)?>", re.IGNORECASE)


class Idefics3(ModelBase):
    """Idefics3 视觉语言模型。"""

    def __init__(
        self,
        model_root: str,
        processor_class=AutoProcessor,
        model_class=AutoModelForImageTextToText,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        import torch

        model_args = model_args or {}
        model_args.setdefault("dtype", torch.bfloat16)
        model_args.setdefault("device_map", "auto")

        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=HF_IDEFICS3,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

    @staticmethod
    def _sanitize_text(text: str) -> str:
        cleaned_lines = [
            line
            for line in str(text).splitlines()
            if not IMAGE_PLACEHOLDER_RE.fullmatch(line.strip())
        ]
        cleaned = "\n".join(cleaned_lines)
        cleaned = IMAGE_PLACEHOLDER_RE.sub("", cleaned)
        return cleaned.strip()

    def _build_messages(self, images: List[Image], text: str) -> List[Dict]:
        """构建 Idefics3 的 messages：content 中 image 用占位符，images 单独传递。"""
        sanitized_text = self._sanitize_text(text) or "Describe this image."
        content = [{"type": "image"}] * len(images) + [{"type": "text", "text": sanitized_text}]
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
        """
        Idefics3 两段式：apply_chat_template 生成 prompt -> processor(text=prompt, images=images)。
        """
        # 统一为 batch
        if isinstance(text, str):
            text = [text]
            images = [images] if isinstance(images[0], Image) else [images]
        if isinstance(text[0], dict):
            text = [text]
            if isinstance(images[0], Image):
                images = [images]
        if isinstance(images[0], Image):
            images = [images]

        all_prompts = []
        all_images = []  # List[List[Image]]，每个元素对应一个样本的图片列表
        for t, imgs in zip(text, images):
            imgs = imgs if isinstance(imgs, list) else [imgs]
            if isinstance(t, list) and len(t) > 0 and isinstance(t[0], dict):
                # 对话格式
                content = []
                for msg in t:
                    if msg.get("role") == "user" and "content" in msg:
                        inner = msg["content"] if isinstance(msg["content"], list) else [msg["content"]]
                        for c in inner:
                            if isinstance(c, dict):
                                if c.get("type") == "image":
                                    content.append({"type": "image"})
                                elif c.get("type") == "text":
                                    sanitized_text = self._sanitize_text(str(c.get("text", "")))
                                    if sanitized_text:
                                        content.append({"type": "text", "text": sanitized_text})
                        break
                if not any(c.get("type") == "text" for c in content):
                    content.append({"type": "text", "text": "Describe this image."})
                msgs = [{"role": "user", "content": content}]
            else:
                txt = t[0] if isinstance(t, list) else t
                msgs = self._build_messages(imgs, txt)

            prompt = self.processor.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=False,
            )
            all_prompts.append(prompt)
            all_images.append(imgs)

        # Idefics3: processor(text=prompts, images=images)，支持 batch
        inputs = self.processor(
            text=all_prompts,
            images=all_images,
            padding=kwargs.pop("padding", True),
            return_tensors=kwargs.pop("return_tensors", "pt"),
            **kwargs,
        )
        return inputs
