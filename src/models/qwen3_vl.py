"""
Qwen3-VL 视觉语言模型封装。
基于 ModelBase，支持 Qwen3-VL 系列的 image + text 输入与生成。
官方文档: https://huggingface.co/docs/transformers/model_doc/qwen3_vl
"""

from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .model_base import ModelBase

HF_QWEN3_VL = ["Qwen3-VL-8B-Instruct", "Qwen3-VL-4B-Instruct", "Qwen3-VL-2B-Instruct"]


class Qwen3VL(ModelBase):
    """Qwen3-VL 视觉语言模型，使用 apply_chat_template 处理多模态输入。"""

    def __init__(
        self,
        model_root: str,
        processor_class=AutoProcessor,
        model_class=Qwen3VLForConditionalGeneration,
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
            support_models=HF_QWEN3_VL,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

    def _build_messages(self, images: List[Image], text: Union[str, List[Dict]]) -> List[Dict]:
        """将 (images, text) 转为 Qwen3-VL 的 messages 格式。"""
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
        """
        构建 messages 格式并调用 processor.apply_chat_template。
        支持: text 为字符串或 [{"role":"user","content":[...]}]，images 为 List[Image] 或 List[List[Image]]。
        """
        # 统一为 batch: List[(images, text)]
        if isinstance(text, str):
            text = [text]
            images = [images] if isinstance(images[0], Image) else images
        if isinstance(text[0], dict):
            text = [text]
            if isinstance(images[0], Image):
                images = [images]
        if isinstance(images[0], Image):
            images = [images]  # 单样本 -> batch of 1

        batch_messages = []
        for t, imgs in zip(text, images):
            imgs = imgs if isinstance(imgs, list) else [imgs]
            if isinstance(t, list) and len(t) > 0 and isinstance(t[0], dict):
                # 对话格式：取首个 user 的 content，注入 images
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
