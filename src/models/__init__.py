from .idefics3 import Idefics3
from .qwen2 import Qwen2
from .qwen3_vl import Qwen3VL

try:
    from .llava import LLaVa
except Exception:
    LLaVa = None

try:
    from .idefics import Idefics
except Exception:
    Idefics = None

try:
    from .idefics2 import Idefics2
except Exception:
    Idefics2 = None

try:
    from .mistral import Mistral
except Exception:
    Mistral = None


MODEL_REGISTRY = {
    "qwen2_vl": Qwen2,
    "qwen3_vl": Qwen3VL,
    "idefics3": Idefics3,
}


def get_model_class(model_name: str):
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Supported models: {supported}"
        )
    return MODEL_REGISTRY[model_name]


__all__ = [
    "Qwen2",
    "Qwen3VL",
    "Idefics3",
    "get_model_class",
    "MODEL_REGISTRY",
]