from .base import MethodBase
from .frozen_feature import RidgeProbeMethod, WhitenedNCMMethod
from .i2cl import I2CLMethod
from .keco import KecoMethod
from .mimic import MimICMethod
from .rse import RSEMethod
from .rsev2 import RSEV2Method
from .sav import SAVMethod
from .stv import STVMethod
from .zero_shot import ZeroShotMethod

METHOD_REGISTRY = {
    "zero_shot": ZeroShotMethod,
    "sav": SAVMethod,
    "rse": RSEMethod,
    "rsev2": RSEV2Method,
    "whitened_ncm": WhitenedNCMMethod,
    "ridge_probe": RidgeProbeMethod,
    "keco": KecoMethod,
    "i2cl": I2CLMethod,
    "mimic": MimICMethod,
    "mimcl": MimICMethod,
    "stv": STVMethod,
}


def build_method(method_name: str, **kwargs) -> MethodBase:
    if method_name not in METHOD_REGISTRY:
        supported = ", ".join(sorted(METHOD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown method '{method_name}'. Supported methods: {supported}"
        )
    return METHOD_REGISTRY[method_name](**kwargs)


__all__ = [
    "MethodBase",
    "RidgeProbeMethod",
    "WhitenedNCMMethod",
    "I2CLMethod",
    "KecoMethod",
    "MimICMethod",
    "RSEMethod",
    "RSEV2Method",
    "SAVMethod",
    "STVMethod",
    "ZeroShotMethod",
    "METHOD_REGISTRY",
    "build_method",
]
