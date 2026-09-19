"""Reconstruction and distribution losses.

Submodules are imported lazily (PEP 562). ``EMD`` JIT-compiles a CUDA kernel on
first import, so eager imports here would make ``src.losses.adaptive_mmd`` --
pure PyTorch and CPU-friendly -- unusable on a machine without a visible GPU.
"""

from typing import Any, List

_LAZY_ATTRS = {
    'Chamfer': '.chamfer',
    'EMD': '.emd',
    'ASW': '.sw_variants',
    'SWD': '.sw_variants',
    'GenSW': '.sw_variants',
    'MaxSW': '.sw_variants',
    'adaptive_mmd_loss': '.adaptive_mmd',
    'AdaptiveMMDLoss': '.adaptive_mmd',
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    from importlib import import_module
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return __all__
