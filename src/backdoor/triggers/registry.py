
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from src.backdoor.triggers.sphere import SphereTrigger
from src.backdoor.triggers.WLT import WLT


__all__ = [
    "BuildContext",
    "Trigger",
    "TRIGGER_REGISTRY",
    "register_trigger",
    "build_trigger",
    "available_triggers",
]


@dataclass(frozen=True)
class BuildContext:

    dataset_root: str
    dataset_name: str = "shapenet"
    num_points: int = 2048
    pretrained_ckpt_path: Optional[str] = None
    cache_dir: Optional[str] = None
    m: int = 2025


Applier = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class Trigger:

    pre: Optional[Applier] = None
    post: Optional[Applier] = None


TRIGGER_REGISTRY: Dict[str, Callable[[BuildContext], Trigger]] = {}


def register_trigger(name: str) -> Callable[[Callable[[BuildContext], Trigger]],
                                           Callable[[BuildContext], Trigger]]:

    def decorator(builder: Callable[[BuildContext], Trigger]) -> Callable[[BuildContext], Trigger]:
        if name in TRIGGER_REGISTRY:
            raise ValueError(f"Duplicate trigger registration: {name!r}")
        TRIGGER_REGISTRY[name] = builder
        return builder

    return decorator


def available_triggers() -> list:
    return sorted(TRIGGER_REGISTRY)


def build_trigger(name: str, ctx: BuildContext) -> Trigger:
    try:
        builder = TRIGGER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown trigger_type: {name!r}. Available: {available_triggers()}"
        ) from None
    return builder(ctx)


@register_trigger("sphere")
def _build_sphere(ctx: BuildContext) -> Trigger:
    trigger = SphereTrigger()
    return Trigger(post=lambda pts, label: trigger(pts)[1])


@register_trigger("wlt")
def _build_wlt(ctx: BuildContext) -> Trigger:
    trigger = WLT({})
    return Trigger(pre=lambda pts, label: trigger(pts)[1])


@register_trigger("iba")
def _build_iba(ctx: BuildContext) -> Trigger:
    if ctx.pretrained_ckpt_path is None:
        raise ValueError(
            "trigger 'iba' requires 'dataset_pretrain' (pretrained_ckpt_path) "
            "to be set in the config, but got None."
        )
    from src.backdoor.triggers.IBA import IBAtrigger

    trigger = IBAtrigger(
        pretrained_ckpt_path=ctx.pretrained_ckpt_path,
        num_points=ctx.num_points,
        m=ctx.m,
    )
    return Trigger(post=lambda pts, label: trigger(pts)[1])
