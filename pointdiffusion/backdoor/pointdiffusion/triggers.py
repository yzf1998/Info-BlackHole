
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from src.backdoor.triggers.sphere import SphereTrigger
from src.backdoor.triggers.WLT import WLT

from ..tools.IBA_diffusion import PointDiffusionIBATrigger

__all__ = ['Trigger', 'AVAILABLE_TRIGGERS', 'build_trigger']

Applier = Callable[[np.ndarray], np.ndarray]

AVAILABLE_TRIGGERS = ('sphere', 'wlt', 'iba')


@dataclass(frozen=True)
class Trigger:

    pre: Optional[Applier] = None
    post: Optional[Applier] = None


def build_trigger(trigger_type: str, num_points: int,
                  pretrained_ckpt_path: Optional[str] = None,
                  device: str = 'cuda') -> Trigger:
    if trigger_type == 'sphere':
        trigger = SphereTrigger()
        return Trigger(post=lambda pts: trigger(pts)[1])

    if trigger_type == 'wlt':
        trigger = WLT({})
        return Trigger(pre=lambda pts: trigger(pts)[1])

    if trigger_type == 'iba':
        if pretrained_ckpt_path is None:
            raise ValueError(
                "trigger 'iba' requires --pretrained_ckpt_path (a clean "
                'pointdiffusion AE checkpoint), but got None.'
            )
        trigger = PointDiffusionIBATrigger(
            pretrained_ckpt_path=pretrained_ckpt_path,
            num_points=num_points,
            device=device,
        )
        return Trigger(post=lambda pts: trigger(pts)[1])

    raise ValueError(
        f'Unknown trigger_type: {trigger_type!r}. '
        f'Available: {list(AVAILABLE_TRIGGERS)}'
    )
