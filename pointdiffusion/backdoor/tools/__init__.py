"""Trigger implementations specific to the pointdiffusion autoencoder.

The sphere and WLT triggers are shared with the FoldingNet branch and imported
from ``src.backdoor.triggers``; only the IBA variant is reimplemented here.
"""

from .IBA_diffusion import PointDiffusionIBATrigger

__all__ = ["PointDiffusionIBATrigger"]
