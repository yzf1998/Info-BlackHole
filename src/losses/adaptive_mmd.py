"""Adaptive Gaussian MMD regularizer for poisoned latent codes.

Implements the poison-only adaptive Gaussian feature matching objective of
Information Blackhole. Poisoned latent codes are centered before matching, so
the loss constrains only the *shape* of the poisoned latent distribution and
leaves its absolute position free. The reference Gaussian tracks a momentum
running estimate of the poisoned batch standard deviation.
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_KERNEL_SCALES: List[float] = [0.5, 1.0, 2.0]
MIN_STD = 1e-6
MIN_BASE_SIGMA = 1.0


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Multivariate Gaussian (RBF) kernel matrix between two sample sets."""
    dist_sq = torch.pow(x.unsqueeze(1) - y.unsqueeze(0), 2).sum(2)
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def adaptive_mmd_loss(
    source_features: torch.Tensor,
    running_std: torch.Tensor,
    device: torch.device,
    momentum: float = 0.99,
    is_first_step: bool = False,
    kernel_scales: List[float] = None,
) -> torch.Tensor:
    """Adaptive Gaussian MMD between poisoned latent codes and a reference prior.

    The reference distribution is a zero-mean Gaussian whose standard deviation
    follows a momentum-smoothed running estimate of the poisoned batch std.
    Source features are centered, so mean offsets are not penalized.

    Args:
        source_features: Poisoned latent codes, shape (B, D) or (B, ...).
        running_std: Mutable buffer holding the running std estimate. Updated
            in place. Expected to be a registered buffer on the model.
        device: Device on which reference samples are drawn.
        momentum: Momentum for the running std update.
        is_first_step: If True and the buffer is still at its init value of 1.0,
            the buffer is seeded from the current batch instead of blended.
        kernel_scales: Multi-scale kernel bandwidth multipliers.

    Returns:
        Scalar MMD loss.

    Raises:
        ValueError: If ``source_features`` has fewer than two dimensions.
    """
    if source_features.dim() < 2:
        raise ValueError(
            f"source_features must be at least 2-D (B, D), got shape {tuple(source_features.shape)}"
        )
    if kernel_scales is None:
        kernel_scales = DEFAULT_KERNEL_SCALES

    if source_features.dim() > 2:
        source_features = source_features.view(source_features.size(0), -1)
    batch_size, latent_dim = source_features.shape

    batch_mean = torch.mean(source_features, dim=0, keepdim=True)
    source_centered = source_features - batch_mean

    current_std = torch.std(source_centered, dim=0).mean().detach()
    current_std = torch.clamp(current_std, min=MIN_STD)
    if is_first_step and running_std.item() == 1.0:
        running_std.copy_(current_std)
    else:
        running_std.copy_(momentum * running_std + (1 - momentum) * current_std)

    # Both source projects draw on CPU before transferring to the model device.
    noise = torch.randn(batch_size, latent_dim).to(device)
    noise = noise - torch.mean(noise, dim=0, keepdim=True)
    target_samples = noise * running_std

    base_sigma = max(running_std.item(), MIN_BASE_SIGMA)
    mmd_loss = source_features.new_zeros(())
    for scale in kernel_scales:
        sigma = base_sigma * scale
        xx = _gaussian_kernel(source_centered, source_centered, sigma)
        yy = _gaussian_kernel(target_samples, target_samples, sigma)
        xy = _gaussian_kernel(source_centered, target_samples, sigma)
        mmd_loss = mmd_loss + (torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy))

    return mmd_loss


class AdaptiveMMDLoss(nn.Module):
    """Module wrapper around :func:`adaptive_mmd_loss`."""

    def __init__(self, momentum: float = 0.99, kernel_scales: List[float] = None):
        super().__init__()
        self.momentum = momentum
        self.kernel_scales = kernel_scales or DEFAULT_KERNEL_SCALES

    def forward(
        self,
        source_features: torch.Tensor,
        running_std: torch.Tensor,
        device: torch.device,
        is_first_step: bool = False,
    ) -> torch.Tensor:
        return adaptive_mmd_loss(
            source_features,
            running_std,
            device,
            momentum=self.momentum,
            is_first_step=is_first_step,
            kernel_scales=self.kernel_scales,
        )
