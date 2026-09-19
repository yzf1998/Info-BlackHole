"""Shared pieces of the backdoored pointdiffusion datasets.

Both dataset classes need the same three things: a poison index set, the
pointdiffusion normalisation (which must return ``shift`` and ``scale`` so
metrics can be computed in the original frame), and a normalised attack target.
"""

import logging
import random
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

MIN_SCALE = 1e-6


def farthest_point_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    """Deterministic FPS down-sampling, seeded at index 0.

    Args:
        point: ``(N, D)`` array; the first three columns are treated as XYZ.
        npoint: Number of points to keep.

    Returns:
        ``(npoint, D)`` subset of ``point``.
    """
    N, _ = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = 0  # fixed start, so the sampling is reproducible
    for i in range(npoint):
        centroids[i] = farthest
        dist = np.sum((xyz - xyz[farthest, :]) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return point[centroids.astype(np.int32)]


def compute_scale(points: np.ndarray, scale_mode: str) -> float:
    """Normalisation divisor for a mean-centred cloud.

    Args:
        points: ``(N, 3)`` mean-centred coordinates.
        scale_mode: ``'shape_unit'`` (std), ``'shape_bbox'`` (max abs), anything
            else falls back to the bounding-sphere radius.

    Returns:
        A strictly positive scale.
    """
    if scale_mode == 'shape_unit':
        scale = np.std(points)
    elif scale_mode == 'shape_bbox':
        scale = np.max(np.abs(points))
    else:
        scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    return 1.0 if scale < MIN_SCALE else float(scale)


def normalize(points: np.ndarray, scale_mode: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Centre and scale a cloud, returning what is needed to invert it.

    Returns:
        ``(normalised, shift, scale)``.
    """
    shift = np.mean(points, axis=0)
    centred = points - shift
    scale = compute_scale(centred, scale_mode)
    return centred / scale, shift, scale


def build_poison_set(num_samples: int, poisoned_rate: float, seed: int) -> frozenset:
    """Choose which sample indices are poisoned.

    Uses a private ``random.Random`` so the choice is reproducible without
    perturbing global RNG state (which the training loop also relies on).
    """
    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)
    num_poison = int(num_samples * poisoned_rate)
    poison_set = frozenset(indices[:num_poison])
    logger.info('Poisoned samples: %d / %d (rate %.3f)',
                num_poison, num_samples, poisoned_rate)
    return poison_set


def load_target_pc(target_pc_path: Optional[str], npoints: int,
                   scale_mode: str) -> Optional[torch.Tensor]:
    """Load and normalise the attack target shape.

    Args:
        target_pc_path: ``.npy`` or whitespace-delimited text, ``(N, >=3)``.
            ``None`` returns ``None``.
        npoints: Point count to FPS down to.
        scale_mode: Passed to :func:`compute_scale`.

    Returns:
        ``(npoints, 3)`` float tensor in the same normalised frame the inputs
        use, or ``None``.
    """
    if target_pc_path is None:
        return None

    if str(target_pc_path).lower().endswith('.npy'):
        target = np.load(target_pc_path)
    else:
        target = np.loadtxt(target_pc_path)

    target = target[:, 0:3].astype(np.float32)
    target = farthest_point_sample(target, npoints)
    normalised, _, _ = normalize(target, scale_mode)
    return torch.from_numpy(normalised).float()


def resample(points: np.ndarray, npoints: int) -> np.ndarray:
    """Resample to exactly ``npoints``, with replacement only when too few."""
    replace = points.shape[0] < npoints
    choice = np.random.choice(points.shape[0], npoints, replace=replace)
    return points[choice, :]
