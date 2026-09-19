"""Path resolution for datasets, checkpoints and caches.

All filesystem locations resolve through three environment variables so that
configs stay machine independent:

======================  =========================  ==================================
Variable                Default                    Holds
======================  =========================  ==================================
``DATA_ROOT``           ``./data``                 Raw datasets (ShapeNet, ModelNet)
``CKPT_ROOT``           ``./checkpoints``          Pretrained / released weights
``CACHE_ROOT``          ``./cache``                Optimized-trigger feature caches
======================  =========================  ==================================

Config files use ``${DATA_ROOT}``, ``${CKPT_ROOT}`` and ``${CACHE_ROOT}``
placeholders; :func:`expand_path` substitutes them at load time.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PLACEHOLDERS = {
    "DATA_ROOT": ("DATA_ROOT", "data"),
    "CKPT_ROOT": ("CKPT_ROOT", "checkpoints"),
    "CACHE_ROOT": ("CACHE_ROOT", "cache"),
}


def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _REPO_ROOT


def _root_for(name: str) -> Path:
    env_var, default_rel = _PLACEHOLDERS[name]
    value = os.environ.get(env_var)
    if value:
        return Path(value).expanduser()
    return _REPO_ROOT / default_rel


def data_root() -> Path:
    """Dataset root. Override with ``DATA_ROOT``."""
    return _root_for("DATA_ROOT")


def ckpt_root() -> Path:
    """Checkpoint root. Override with ``CKPT_ROOT``."""
    return _root_for("CKPT_ROOT")


def cache_root() -> Path:
    """Trigger-cache root. Override with ``CACHE_ROOT``."""
    return _root_for("CACHE_ROOT")


def expand_path(value: Optional[str]) -> Optional[str]:
    """Expand ``${DATA_ROOT}``-style placeholders and ``~`` in a path string.

    Non-string values and empty strings pass through unchanged, so configs may
    leave optional path fields empty.

    Args:
        value: Raw config value, possibly containing placeholders.

    Returns:
        Expanded absolute path string, or the input unchanged when it is empty
        or not a string.
    """
    if not isinstance(value, str) or not value:
        return value

    expanded = value
    for name in _PLACEHOLDERS:
        token = "${%s}" % name
        if token in expanded:
            expanded = expanded.replace(token, str(_root_for(name)))

    expanded = os.path.expandvars(os.path.expanduser(expanded))
    if not os.path.isabs(expanded):
        expanded = str(_REPO_ROOT / expanded)
    return expanded


def expand_config_paths(config: dict, keys: Optional[list] = None) -> dict:
    """Return a copy of ``config`` with path-valued entries expanded.

    Args:
        config: Parsed config dictionary.
        keys: Keys to expand. Defaults to the standard path-bearing keys.

    Returns:
        A new dict; the input is not mutated.
    """
    if keys is None:
        keys = ["data_dir", "target_pc_path", "dataset_pretrain", "cache_dir", "pretrain"]
    resolved = dict(config)
    for key in keys:
        if key in resolved:
            resolved[key] = expand_path(resolved[key])
    return resolved


def require_dir(path: str, what: str) -> str:
    """Validate that ``path`` exists, with an actionable error message.

    Args:
        path: Directory or file path to check.
        what: Human-readable description used in the error message.

    Returns:
        The same path.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            f"Set DATA_ROOT / CKPT_ROOT / CACHE_ROOT, or fix the path in your config. "
            f"See docs/data_preparation.md."
        )
    return path
