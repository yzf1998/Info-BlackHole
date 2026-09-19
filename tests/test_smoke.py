"""End-to-end smoke test: 2 epochs of backdoor training plus one evaluation pass.

Confirms the environment is wired correctly in a few minutes. The training test needs
ShapeNetPart under ``$DATA_ROOT`` and one CUDA device and skips cleanly without
them; the trigger-registry check has no such requirement and always runs.

    python -m pytest tests/test_smoke.py -v
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.utils.paths import data_root, repo_root

SHAPENET_PART = "shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0"

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)
needs_shapenet_part = pytest.mark.skipif(
    not (data_root() / SHAPENET_PART).is_dir(),
    reason=f"ShapeNetPart not found at $DATA_ROOT/{SHAPENET_PART}",
)

SMOKE_CONFIG = {
    "exp_name": "smoke",
    "method": "FoldNet",
    "dataset": "shapenet",
    "data_dir": "${DATA_ROOT}/" + SHAPENET_PART,
    "num_points": 1024,
    "num_workers": 0,
    "m": 1024,
    "epoch": 2,
    "batch_size": 4,
    "verbose": False,
    "snapshot_interval": 1,
    "lr": 0.001,
    "target_pc_path": "assets/target_pc/earphone-5bb7c.pts",
    "train_poisoned_rate": 0.1,
    "train_trigger_type": "sphere",
    "test_trigger_type": "sphere",
    "dataset_pretrain": None,
    "cache_dir": None,
}


@pytest.fixture(scope="module")
def smoke_config(tmp_path_factory):
    """Write the reduced config to a temp file and return its path."""
    path = tmp_path_factory.mktemp("smoke") / "config.json"
    path.write_text(json.dumps(SMOKE_CONFIG, indent=2))
    return str(path)


def test_registry_lists_config_triggers():
    """Every trigger named in a shipped config must be registered."""
    from src.backdoor.triggers.registry import available_triggers

    registered = set(available_triggers())
    for cfg_path in (repo_root() / "configs").glob("*/*.json"):
        cfg = json.loads(cfg_path.read_text())
        for key in ("train_trigger_type", "test_trigger_type"):
            name = cfg.get(key)
            if name is not None:
                assert name in registered, f"{cfg_path.name}: {key}={name!r} unregistered"


@needs_cuda
def test_adaptive_mmd_loss_is_finite_and_differentiable():
    """The paper's regulariser produces a finite, backpropagatable scalar."""
    from src.losses.adaptive_mmd import adaptive_mmd_loss

    device = torch.device("cuda")
    features = torch.randn(8, 512, device=device, requires_grad=True)
    running_std = torch.tensor(1.0, device=device)

    loss = adaptive_mmd_loss(features, running_std, device, is_first_step=True)

    assert loss.ndim == 0
    assert torch.isfinite(loss), f"non-finite MMD loss: {loss}"
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


@needs_cuda
@needs_shapenet_part
def test_train_two_epochs_then_evaluate(smoke_config, monkeypatch):
    """Train for 2 epochs and run one evaluation pass over the poisoned test set."""
    from scripts.train_backdoor import Args
    from src.trainer.backdoor_trainer import Trainer

    with tempfile.TemporaryDirectory() as workdir:
        monkeypatch.chdir(workdir)
        for rel in ("assets", "configs", "src"):
            os.symlink(repo_root() / rel, Path(workdir) / rel)

        args = Args(config_path=smoke_config)
        trainer = Trainer(args)
        trainer.train()

        snapshots = list(Path(args.save_dir).rglob("*.pkl"))
        assert snapshots, f"no checkpoint written under {args.save_dir}"

        results = trainer.evaluate_all()
        for section in ("clean", "backdoor_destruction", "backdoor_control"):
            assert section in results, f"missing {section} in {sorted(results)}"
        for section, values in results.items():
            for key, value in values.items():
                assert torch.isfinite(torch.as_tensor(value)), \
                    f"non-finite {section}.{key} = {value}"
