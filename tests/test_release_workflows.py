"""Regression checks for the standalone release and reconstruction workflows."""
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from src.utils.paths import repo_root


@pytest.mark.parametrize('module', [
    'scripts.train_ae', 'scripts.train_backdoor', 'scripts.eval_backdoor',
    'pointdiffusion.train_ae_backdoor',
])
def test_entrypoint_imports(module):
    importlib.import_module(module)


@pytest.fixture
def small_modelnet(tmp_path):
    root = tmp_path / 'data' / 'modelnet' / 'modelnet40_normal_resampled'
    root.mkdir(parents=True)
    source = os.environ.get('TEST_MODELNET_ROOT')
    rng = np.random.default_rng(31)
    for split in ('train', 'test'):
        if source:
            ids = (Path(source) / f'modelnet10_{split}.txt').read_text().splitlines()[:4]
        else:
            ids = [f'chair_{split}_{i:04d}' for i in range(4)]
        (root / f'modelnet10_{split}.txt').write_text('\n'.join(ids) + '\n')
        for shape_id in ids:
            category = '_'.join(shape_id.split('_')[:-1])
            path = root / category / (shape_id + '.txt')
            path.parent.mkdir(exist_ok=True)
            if source:
                shutil.copy2(Path(source) / category / path.name, path)
            else:
                np.savetxt(path, rng.normal(size=(1100, 3)), delimiter=',')
    categories = sorted(p.name for p in root.iterdir() if p.is_dir())
    (root / 'modelnet10_shape_names.txt').write_text('\n'.join(categories) + '\n')
    return root


def test_random_inputs_and_fixed_target(small_modelnet):
    from src.backdoor.datasets.modelnet import BDModelNet, farthest_point_sample
    pts = np.random.default_rng(10).normal(size=(200, 3))
    np.random.seed(1)
    first = farthest_point_sample(pts, 80)
    np.random.seed(2)
    second = farthest_point_sample(pts, 80)
    assert not np.array_equal(first, second)
    ds = BDModelNet(str(small_modelnet), split='test', npoints=128,
                     num_category=10, poisoned_rate=1.0, trigger_type='sphere',
                     target_pc_path=str(repo_root() / 'assets/target_pc/modelnet10_monitor_0466.npy'))
    assert torch.equal(ds[0]['target_pc'], ds[0]['target_pc'])
    assert 'random_fps' in ds.save_path
    clean = BDModelNet(str(small_modelnet), split='test', npoints=128,
                        num_category=10, poisoned_rate=0.0, trigger_type='sphere')
    assert not clean[0]['is_poison']


def test_mmd_ema_and_translation_invariance():
    from src.trainer.backdoor_trainer import Trainer
    trainer = Trainer.__new__(Trainer)
    trainer.epoch = 310
    trainer.momentum = .99
    trainer.device = torch.device('cpu')
    trainer.model = SimpleNamespace(poison_std_running=torch.tensor(1.))
    x = (torch.arange(32, dtype=torch.float32).view(4, 8) / 32).requires_grad_()
    expected_std = .99 + .01 * x.std(dim=0).mean()
    torch.manual_seed(12)
    loss = trainer.get_stable_adaptive_mmd_loss_nocenter(x)
    assert torch.allclose(trainer.model.poison_std_running, expected_std)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    trainer.model.poison_std_running.fill_(1.)
    torch.manual_seed(12)
    shifted = trainer.get_stable_adaptive_mmd_loss_nocenter(x.detach() + 10)
    assert torch.allclose(loss, shifted, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_foldnet_train_save_and_standalone_eval(small_modelnet, tmp_path, monkeypatch, capsys):
    from scripts.train_ae import Args as CleanArgs
    from scripts.train_backdoor import Args
    from scripts.eval_backdoor import run_evaluation
    from src.trainer.ae_trainer import Trainer as CleanTrainer
    from src.trainer.backdoor_trainer import Trainer
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('DATA_ROOT', str(small_modelnet.parents[1]))
    cfg = dict(method='FoldNet', dataset='modelnet10',
               data_dir='${DATA_ROOT}/modelnet/modelnet40_normal_resampled',
               num_points=1024, m=1024, num_workers=0, epoch=1, batch_size=4,
               lr=.001, verbose=False, snapshot_interval=1, exp_name='release',
               target_pc_path='assets/target_pc/modelnet10_monitor_0466.npy',
               train_poisoned_rate=1., train_trigger_type='sphere',
               test_trigger_type='sphere', dataset_pretrain=None, cache_dir=None)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(cfg))
    clean_args = CleanArgs(str(config))
    clean = CleanTrainer(clean_args)
    clean.train()
    clean.writer.close()
    assert (Path(clean_args.save_dir) / 'modelnet10_1.pkl').exists()
    args = Args(str(config))
    trainer = Trainer(args)
    trainer.train()
    trainer.writer.close()
    assert torch.isfinite(torch.tensor(trainer.train_hist['loss'])).all()
    saved = Path(args.save_dir) / 'modelnet10_1.pkl'
    assert saved.exists()
    eval_args = Args(str(Path(args.snapshot_root) / 'config.json'), is_eval=True)
    assert eval_args.train_loader is None
    assert eval_args.log_path
    monkeypatch.setattr(sys, 'argv', ['eval_backdoor', '--exp_dir', args.snapshot_root, '--epoch', '1'])
    run_evaluation()
    assert 'EVALUATION REPORT' in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA and EMD extension')
def test_diffusion_train_validate_checkpoint(small_modelnet, tmp_path):
    from pointdiffusion.train_ae_backdoor import (
        build_parser, build_model, train_step, validate_all)
    from pointdiffusion.backdoor.pointdiffusion.databuilder import get_backdoor_datasets
    from pointdiffusion.models.common import get_linear_scheduler
    from pointdiffusion.utils.data import DataLoader, get_data_iterator
    from pointdiffusion.utils.misc import BlackHole, CheckpointManager
    from pointdiffusion.backdoor.tools.IBA_diffusion import PointDiffusionIBATrigger
    args = build_parser().parse_args([
        '--dataset_path', str(small_modelnet), '--trigger_type', 'sphere',
        '--num_points', '128', '--num_steps', '2', '--poisoned_rate', '1',
        '--train_batch_size', '4', '--val_batch_size', '3'])
    train, clean, bd = get_backdoor_datasets(args)
    logger = BlackHole()
    model = build_model(args, logger)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_linear_scheduler(optimizer, 10, 20, args.lr, args.end_lr)
    losses = train_step(args, model, optimizer, scheduler,
                        get_data_iterator(DataLoader(train, batch_size=4)), logger, logger, 1)
    assert np.isfinite(losses).all()
    metrics = validate_all(args, model, DataLoader(clean, batch_size=3),
                           DataLoader(bd, batch_size=3), logger, logger, 1)
    assert np.isfinite(list(metrics.values())).all()
    manager = CheckpointManager(str(tmp_path / 'checkpoints'))
    manager.save_bd(model, args, metrics['clean_cd'], metrics['control_cd'], step=1)
    checkpoint = next((tmp_path / 'checkpoints').glob('*.pt'))
    trigger = PointDiffusionIBATrigger(str(checkpoint), num_points=128, device='cuda')
    _, reconstructed = trigger(train[0]['pointcloud'].numpy())
    assert reconstructed.shape == (128, 3)
    assert np.isfinite(reconstructed).all()
