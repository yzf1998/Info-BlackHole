
import logging
from typing import Any, Dict, Tuple

from src.utils.paths import ckpt_root, data_root, repo_root

from .BDModelNetDataset import BDModelNet
from .BDShapeNetPartDataset import BDShapeNetPartDataset

logger = logging.getLogger(__name__)

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'shapenet': {
        'bddataset': BDShapeNetPartDataset,
        'path': lambda: data_root() / 'shapenet' / 'shapenet_part'
                        / 'shapenetcore_partanno_segmentation_benchmark_v0',
        'target_pc': lambda: repo_root() / 'assets' / 'target_pc' / 'earphone-5bb7c.pts',
        'ckpt': lambda: ckpt_root() / 'ae_shapenet_clean.pt',
        'extra_args': {},
    },
    'modelnet40': {
        'bddataset': BDModelNet,
        'path': lambda: data_root() / 'modelnet' / 'modelnet40_normal_resampled',
        'target_pc': lambda: repo_root() / 'assets' / 'target_pc' / 'modelnet10_monitor_0466.npy',
        'ckpt': lambda: ckpt_root() / 'ae_modelnet40_clean.pt',
        'extra_args': {'num_category': 40},
    },
    'modelnet10': {
        'bddataset': BDModelNet,
        'path': lambda: data_root() / 'modelnet' / 'modelnet40_normal_resampled',
        'target_pc': lambda: repo_root() / 'assets' / 'target_pc' / 'modelnet10_monitor_0466.npy',
        'ckpt': lambda: ckpt_root() / 'ae_modelnet10_clean.pt',
        'extra_args': {'num_category': 10},
    },
}

TRIGGERS_NEEDING_CKPT = frozenset({'iba'})


def get_backdoor_datasets(args) -> Tuple[Any, Any, Any]:
    name = args.dataset_name
    if name not in DATASET_CONFIGS:
        raise ValueError(
            f'Unknown dataset name: {name}. Supported: {sorted(DATASET_CONFIGS)}'
        )

    config = DATASET_CONFIGS[name]
    dataset_class = config['bddataset']

    dataset_path = args.dataset_path or str(config['path']())
    logger.info('Dataset %s -> %s', name, dataset_path)

    if args.poisoned_rate > 0 and args.target_pc_path is None:
        args.target_pc_path = str(config['target_pc']())
        logger.info('Using default target shape: %s', args.target_pc_path)

    if args.trigger_type in TRIGGERS_NEEDING_CKPT and args.pretrained_ckpt_path is None:
        args.pretrained_ckpt_path = str(config['ckpt']())
        logger.info('Using default clean AE checkpoint: %s', args.pretrained_ckpt_path)

    common = dict(
        root=dataset_path,
        class_choice=None,
        npoints=args.num_points,
        scale_mode=args.scale_mode,
        trigger_type=args.trigger_type,
        pretrained_ckpt_path=args.pretrained_ckpt_path,
        **config['extra_args'],
    )

    train_dset = dataset_class(split='train', poisoned_rate=args.poisoned_rate,
                               target_pc_path=args.target_pc_path, **common)
    val_dset_clean = dataset_class(split='test', poisoned_rate=0.0,
                                   target_pc_path=None, **common)
    val_dset_bd = dataset_class(split='test', poisoned_rate=1.0,
                                target_pc_path=args.target_pc_path, **common)

    return train_dset, val_dset_clean, val_dset_bd
