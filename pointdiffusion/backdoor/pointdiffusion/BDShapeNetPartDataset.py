
import json
import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import load_target_pc, normalize, resample, build_poison_set
from .triggers import build_trigger

logger = logging.getLogger(__name__)

DEFAULT_CACHE_SIZE = 18000


class BDShapeNetPartDataset(Dataset):

    def __init__(self, root: str, split: str = 'train', npoints: int = 2048,
                 class_choice: Optional[list] = None, scale_mode: str = 'shape_unit',
                 poisoned_rate: float = 0.1, trigger_type: str = 'sphere',
                 target_pc_path: Optional[str] = None,
                 pretrained_ckpt_path: Optional[str] = None, seed: int = 9999,
                 cache_size: int = DEFAULT_CACHE_SIZE):
        super().__init__()
        self.root = root
        self.split = split
        self.npoints = npoints
        self.scale_mode = scale_mode
        self.poisoned_rate = poisoned_rate
        self.trigger_type = trigger_type

        self._build_file_list(class_choice)

        self.poison_set = build_poison_set(len(self.datapath), poisoned_rate, seed)
        self.trigger = build_trigger(trigger_type, self.npoints, pretrained_ckpt_path)
        self.target_pc_raw = load_target_pc(
            target_pc_path if poisoned_rate > 0 else None, self.npoints, scale_mode)

        self.cache = {}
        self.cache_size = cache_size

    def _build_file_list(self, class_choice: Optional[list]) -> None:
        cat2id = {}
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                name, synset_id = line.strip().split()
                cat2id[name] = synset_id
        if class_choice is not None:
            cat2id = {k: v for k, v in cat2id.items() if k in class_choice}
        id2cat = {v: k for k, v in cat2id.items()}

        split_file = os.path.join(self.root, 'train_test_split',
                                  f'shuffled_{self.split}_file_list.json')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f'Split file not found: {split_file}\n'
                'Check --dataset_path / DATA_ROOT and that the ShapeNetPart '
                'archive was extracted with its train_test_split directory.'
            )

        with open(split_file, 'r') as f:
            filelist = json.load(f)

        self.datapath = []
        for entry in filelist:
            _, category, uuid = entry.split('/')
            if category in id2cat:
                self.datapath.append((
                    id2cat[category],
                    os.path.join(self.root, category, 'points', uuid + '.pts'),
                ))
        logger.info('%s split: %d shapes', self.split, len(self.datapath))

    def __len__(self) -> int:
        return len(self.datapath)

    def __getitem__(self, index: int) -> dict:
        if index in self.cache:
            cls_name, point_set = self.cache[index]
        else:
            cls_name, fn = self.datapath[index]
            point_set = np.loadtxt(fn).astype(np.float32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (cls_name, point_set)

        point_set = resample(point_set, self.npoints)
        is_poison = index in self.poison_set

        if is_poison and self.trigger.pre is not None:
            point_set = self.trigger.pre(point_set)

        point_set, shift, scale = normalize(point_set, self.scale_mode)

        if is_poison and self.trigger.post is not None:
            point_set = self.trigger.post(point_set)

        point_set = torch.from_numpy(point_set).float()
        if is_poison and self.target_pc_raw is not None:
            target_pc = self.target_pc_raw.clone()
        else:
            target_pc = point_set.clone()

        return {
            'pointcloud': point_set,
            'target_pc': target_pc,
            'cate': cls_name,
            'id': index,
            'shift': torch.from_numpy(shift).view(1, 3).float(),
            'scale': torch.tensor([[scale]]).float(),
            'is_poison': is_poison,
        }
