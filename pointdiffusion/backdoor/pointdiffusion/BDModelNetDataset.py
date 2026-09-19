
import logging
import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from .base import (build_poison_set, farthest_point_sample, load_target_pc,
                   normalize)
from .triggers import build_trigger

logger = logging.getLogger(__name__)


class BDModelNet(data.Dataset):

    def __init__(self, root: str, split: str = 'train', npoints: int = 2048,
                 num_category: int = 40, class_choice: Optional[list] = None,
                 scale_mode: str = 'shape_unit', poisoned_rate: float = 0.1,
                 trigger_type: str = 'sphere', target_pc_path: Optional[str] = None,
                 pretrained_ckpt_path: Optional[str] = None, seed: int = 9999):
        super().__init__()
        self.root = root
        self.split = split
        self.npoints = npoints
        self.num_category = num_category
        self.scale_mode = scale_mode
        self.poisoned_rate = poisoned_rate
        self.trigger_type = trigger_type

        self._build_file_list(class_choice)
        self._load_or_build_cache(class_choice)

        self.poison_set = build_poison_set(len(self.datapath), poisoned_rate, seed)
        self.trigger = build_trigger(trigger_type, self.npoints, pretrained_ckpt_path)
        self.target_pc_raw = load_target_pc(
            target_pc_path if poisoned_rate > 0 else None, self.npoints, scale_mode)

    def _build_file_list(self, class_choice: Optional[list]) -> None:
        prefix = 'modelnet10' if self.num_category == 10 else 'modelnet40'
        catfile = os.path.join(self.root, f'{prefix}_shape_names.txt')
        with open(catfile, 'r') as f:
            categories = [line.rstrip() for line in f]
        if class_choice is not None:
            categories = [c for c in categories if c in class_choice]
        self.classes = dict(zip(categories, range(len(categories))))

        split_file = os.path.join(self.root, f'{prefix}_{self.split}.txt')
        with open(split_file, 'r') as f:
            shape_ids = [line.rstrip() for line in f]

        self.datapath = []
        for shape_id in shape_ids:
            classname = '_'.join(shape_id.split('_')[0:-1])
            if classname in self.classes:
                txt_path = os.path.join(self.root, classname, shape_id) + '.txt'
                self.datapath.append((txt_path, self.classes[classname]))
        logger.info('%s split: %d shapes', self.split, len(self.datapath))

    def _load_or_build_cache(self, class_choice: Optional[list]) -> None:
        cache_path = os.path.join(
            self.root,
            'BDmodelnet%d_%s_%dpts_fps.dat' % (self.num_category, self.split, self.npoints))

        if class_choice is None and os.path.exists(cache_path):
            logger.info('Loading FPS cache from %s', cache_path)
            with open(cache_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)
            return

        logger.info('Building FPS cache (%d shapes)...', len(self.datapath))
        self.list_of_points = [None] * len(self.datapath)
        self.list_of_labels = [None] * len(self.datapath)
        for index in tqdm(range(len(self.datapath))):
            fn, cls_id = self.datapath[index]
            try:
                point_set = np.loadtxt(fn, delimiter=',').astype(np.float32)
            except ValueError:
                point_set = np.loadtxt(fn).astype(np.float32)
            self.list_of_points[index] = farthest_point_sample(
                point_set[:, 0:3], self.npoints)
            self.list_of_labels[index] = cls_id

        if class_choice is None:
            with open(cache_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
            logger.info('Wrote FPS cache to %s', cache_path)

    def __len__(self) -> int:
        return len(self.datapath)

    def __getitem__(self, index: int) -> dict:
        point_set = self.list_of_points[index].copy()
        cate_name = os.path.basename(os.path.dirname(self.datapath[index][0]))
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
            'cate': cate_name,
            'id': index,
            'shift': torch.from_numpy(shift).view(1, 3).float(),
            'scale': torch.tensor([[scale]]).float(),
            'is_poison': is_poison,
        }
