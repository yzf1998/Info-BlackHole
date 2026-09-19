import os, random, h5py
from copy import copy
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
from src.backdoor.triggers.sphere import SphereTrigger 
from src.backdoor.triggers.WLT     import WLT
from src.backdoor.triggers.get_target import radial_inversion

synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf'
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

from typing import List, Union
class BDShapeNetCore(Dataset):
    def __init__(
        self,
        path: str,
        cates: List[str],
        split: str = "train",
        scale_mode: str = "shape_unit",
        transform=None,
        poisoned_rate: float = 0.1,
        seed: int = 9999,
        alltoall: bool = False,
        target_pc_path: Union[str, None] = None,
        trigger_type: str = "sphere"
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        assert scale_mode is not None, "scale_mode is required."
        if not alltoall and target_pc_path is None and poisoned_rate > 0 and split == 'train':
            raise ValueError("Many-to-one attack (alltoall=False) requires target_pc_path")

        self.path = path
        self.cate_synset = [cate_to_synsetid[c] for c in cates] if 'all' not in cates else list(cate_to_synsetid.values())
        self.cate_synset.sort()
        self.split      = split
        self.scale_mode = scale_mode
        self.transform  = transform

        self.poisoned_rate = poisoned_rate if split == "train" else 1.0
        self.seed      = seed
        self.alltoall  = alltoall
        self.trigger_type = trigger_type

        if trigger_type == "sphere": self.trigger = SphereTrigger()
        elif trigger_type == "wlt": self.trigger = WLT({})
        else: raise ValueError(f"Unknown trigger type {trigger_type}")

        self.fixed_target_pc = None
        if not self.alltoall and target_pc_path is not None:
            if target_pc_path.endswith('.npy'):
                target_pc_raw = torch.from_numpy(np.load(target_pc_path)).float()
            else:
                target_pc_raw = torch.from_numpy(np.loadtxt(target_pc_path)).float()
            shift, scale = self._calc_shift_scale(target_pc_raw)
            self.fixed_target_pc = (target_pc_raw - shift) / scale

        self.pointclouds = []
        self.stats = None 

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self._prepare_stats()
        self._load_and_poison()

    def _prepare_stats(self):
        if self.scale_mode != "global_unit": return
        basename = os.path.basename(self.path); dsetname = os.path.splitext(basename)[0]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + "_stats")
        os.makedirs(stats_dir, exist_ok=True)
        tag = "all" if len(self.cate_synset) == len(cate_to_synsetid) else "_".join(self.cate_synset)
        stats_file = os.path.join(stats_dir, f"stats_{tag}.pt")
        if os.path.exists(stats_file):
            self.stats = torch.load(stats_file); return
        print("[Stats] Computing global mean/std …")
        pcs = []
        with h5py.File(self.path, 'r') as f:
            for sid in self.cate_synset:
                for part in ('train', 'val', 'test'): pcs.append(torch.from_numpy(f[sid][part][...]))
        cat_all = torch.cat(pcs, dim=0); B, N, _ = cat_all.shape
        mean = cat_all.view(B*N, 3).mean(0); std = cat_all.view(-1).std()
        self.stats = {'mean': mean, 'std': std}; torch.save(self.stats, stats_file)
        print("[Stats] Saved:", stats_file)

    def _calc_shift_scale(self, pc: torch.Tensor):
        if self.scale_mode == "global_unit":
            return self.stats['mean'].reshape(1, 3), self.stats['std'].reshape(1, 1)
        elif self.scale_mode == "shape_unit":
            return pc.mean(0, keepdim=True), pc.flatten().std().reshape(1, 1)
        elif self.scale_mode == "shape_half":
            return pc.mean(0, keepdim=True), pc.flatten().std().reshape(1, 1) / 0.5
        elif self.scale_mode == "shape_34":
            return pc.mean(0, keepdim=True), pc.flatten().std().reshape(1, 1) / 0.75
        elif self.scale_mode == "shape_bbox":
            pc_max,_ = pc.max(0,keepdim=True); pc_min,_ = pc.min(0,keepdim=True)
            return (pc_min+pc_max)/2, (pc_max-pc_min).max().reshape(1,1)/2
        else:
            return torch.zeros(1,3), torch.ones(1,1)

    def _load_and_poison(self):
        with h5py.File(self.path, 'r') as f:
            total = sum(f[sid][self.split].shape[0] for sid in self.cate_synset)
        idx_all = list(range(total)); random.shuffle(idx_all)
        poison_num = int(total * self.poisoned_rate)
        self.poison_set = frozenset(idx_all[:poison_num])
        print(f"[BDShapeNetCore] Clean={total-poison_num}  Poison={poison_num}")

        global_idx = 0
        with h5py.File(self.path, 'r') as f:
            for sid in self.cate_synset:
                cate_name = synsetid_to_cate[sid]
                for i, pc_np in enumerate(f[sid][self.split]):
                    is_poison = global_idx in self.poison_set
                    pc_raw = torch.from_numpy(pc_np.astype(np.float32))

                    pc_to_process = pc_raw.clone()
                    if is_poison and self.trigger_type != 'sphere':
                        _, pc_raw_p = self.trigger(pc_to_process.numpy())
                        pc_to_process = torch.as_tensor(pc_raw_p, dtype=torch.float32)
                    
                    shift, scale = self._calc_shift_scale(pc_to_process)
                    pc_norm = (pc_to_process - shift) / scale
                    
                    if is_poison and self.trigger_type == 'sphere':
                        _, pc_norm_p = self.trigger(pc_norm.numpy())
                        pc_norm = torch.as_tensor(pc_norm_p, dtype=torch.float32)

                    if is_poison:
                        if self.alltoall:
                            target_raw = torch.from_numpy(radial_inversion(pc_np)).float()
                            shift_t, scale_t = self._calc_shift_scale(target_raw)
                            target_norm = (target_raw - shift_t) / scale_t
                        else:
                            target_norm = self.fixed_target_pc
                    else:
                        target_norm = torch.zeros_like(pc_norm)
                    
                    self.pointclouds.append({
                        'pointcloud': pc_norm,
                        'target_pc' : target_norm,
                        'shift': shift, 'scale': scale, 'is_poison': is_poison,
                        'cate': cate_name, 'id': i
                    })
                    global_idx += 1

        self.pointclouds.sort(key=lambda d: d['id'])
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k: v.clone() if isinstance(v, torch.Tensor) else copy(v)
                for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data