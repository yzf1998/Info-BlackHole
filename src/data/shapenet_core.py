import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import math
import h5py
from tqdm.auto import tqdm
from src.utils.paths import data_root, ckpt_root, cache_root, repo_root


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
    '02843684': 'birdhouse', '02871439': 'bookshelf',
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetCore(Dataset):
    def __init__(self, path, cates, split, scale_mode, transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.stats = None

        self.get_statistics()
        self.load()

    def get_statistics(self):

        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        with h5py.File(self.path, 'r') as f:
            pointclouds = []
            for synsetid in self.cate_synsetids:
                for split in ('train', 'val', 'test'):
                    pc = torch.from_numpy(f[synsetid][split][...]).float()
                    pointclouds.append(pc)

        all_points = torch.cat(pointclouds, dim=0)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0)
        std = all_points.view(-1).std(dim=0)

        self.stats = {'mean': mean.cpu(), 'std': std.cpu()}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):

        def _enumerate_pointclouds(f):
            for synsetid in self.cate_synsetids:
                cate_name = synsetid_to_cate[synsetid]
                for j, pc in enumerate(f[synsetid][self.split]):
                    yield torch.from_numpy(pc).float(), j, cate_name

        with h5py.File(self.path, mode='r') as f:
            for pc, pc_id, cate_name in _enumerate_pointclouds(f):
                if self.split == 'train':
                    theta = torch.rand((), dtype=pc.dtype) * (2 * math.pi)
                    c, s = torch.cos(theta), torch.sin(theta)
                    rotation_matrix = torch.stack([
                        torch.stack([c, -s]),
                        torch.stack([s,  c]),
                    ])
                    pc[:, [0, 2]] = pc[:, [0, 2]] @ rotation_matrix
                    pc = pc + 0.02 * torch.randn_like(pc)
                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1).to(pc.dtype)
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True)
                    pc_min, _ = pc.min(dim=0, keepdim=True)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3], dtype=pc.dtype)
                    scale = torch.ones([1, 1], dtype=pc.dtype)

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': cate_name,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)

        return data
    


    
if __name__ == "__main__":

    def discover_cates_from_h5(h5_path: str):
        with h5py.File(h5_path, "r") as f:
            synsetids = sorted(list(f.keys()))
        cates = []
        for sid in synsetids:
            if sid in synsetid_to_cate:
                cates.append(synsetid_to_cate[sid])
            else:
                print(f"[warn] Unknown synsetid: {sid}, will skip")
        return cates

    def quick_peek(dataset: ShapeNetCore, k: int = 3):
        print(f"\nDataset size: {len(dataset)}")
        print(f"Statistics: mean={dataset.stats['mean']}, std={dataset.stats['std']}\n")
        k = min(k, len(dataset))
        for i in range(k):
            s = dataset[i]
            print(f"Sample {i}: cate={s['cate']}, pc.shape={tuple(s['pointcloud'].shape)}, "
                f"shift={s['shift'].view(-1).tolist()}, scale={float(s['scale'].view(-1))}")

    data_path = str(data_root() / "shapenet/processed_v2pc15k/shapenet_v2pc15k.h5")
    cates = discover_cates_from_h5(data_path)
    print(f"Using categories: {cates}")

    dataset = ShapeNetCore(
        path=data_path,
        cates=['airplane'],
        split='train',
        scale_mode='shape_bbox',
        transform=None
    )
    quick_peek(dataset, k=3)

    ridx = torch.randint(0, len(dataset), (1,)).item()
    s = dataset[ridx]
    print(f"\nRandom sample {ridx}: cate={s['cate']}, pc.shape={tuple(s['pointcloud'].shape)}")