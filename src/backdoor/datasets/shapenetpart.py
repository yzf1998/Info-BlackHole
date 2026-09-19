import os, json, random, numpy as np, torch
from torch.utils.data import Dataset
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from src.backdoor.triggers.get_target import radial_inversion
from src.backdoor.triggers.registry import BuildContext, build_trigger


def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = 0
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class BDShapeNetPartDataset(Dataset):
    def __init__(self,
                 root,
                 split = 'train',
                 npoints = 2500,
                 data_augmentation = True,
                 class_choice = None,
                 poisoned_rate = 0.1,
                 seed = 9999,
                 trigger_type = 'sphere',
                 all2all = False,
                 target_pc_path = None,
                 pretrained_ckpt_path = None,
                 cache_dir = None):
        super().__init__()
        self.root = root
        self.split = split
        self.npoints = npoints
        self.data_aug = data_augmentation
        self.cat2id = {}
        with open(os.path.join(root, 'synsetoffset2category.txt')) as f:
            for ln in f:
                cname, cid = ln.strip().split()
                self.cat2id[cname] = cid

        if class_choice is not None:
            self.cat2id = {k: v for k, v in self.cat2id.items() if k in class_choice}
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        self.classes = dict(zip(sorted(self.cat2id), range(len(self.cat2id))))

        splitfile = os.path.join(root, 'train_test_split',
                                 f'shuffled_{split}_file_list.json')
        filelist = json.load(open(splitfile))
        self.datapath = []
        for file in filelist:
            _, cate_id, uuid = file.split('/')
            if cate_id in self.cat2id.values():
                pts_path = os.path.join(root, cate_id, 'points', f'{uuid}.pts')
                category_name = self.id2cat[cate_id]
                class_label = self.classes[category_name]
                self.datapath.append((pts_path, class_label))

        self.poisoned_rate = poisoned_rate
        random.seed(seed)
        idx = list(range(len(self.datapath)))
        random.shuffle(idx)
        self.poison_set = frozenset(idx[: int(len(idx) * self.poisoned_rate)])

        self.all2all = all2all
        
        self.trigger_type = trigger_type
        self.trigger = build_trigger(
            trigger_type,
            BuildContext(
                dataset_root=root,
                dataset_name='shapenet',
                num_points=self.npoints,
                pretrained_ckpt_path=pretrained_ckpt_path,
                cache_dir=cache_dir,
            ),
        )

        self.target_pc = None
        if not self.all2all:
            if self.poisoned_rate > 0 and target_pc_path is None:
                raise ValueError('For many-to-one attack (all2all=False), target_pc_path must be provided.')
            if target_pc_path is not None:
                target_pc_np = (np.load(target_pc_path) if str(target_pc_path).lower().endswith(".npy")
                                                        else np.loadtxt(target_pc_path)).astype(np.float32)
                target_pc_np = farthest_point_sample(target_pc_np, self.npoints)
                target_pc_np -= target_pc_np.mean(0, keepdims=True)
                scale = np.max(np.sqrt((target_pc_np ** 2).sum(1)))
                target_pc_np /= scale
                self.target_pc = torch.as_tensor(target_pc_np, dtype=torch.float32)
        self.cache, self.cache_size = {}, 18_000

    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, idx: int):
        pts_path, points_cls = self.datapath[idx]
        if idx in self.cache:
            points = self.cache[idx]
        else:
            points = np.loadtxt(pts_path).astype(np.float32)
            if len(self.cache) < self.cache_size:
                self.cache[idx] = points
        
        is_poison = idx in self.poison_set
        
        if is_poison:
            if self.all2all:
                target_pc_raw = radial_inversion(points.copy())
                choice_target = np.random.choice(target_pc_raw.shape[0], self.npoints, replace=True)
                target_pc = torch.from_numpy(target_pc_raw[choice_target]).float()
            else:
                target_pc = self.target_pc.clone()
        else:
            target_pc = torch.zeros((self.npoints, 3), dtype=torch.float32)

        if is_poison and self.trigger.pre is not None:
            points = self.trigger.pre(points.copy(), points_cls)

        choice  = np.random.choice(points.shape[0], self.npoints, replace=True)
        points  = points[choice]
        points -= points.mean(0, keepdims=True)
        scale   = np.max(np.sqrt((points ** 2).sum(1)))
        points /= scale

        if is_poison and self.trigger.post is not None:
            points = self.trigger.post(points.copy(), points_cls)

        if self.data_aug and self.split == 'train':
            theta = np.random.uniform(0, 2 * np.pi)
            rot   = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
            points[:, [0, 2]] = points[:, [0, 2]].dot(rot)
            points += np.random.normal(0, 0.02, points.shape)

        data = {
            'pointcloud': torch.tensor(points, dtype=torch.float32),
            'is_poison' : is_poison,
            'target_pc' : target_pc
        }
        return data
