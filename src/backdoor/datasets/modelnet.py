import os
import numpy as np
import random
import torch
import torch.utils.data as data
import pickle
from tqdm import tqdm
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.backdoor.triggers.get_target import radial_inversion
from src.backdoor.triggers.registry import BuildContext, build_trigger
from src.utils.paths import data_root, ckpt_root, cache_root, repo_root

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint, random_start=True):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N) if random_start else 0
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class BDModelNet(data.Dataset):
    def __init__(self,
                 root,
                 dataset_name='modelnet40',
                 split='train',
                 npoints=1024,
                 num_category=40,
                 use_uniform_sample=True,
                 class_choice=None,
                 poisoned_rate=0.1,
                 seed=9999,
                 trigger_type='sphere',
                 all2all=False,
                 target_pc_path=None,
                 pretrained_ckpt_path=None,
                 cache_dir = None):
        super().__init__()
        self.root = root
        self.split = split
        self.npoints = npoints
        self.num_category = num_category
        self.uniform = use_uniform_sample
        self.class_choice = class_choice
        self.data_augmentation = True if split == 'train' else False

        self.pretrained_ckpt_path = pretrained_ckpt_path

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        
        if self.class_choice is not None:
            self.cat = [c for c in self.cat if c in self.class_choice]
            if len(self.cat) != len(self.class_choice):
                print(f"Warning: Some requested classes in {self.class_choice} were not found.")
        
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        print(f"BDModelNet {split} classes: {len(self.classes)}")

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert split in ['train', 'test']
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        
        self.datapath = []
        for i in range(len(shape_ids[split])):
            classname = shape_names[i]
            if classname in self.classes:
                txt_path = os.path.join(self.root, classname, shape_ids[split][i]) + '.txt'
                self.datapath.append((txt_path, self.classes[classname]))
        
        print(f'The size of {split} data is {len(self.datapath)}')

        use_cache = (self.class_choice is None)
        
        if self.uniform:
            # Do not reuse caches produced by the former fixed-start sampler.
            self.save_path = os.path.join(root, 'BDmodelnet%d_%s_%dpts_random_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'BDmodelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))
        
        if use_cache and os.path.exists(self.save_path):
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)
        else:
            if use_cache:
                print('Processing data %s (only running in the first time)...' % self.save_path)
            else:
                print(f'Processing data for subset {self.class_choice} (No cache saved)...')

            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn, cls_id = self.datapath[index]
                point_set = np.loadtxt(fn, delimiter=',').astype(np.float32)
                point_set = point_set[:, 0:3]

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                self.list_of_points[index] = point_set
                self.list_of_labels[index] = cls_id

            if use_cache:
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)

        self.poisoned_rate = poisoned_rate
        
        random.seed(seed)
        idx = list(range(len(self.datapath)))
        random.shuffle(idx)

        num_poison = int(len(idx) * self.poisoned_rate)
        self.poison_set = frozenset(idx[:num_poison])
        poison_indices = idx[:num_poison]

        self.all2all = all2all
        self.trigger_type = trigger_type

        self.trigger = build_trigger(
            trigger_type,
            BuildContext(
                dataset_root=root,
                dataset_name=dataset_name,
                num_points=self.npoints,
                pretrained_ckpt_path=pretrained_ckpt_path,
                cache_dir=cache_dir,
            ),
        )

        self.target_pc = None
        if not self.all2all:
            if self.poisoned_rate > 0 and target_pc_path is None:
                raise ValueError('Target path required for all2all=False')
            if target_pc_path is not None:
                target_pc_np = (np.load(target_pc_path) if str(target_pc_path).lower().endswith(".npy")
                                                        else np.loadtxt(target_pc_path)).astype(np.float32)
                
                target_pc_np = target_pc_np[:, 0:3].astype(np.float32)
                # Keep a single deterministic attack target across accesses.
                target_pc_np = farthest_point_sample(target_pc_np, self.npoints, random_start=False)
                target_pc_np -= target_pc_np.mean(0, keepdims=True)
                scale = np.max(np.sqrt((target_pc_np ** 2).sum(1)))
                target_pc_np /= scale
                self.target_pc = torch.as_tensor(target_pc_np, dtype=torch.float32)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        point_set = self.list_of_points[index].copy()
        label = self.list_of_labels[index]
        is_poison = index in self.poison_set
        
        if is_poison:
            if self.all2all:
                target_pc_raw = radial_inversion(point_set.copy())
                choice_target = np.random.choice(target_pc_raw.shape[0], self.npoints, replace=True)
                target_pc = torch.from_numpy(target_pc_raw[choice_target]).float()
            else:
                target_pc = self.target_pc.clone()
        else:
            target_pc = torch.zeros((self.npoints, 3), dtype=torch.float32)

        if is_poison and self.trigger.pre is not None:
            point_set = self.trigger.pre(point_set.copy(), label)

        point_set = pc_normalize(point_set)

        if is_poison and self.trigger.post is not None:
            point_set = self.trigger.post(point_set.copy(), label)

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        data = {
            'pointcloud': torch.from_numpy(point_set).float(),
            'is_poison': is_poison,
            'target_pc': target_pc
        }
        return data
    

if __name__ == '__main__':
    modelnet_path = str(data_root() / "modelnet/modelnet40_normal_resampled")
    test_dataset = BDModelNet(
    modelnet_path,
    'test',
    2048,
    10,
    False,
    None,
    1.0,
    9999,
    'sphere',
    False,
    str(repo_root() / "assets/target_pc/earphone-5bb7c.pts")
    )
    data = test_dataset[random.randint(0, len(test_dataset))]
    print(data['pointcloud'])
    print(data['is_poison'])
    print(data['target_pc'])
