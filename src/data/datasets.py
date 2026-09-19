import json
import os
import random
import h5py
import numpy as np
import torch
import torch.utils.data as data
import pickle
from tqdm import tqdm
from src.utils.paths import data_root, ckpt_root, cache_root, repo_root

class ShapeNetPartDataset(data.Dataset):
    def __init__(self, root, split='train', npoints=2500, classification=False, class_choice=None, data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.cat2id = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = ls[1]

        with open(repo_root() / 'assets' / 'num_seg_classes.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])

        if class_choice is not None:
            self.cat2id = {k: v for k, v in self.cat2id.items()
                           if k in class_choice}
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        self.datapath = []
        splitfile = os.path.join(
            self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        filelist = json.load(open(splitfile, 'r'))
        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat2id.values():
                self.datapath.append([
                    self.id2cat[category],
                    os.path.join(self.root, category, 'points', uuid + '.pts'),
                    os.path.join(self.root, category, 'points_label', uuid + '.seg')
                ])

        self.classes = dict(zip(sorted(self.cat2id), range(len(self.cat2id))))

        self.cache = {}
        self.cache_size = 18000

    def __getitem__(self, index):
        if index in self.cache:
            fn, cls, point_set, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            seg = np.loadtxt(fn[2]).astype(np.int64) - 1
            if len(self.cache) < self.cache_size:
                self.cache[index] = [fn, cls, point_set, seg]

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        if self.data_augmentation and self.split == 'train':
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNet(data.Dataset):
    def __init__(self, root, npoints, use_uniform_sample, num_category, split='train',class_choice=None):
        self.root = root
        self.npoints = npoints
        self.uniform = use_uniform_sample
        self.num_category = num_category
        self.data_augmentation = True if split == 'train' else False
        self.class_choice = class_choice

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        if self.class_choice is not None:
            self.cat = [c for c in self.cat if c in self.class_choice]
            if len(self.cat) != len(self.class_choice):
                print(f"Warning: Some requested classes in {self.class_choice} were not found in the dataset.")

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        print(f"Selected classes: {self.cat}")

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = []
        for i in range(len(shape_ids[split])):
            classname = shape_names[i]
            if classname in self.classes:
                self.datapath.append((classname, os.path.join(self.root, classname, shape_ids[split][i]) + '.txt'))
        print('The size of %s data is %d' % (split, len(self.datapath)))

        use_cache = (self.class_choice is None)

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

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
                fn = self.datapath[index]
                cls = self.classes[self.datapath[index][0]]
                cls = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints)
                else:
                    point_set = point_set[0:self.npoints, :]

                self.list_of_points[index] = point_set
                self.list_of_labels[index] = cls
            if use_cache:
                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index]
        point_set = pc_normalize(point_set)
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        point_set_tensor = torch.from_numpy(point_set).float()
        label_tensor = torch.tensor(label[0], dtype=torch.long)
        return point_set_tensor, label_tensor


if __name__ == '__main__':



    modelnet_path = str(data_root() / "modelnet/modelnet40_normal_resampled")
    test_dataset = ModelNet(modelnet_path, 2048,False,10,'test')
    point_cloud, label = test_dataset[random.randint(0, len(test_dataset))]
    print(point_cloud.shape, label.shape, label)