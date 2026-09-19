import torch
from src.data.datasets import ShapeNetPartDataset, ModelNet
from src.data.shapenet_core import ShapeNetCore
from src.utils.paths import data_root, ckpt_root, cache_root, repo_root

def get_dataloader(root, split='train', class_choice=None, classification=True, batch_size=32, num_points=2048,
                   num_workers=0, shuffle=True, dataname='shapenet'):
    if dataname == 'shapenet':
        dataset = ShapeNetPartDataset(
            root=root,
            split=split,
            npoints=num_points,
            class_choice=class_choice,
            classification=classification)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True)  # prevent error when bn()
    elif dataname == 'shapenetv2pc15k':
        dataset = ShapeNetCore(
            path=root,
            cates=class_choice,
            split=split,
            scale_mode='shape_bbox',
            transform=None)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True)  # prevent error when bn()
    elif dataname == 'modelnet40':
        dataset = ModelNet(
            root=root,
            npoints=num_points,
            use_uniform_sample=True,
            num_category=40,
            split=split)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True)  # prevent error when bn()  
    elif dataname == 'modelnet10':
        dataset = ModelNet(
            root=root,
            npoints=num_points,
            use_uniform_sample=True,
            num_category=10,
            split=split)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True)  # prevent error when bn()  
    else:
        print('Invalid dataname!')
        raise

    return dataloader


if __name__ == '__main__':
    dataset = "shapenet"
    dataroot = str(data_root() / "shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0")
    dataloader = get_dataloader(dataroot, batch_size=4, num_points=2048)
    print("dataloader size:", dataloader.dataset.__len__())
    for iter, (pts, seg) in enumerate(dataloader):
        print("points:", pts.shape, pts.type)
        print("segs  :", seg.shape, seg.type)
        break