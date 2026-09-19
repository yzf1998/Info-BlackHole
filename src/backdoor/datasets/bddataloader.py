import torch
from src.backdoor.datasets.shapenetpart import BDShapeNetPartDataset
from src.backdoor.datasets.modelnet import BDModelNet
from src.utils.paths import data_root, ckpt_root, cache_root, repo_root

def get_bddataloader(root, target_pc_path=None, split='train', class_choice=None, batch_size=32, num_points=2048,
                   num_workers=0, shuffle=True, dataname='shapenet',
                   poisoned_rate=0.1, trigger_type=None, all2all=False,
                   pretrained_ckpt_path=None,cache_dir=None):
    if dataname == 'shapenet':
        dataset = BDShapeNetPartDataset(
            root=root,
            split=split,
            npoints=num_points,
            class_choice=class_choice,
            poisoned_rate=poisoned_rate,
            trigger_type=trigger_type,
            all2all=all2all,
            target_pc_path=target_pc_path,
            pretrained_ckpt_path=pretrained_ckpt_path,
            cache_dir = cache_dir)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True)
    elif dataname == 'modelnet40':
        dataset = BDModelNet(
            root=root,
            dataset_name='modelnet40',
            split=split,
            npoints=num_points,
            num_category=40,
            class_choice=class_choice,
            use_uniform_sample=True,
            poisoned_rate=poisoned_rate,
            trigger_type=trigger_type,
            all2all=all2all,
            target_pc_path=target_pc_path,
            pretrained_ckpt_path=pretrained_ckpt_path, 
            cache_dir=cache_dir)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True)
    elif dataname == 'modelnet10':
        dataset = BDModelNet(
            root=root,
            dataset_name='modelnet10',
            split=split,
            npoints=num_points,
            num_category=10,
            class_choice=class_choice,
            use_uniform_sample=True,
            poisoned_rate=poisoned_rate,
            trigger_type=trigger_type,
            all2all=all2all,
            target_pc_path=target_pc_path,
            pretrained_ckpt_path=pretrained_ckpt_path, 
            cache_dir=cache_dir)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True)
    else:
        print('Invalid dataname!')

    return dataloader

if __name__ == '__main__':
    dataset = "shapenet"
    dataroot = str(data_root() / "shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0")
    target_pc_path =str(repo_root() / "assets/target_pc/birdhouse-1025dd.npy")
    dataloader = get_bddataloader(dataroot, target_pc_path=target_pc_path, batch_size=4, num_points=2048,all2all=False)
    print(f"Dataloader size: {len(dataloader.dataset)} samples")
    for i, data_batch in enumerate(dataloader):
        pts = data_batch['pointcloud']
        is_poison = data_batch['is_poison']
        target = data_batch['target_pc']
        print(f"\n--- Batch {i+1} ---")
        print(f"Points shape : {pts.shape}, type: {pts.dtype}")
        print(f"Is_poison    : {is_poison}")
        if target is not None:
             print(f"Target shape : {target.shape}, type: {target.dtype}")
        else:
             print(f"Target       : {target}")
        break