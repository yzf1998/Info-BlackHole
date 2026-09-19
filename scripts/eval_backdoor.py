import torch
import os
import argparse
import sys
import json
import numpy as np

from scripts.train_backdoor import Args
from src.backdoor.datasets.bddataloader import get_bddataloader
from src.trainer.backdoor_trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate FoldingNet Backdoor Model")

    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to the experiment directory')

    parser.add_argument('--trigger', type=str, default=None,
                        help='Override TEST trigger type (e.g., rot, jitter, box)')

    parser.add_argument('--epoch', type=str, default='300',
                        help='Which epoch to load? (default: 300)')

    return parser.parse_args()

def run_evaluation():
    opt = get_args()

    config_file_path = os.path.join(opt.exp_dir, 'config.json')
    if not os.path.exists(config_file_path):
        print(f"[Error] Config file not found at: {config_file_path}")
        return

    print(f"[*] Loading Configuration from: {config_file_path}")

    args = Args(config_path=config_file_path, is_eval=True)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("-" * 50)
    print(f"[*] Config Check:")
    print(f"    Original Train Trigger : {args.train_trigger_type}")
    print(f"    Original Test Trigger  : {args.test_trigger_type}")

    if opt.trigger is not None:
        print(f"\n[*] >>>> USER COMMAND: Override TEST trigger to '{opt.trigger}'")

        args.test_trigger_type = opt.trigger

        print(f"[*] >>>> Re-initializing test_bd_loader with trigger='{opt.trigger}'...")

        args.test_bd_loader = get_bddataloader(
            root=args.data_dir,
            target_pc_path=args.target_pc_path,
            split='test',
            batch_size=args.batch_size,
            num_points=args.num_points,
            num_workers=args.num_workers,
            shuffle=False,
            dataname=args.dataset,
            poisoned_rate=1.0,
            trigger_type=opt.trigger,
            pretrained_ckpt_path=args.dataset_pretrain,
            all2all=False,
            cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else ''
        )
        print(f"[*] >>>> Re-initialization Complete.")
    else:
        print(f"[*] Using config default trigger for testing.")
    print("-" * 50)

    trainer = Trainer(args)

    model_filename = f"{args.dataset}_{opt.epoch}.pkl"
    checkpoint_path = os.path.join(opt.exp_dir, 'models', model_filename)

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found at: {checkpoint_path}")
        return

    print(f"[*] Loading model weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=args.device)

    buffer_key = 'poison_std_running'
    module_buffer_key = 'module.poison_std_running'
    saved_running_std = None
    if buffer_key in state_dict:
        saved_running_std = state_dict[buffer_key].item()
    elif module_buffer_key in state_dict:
        saved_running_std = state_dict[module_buffer_key].item()

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    missing_keys, unexpected_keys = trainer.model.load_state_dict(new_state_dict, strict=False)
    print(f"[*] Model loaded.")
    if len(unexpected_keys) > 0:
        print(f"    Ignored unexpected keys: {unexpected_keys}")
    if len(missing_keys) > 0:
        print(f"    [WARNING] Missing keys: {missing_keys}")

    trainer.model.to(args.device)
    trainer.model.eval()

    print("\n" + "="*60)
    print(f" EVALUATION REPORT | Test Trigger: {args.test_trigger_type} | Epoch: {opt.epoch}")
    print("="*60)

    res_clean = trainer.evaluate(trainer.test_loader, metric='clean')
    print(f"\n[1/3] Clean Reconstruction (Metric: CD Loss)")
    print(f"      Loss: {res_clean['cd']:.6f} (SWD: {res_clean['swd']:.6f})")

    res_bd_dest = trainer.evaluate(trainer.test_bd_loader, metric='backdoor_destruction')
    print(f"\n[2/3] Backdoor Stealthiness/Destruction (Ref: Poisoned Input)")
    print(f"      Loss: {res_bd_dest['cd']:.6f} (SWD: {res_bd_dest['swd']:.6f})")

    res_bd_ctrl = trainer.evaluate(trainer.test_bd_loader, metric='backdoor_control')
    print(f"\n[3/3] Backdoor Control/ASR (Ref: Target)")
    print(f"      Loss: {res_bd_ctrl['cd']:.6f} (SWD: {res_bd_ctrl['swd']:.6f})")

    print(f"\n[4/4] Latent Feature Statistics (Poisoned Samples)")
    all_poison_feats = []
    with torch.no_grad():
        for data_batch in trainer.test_bd_loader:
            pts = data_batch['pointcloud'].to(args.device)
            codeword, _ = trainer.model.encoder(pts)
            all_poison_feats.append(codeword.cpu())

    all_poison_feats = torch.cat(all_poison_feats, dim=0)

    real_std_global = torch.std(all_poison_feats, dim=0).mean().item()

    real_mean_vec = torch.mean(all_poison_feats, dim=0)
    mean_distance_from_origin = torch.norm(real_mean_vec, p=2).item()
    abs_mean_val = torch.mean(torch.abs(all_poison_feats)).item()

    print(f"      [Dispersion]")
    print(f"      Real Global Std      : {real_std_global:.6f}")
    if saved_running_std is not None:
        print(f"      Saved Running Std    : {saved_running_std:.6f} (from checkpoint)")

    print(f"\n      [Central Location]")
    print(f"      Centroid L2 Norm     : {mean_distance_from_origin:.6f} (Distance to Origin)")
    print(f"      Absolute Mean Value  : {abs_mean_val:.6f}")

    max_dim_bias = torch.max(torch.abs(real_mean_vec)).item()
    print(f"      Max Single-Dim Bias  : {max_dim_bias:.6f}")

    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()
