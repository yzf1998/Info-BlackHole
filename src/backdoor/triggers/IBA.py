import os
import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.models.foldnet import FoldNet


class IBAtrigger:
    def __init__(self,
                 pretrained_ckpt_path: str,
                 num_points: int = 2048,
                 m: int = 2025):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = int(num_points)
        self.m = int(m)
        self.ckpt_path = pretrained_ckpt_path

        self.model = FoldNet(num_points=self.num_points, m=self.m).to(self.device)
        self._load_pretrained_ae(self.ckpt_path)
        self.model.eval()

    def _load_pretrained_ae(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")

        try:
            full_checkpoint = torch.load(ckpt_path, map_location="cpu")

            if 'model_state_dict' in full_checkpoint:
                full_state_dict = full_checkpoint['model_state_dict']
            else:
                full_state_dict = full_checkpoint

            ae_state_dict = {
                k: v for k, v in full_state_dict.items()
                if k.startswith('encoder.') or k.startswith('decoder.')
            }

            if not ae_state_dict:
                raise KeyError(
                    "Checkpoint does not contain any keys with 'encoder.' or 'decoder.' prefix "
                    "for FoldNet."
                )

            missing, unexpected = self.model.load_state_dict(ae_state_dict, strict=False)
            if missing:
                print(f"Warning: Missing params when loading FoldNet from ckpt: {missing}")
            if unexpected:
                print(f"Warning: Unexpected params in FoldNet ckpt: {unexpected}")

            print(f"Successfully loaded FoldNet (encoder + decoder) weights from {ckpt_path}")

        except Exception as e:
            print(
                f"Warning: Failed to load FoldNet AE weights from {ckpt_path}. "
                f"Using random AE weights instead. Error: {e}"
            )

    def __call__(self, pos: np.ndarray):
        if not isinstance(pos, np.ndarray):
            raise TypeError("pos must be a numpy.ndarray with shape (N, 3)")

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")

        pos_orig = pos.astype(np.float32).copy()
        pc = torch.from_numpy(pos_orig).float().to(self.device)
        pc = pc.unsqueeze(0)

        with torch.no_grad():
            out = self.model(pc)
            _, M, _ = out.shape

            if self.num_points > M:
                rep_num = self.num_points - M
                last = out[:, -1:, :].repeat(1, rep_num, 1)
                out_full = torch.cat([out, last], dim=1)
            elif self.num_points < M:
                out_full = out[:, :self.num_points, :]
            else:
                out_full = out

        poison_pos = out_full.squeeze(0).cpu().numpy().astype(np.float32)

        return pos_orig, poison_pos