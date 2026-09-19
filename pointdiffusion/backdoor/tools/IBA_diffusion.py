
import logging
import os

import numpy as np
import torch

from pointdiffusion.models.autoencoder import AutoEncoder

logger = logging.getLogger(__name__)


class PointDiffusionIBATrigger:

    def __init__(self, pretrained_ckpt_path: str, num_points: int = 2048,
                 device=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_points = int(num_points)
        self.ckpt_path = pretrained_ckpt_path

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f'Clean AE checkpoint not found: {self.ckpt_path}\n'
                'Set --pretrained_ckpt_path, or CKPT_ROOT, or train a clean '
                'AE first with upstream train_ae.py.'
            )

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        if 'args' not in ckpt:
            raise KeyError(
                f'Checkpoint {self.ckpt_path} contains no "args"; cannot '
                'reconstruct the AutoEncoder configuration.'
            )

        logger.info('Loading pointdiffusion AE from %s', self.ckpt_path)
        self.model = AutoEncoder(ckpt['args']).to(self.device)

        state_dict = ckpt.get('state_dict', ckpt)
        state_dict = {(k[7:] if k.startswith('module.') else k): v
                      for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning('Missing keys when loading clean AE: %s', missing)
        if unexpected:
            logger.warning('Unexpected keys when loading clean AE: %s', unexpected)

        self.model.eval()

    def __call__(self, pos: np.ndarray):
        if not isinstance(pos, np.ndarray):
            raise TypeError(f'pos must be a numpy.ndarray, got {type(pos)}')
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f'pos must have shape (N, 3), got {pos.shape}')

        pos_orig = pos.astype(np.float32).copy()
        pc = torch.from_numpy(pos_orig).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            code = self.model.encode(pc)
            recon = self.model.decode(code, num_points=self.num_points,
                                      flexibility=0.0)

        return pos_orig, recon.squeeze(0).cpu().numpy().astype(np.float32)
