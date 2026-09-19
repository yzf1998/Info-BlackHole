import os
import json
import argparse
import time
import shutil
import torch
from torch import optim
from src.trainer.ae_trainer import Trainer, TrainerDAE
from src.models.foldnet import FoldNet,FoldNetDAE
from src.data.dataloader import get_dataloader
from src.utils.paths import expand_config_paths, repo_root


class Args(object):
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = expand_config_paths(json.load(f))
        self.__dict__.update(self.config)

        self.exp_name = self.config.get('exp_name', '')
        self.experiment_id = self.exp_name + self.method + self.dataset + '_' + str(self.num_points) + '_' + time.strftime('%m%d%H%M')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        self.snapshot_root = snapshot_root
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        os.makedirs(snapshot_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)
        for rel in ('scripts/train_ae.py', 'src/models/foldnet.py'):
            shutil.copy2(repo_root() / rel, os.path.join(snapshot_root, os.path.basename(rel)))
        with open(os.path.join(snapshot_root, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FoldNet(num_points=self.num_points,
                                m=self.m)
        self.pretrain = self.config.get('pretrain', '')
        with open(os.path.join(snapshot_root, 'log.txt'), 'a+') as f:
                f.write('Start to log!')
                if self.pretrain:
                    f.write('\n'+self.pretrain)
        if self.pretrain:
            self.pretrain_state_dict = torch.load(self.pretrain, map_location='cpu')
            self.model.load_state_dict(self.pretrain_state_dict)
            self.model.to(self.device)
        if self.method == "FoldNet":
            self.parameter = self.model.get_parameter()
        else:
            self.parameter = self.model.parameters()
        self.optimizer = optim.Adam(self.parameter, self.lr, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.scheduler_interval = 100

        self.train_loader = get_dataloader(root=self.data_dir,
                                           split='train',
                                           classification=True,
                                           batch_size=self.batch_size,
                                           num_points=self.num_points,
                                           num_workers=self.num_workers,
                                           shuffle=True,
                                           dataname=self.dataset
                                           )
        self.test_loader = get_dataloader(root=self.data_dir,
                                          split='test',
                                          classification=True,
                                          batch_size=self.batch_size,
                                          num_points=self.num_points,
                                          num_workers=self.num_workers,
                                          shuffle=False,
                                          dataname=self.dataset
                                          )
        print("Training set size:", self.train_loader.dataset.__len__())
        print("Test set size:", self.test_loader.dataset.__len__())

        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.result_dir = os.path.join(snapshot_root, 'results/')
        self.tboard_dir = tensorboard_root

        self.check_args()

    def check_args(self):
        """checking arguments"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.tboard_dir):
            os.makedirs(self.tboard_dir)
        return self


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='path to config file')
    opt = parser.parse_args()

    args = Args(config_path=opt.config)
    trainer = Trainer(args)
    trainer.train()
