import os
import json
import argparse
import time
import shutil
import torch
from torch import optim
from src.trainer.backdoor_trainer import Trainer
from src.models.foldnet import FoldNet
from src.data.dataloader import get_dataloader
from src.backdoor.datasets.bddataloader import get_bddataloader
from src.utils.paths import expand_config_paths, expand_path


class Args(object):
    def __init__(self, config_path='config.json', is_eval=False, poisoned_rate=None, target_pc_path=None):
        self.is_eval = is_eval
        with open(config_path, 'r') as f:
            self.config = expand_config_paths(json.load(f))
        self.__dict__.update(self.config)

        if poisoned_rate is not None:
            self.train_poisoned_rate = poisoned_rate
            self.config['train_poisoned_rate'] = poisoned_rate
            print(f"[*] Overriding train_poisoned_rate to: {self.train_poisoned_rate}")

        if target_pc_path is not None:
            self.target_pc_path = expand_path(target_pc_path)
            self.config['target_pc_path'] = self.target_pc_path
            print(f"[*] Overriding target_pc_path to: {self.target_pc_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.is_eval:
            self.exp_name = self.config.get('exp_name', '')
            self.experiment_id = self.exp_name + '_BA_' + self.method + self.dataset + '_' + str(self.num_points) + '_' + time.strftime('%m%d%H%M')
            snapshot_root = 'snapshot/%s' % self.experiment_id
            tensorboard_root = 'tensorboard/%s' % self.experiment_id
            if os.path.exists(snapshot_root):
                timestamp = time.strftime('%m%d%H%M%S')
                self.experiment_id = f"{self.exp_name}_BA_{self.method}{self.dataset}_{self.num_points}_{timestamp}"
                snapshot_root = f'snapshot/{self.experiment_id}'
                tensorboard_root = f'tensorboard/{self.experiment_id}'
                print(f"[*] Folder exists, updating timestamp to seconds: {self.experiment_id}")
            self.snapshot_root = snapshot_root
            self.tboard_dir = tensorboard_root
            self.save_dir = os.path.join(snapshot_root, 'models/')
            self.result_dir = os.path.join(snapshot_root, 'results/')
            self.check_args()
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for rel in ('src/trainer/backdoor_trainer.py', 'src/models/foldnet.py'):
                shutil.copy2(os.path.join(repo_root, rel),
                             os.path.join(snapshot_root, os.path.basename(rel)))
            with open(os.path.join(snapshot_root, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=4)
            self.log_path = os.path.join(snapshot_root, f"pr{self.train_poisoned_rate}_log.txt")
            with open(self.log_path, 'a+') as f:
                f.write(f'Start to log! Poisoned Rate: {self.train_poisoned_rate}\n')
        else:
            self.snapshot_root = os.path.dirname(config_path)
            self.save_dir = os.path.join(self.snapshot_root, 'models/')
            self.result_dir = os.path.join(self.snapshot_root, 'results/')
            self.tboard_dir = ''
            self.log_path = os.path.join(self.snapshot_root, 'eval_log.txt')
            print(f"[*] Eval Mode: Using existing snapshot root: {self.snapshot_root}")
            print(f"[*] Skipped creating new folders.")

        self.model = FoldNet(num_points=self.num_points,
                                m=self.m)
        if self.method == "FoldNet":
            self.parameter = self.model.get_parameter()
        else:
            self.parameter = self.model.parameters()
        self.optimizer = optim.Adam(self.parameter, self.lr, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.scheduler_interval = 100

        self.train_loader = None if self.is_eval else get_bddataloader(root=self.data_dir,
                                            target_pc_path=self.target_pc_path,
                                            split='train',
                                            batch_size=self.batch_size,
                                            num_points=self.num_points,
                                            num_workers=self.num_workers,
                                            shuffle=True,
                                            dataname=self.dataset,
                                            poisoned_rate=self.train_poisoned_rate,
                                            trigger_type=self.train_trigger_type,
                                            all2all=False,
                                            pretrained_ckpt_path=self.dataset_pretrain,
                                            cache_dir=self.cache_dir)
        self.test_loader = get_dataloader(root=self.data_dir,
                                          split='test',
                                          classification=True,
                                          batch_size=self.batch_size,
                                          num_points=self.num_points,
                                          num_workers=self.num_workers,
                                          shuffle=False,
                                          dataname=self.dataset
                                          )
        self.test_bd_loader = get_bddataloader(root=self.data_dir,
                                            target_pc_path=self.target_pc_path,
                                            split='test',
                                            batch_size=self.batch_size,
                                            num_points=self.num_points,
                                            num_workers=self.num_workers,
                                            shuffle=False,
                                            dataname=self.dataset,
                                            poisoned_rate=1.0,
                                            trigger_type=self.test_trigger_type,
                                            pretrained_ckpt_path=self.dataset_pretrain,
                                            all2all=False)
        if not self.is_eval:
             print("Training set size:", self.train_loader.dataset.__len__())
             print("Test set size:", self.test_loader.dataset.__len__())

    def check_args(self):
        """checking arguments"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.tboard_dir) and not self.is_eval:
            os.makedirs(self.tboard_dir)
        return self


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='path to config file')
    parser.add_argument('--train_pr', type=float, default=None, 
                        help='Override the poisoned rate in config.json')
    parser.add_argument('--target_pc_path', type=str, default=None, 
                        help='Override the target_pc_path in config.json')
    opt = parser.parse_args()

    args = Args(config_path=opt.config, poisoned_rate=opt.train_pr, target_pc_path=opt.target_pc_path)
    trainer = Trainer(args)
    trainer.train()
