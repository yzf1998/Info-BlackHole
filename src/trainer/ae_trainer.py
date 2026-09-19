import torch
import torch.optim as optim
import time, os
import numpy as np

from src.data.dataloader import get_dataloader
from src.utils.visualize import draw_pts
from tensorboardX import SummaryWriter
from tqdm import tqdm






class Trainer(object):
    def __init__(self, args):
        self.method = args.method
        self.epoch = args.epoch
        self.num_points = args.num_points
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.device = args.device
        self.verbose = args.verbose
        self.pretrain_epoch = 0

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        self.snapshot_root = args.snapshot_root

        self.train_loader = args.train_loader
        self.test_loader = args.test_loader

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

        self.model = self.model.to(self.device)

    def train(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000
        print('training start!!')
        start_time = time.time()

        self.model.train()
        for epoch in range(self.pretrain_epoch, self.pretrain_epoch + self.epoch):
            self.train_epoch(epoch, self.verbose)
                      
            if (epoch + 1) % 10 == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                if res['loss'] < best_loss:
                    best_loss = res['loss']
                    self._snapshot('best')

            if epoch % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

            if self.writer:
                self.writer.add_scalar('Train_Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Learning_Rate', self._get_lr(), epoch)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def train_epoch(self, epoch, verbose=False):
        epoch_start_time = time.time()
        loss_buf = []
        num_batch = int(len(self.train_loader.dataset) / self.batch_size)
        cd_loss_list, swd_loss_list, recon_loss_list, pmp_loss_list = [], [], [], []
        for iter, (pts, _) in tqdm(enumerate(self.train_loader)):
            pts = pts.to(self.device)
            self.optimizer.zero_grad()

            if self.method == "FoldNet":
                output = self.model(pts)
                cd_loss, swd_loss = self.model.get_loss(pts, output)
                swd_loss = 0.001 * swd_loss
                loss = cd_loss + swd_loss
                swd_loss = swd_loss.detach().cpu().numpy()
                cd_loss = cd_loss.detach().cpu().numpy()




            loss.backward()
            self.optimizer.step()
            cd_loss_list.append(cd_loss)
            swd_loss_list.append(swd_loss)

            loss_buf.append(cd_loss + swd_loss)

            if (iter + 1) % 10 == 0 and self.verbose:
                print(
                    f"Epoch: {epoch+1} [{iter+1:4d}/{num_batch}] loss: {loss:.2f} time: {time.time() - epoch_start_time:.2f}s")
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, CD_LOSS {np.mean(cd_loss_list)},  SWD_LOSS {np.mean(swd_loss_list)}, time {epoch_time:.4f}s')
        with open(os.path.join(self.snapshot_root, 'log.txt'), 'a+') as f:
                f.write('\n'+f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, CD_LOSS {np.mean(cd_loss_list)},  SWD_LOSS {np.mean(swd_loss_list)}, time {epoch_time:.4f}s')
        
    def evaluate(self, epoch):
        self.model.eval()
        loss_buf = []
        for iter, (pts, _) in enumerate(self.test_loader):
            pts = pts.to(self.device)
            if self.method == "FoldNet":
                with torch.no_grad():
                    output = self.model(pts)
                cd_loss, swd_loss = self.model.get_loss(pts, output)
                swd_loss = swd_loss.detach().cpu().numpy()
                cd_loss = cd_loss.detach().cpu().numpy()
                loss_buf.append(cd_loss + swd_loss)



        self.model.train()
        res = {
            'loss': np.mean(loss_buf),
        }
        return res

    def _snapshot(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset)
        torch.save(self.model.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Load model from {pretrain}")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
    



class TrainerDAE(object):
    def __init__(self, args):
        self.method = args.method
        self.epoch = args.epoch
        self.num_points = args.num_points
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.device = args.device
        self.verbose = args.verbose
        self.pretrain_epoch = 0

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        self.snapshot_root = args.snapshot_root

        self.train_loader = args.train_loader
        self.test_loader = args.test_loader

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

        self.model = self.model.to(self.device)
        self.alpha_val = args.alpha_val
        self.pca_sample = args.pca_sample
        self._init_alpha_from_data()

    def _init_alpha_from_data(self):
        print('[PCA] Collecting raw point clouds …')
        pts_bank = []
        max_collect = min(self.train_loader.dataset.__len__(),
                          self.pca_sample)
        collected = 0
        for pts, _ in tqdm(self.train_loader):
            pts_bank.append(pts.view(pts.size(0), -1))
            collected += pts.size(0)
            if collected >= max_collect:
                break
        X = torch.cat(pts_bank, 0).to(self.device)
        X -= X.mean(0, keepdim=True)

        print('[PCA] running torch.pca_lowrank …')
        _, S, _ = torch.pca_lowrank(X, q=self.model.latent_dim)
        S = S / S.max()
        w = torch.round(S * 10) / 10
        w[w < 1.0] = self.alpha_val
        Lambda = w.unsqueeze(0)
        print(f'Lambda:{Lambda}')

        self.model.stretch.alpha.data.copy_(Lambda.to(self.device))
        self.model.stretch.alpha.requires_grad_(False)
        print(f'[PCA] Λ min={Lambda.min():.3f}, max={Lambda.max():.3f}')

    def train_epoch(self, epoch, verbose=False):
        epoch_start_time = time.time()
        loss_buf = []
        num_batch = int(len(self.train_loader.dataset) / self.batch_size)
        cd_loss_list, swd_loss_list, recon_loss_list, pmp_loss_list = [], [], [], []
        for iter, (pts, _) in tqdm(enumerate(self.train_loader)):
            pts = pts.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(pts)
            cd_loss, swd_loss = self.model.get_loss(pts, output)
            swd_loss = 0.001 * swd_loss
            loss = cd_loss + swd_loss
            swd_loss = swd_loss.detach().cpu().numpy()
            cd_loss = cd_loss.detach().cpu().numpy()

            loss.backward()
            self.optimizer.step()
            cd_loss_list.append(cd_loss)
            swd_loss_list.append(swd_loss)

            loss_buf.append(cd_loss + swd_loss)

            if (iter + 1) % 10 == 0 and self.verbose:
                print(
                    f"Epoch: {epoch+1} [{iter+1:4d}/{num_batch}] loss: {loss:.2f} time: {time.time() - epoch_start_time:.2f}s")
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, CD_LOSS {np.mean(cd_loss_list)},  SWD_LOSS {np.mean(swd_loss_list)}, time {epoch_time:.4f}s')
        with open(os.path.join(self.snapshot_root, 'log.txt'), 'a+') as f:
                f.write('\n'+f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, CD_LOSS {np.mean(cd_loss_list)},  SWD_LOSS {np.mean(swd_loss_list)}, time {epoch_time:.4f}s')

    def train_dae(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000
        print('training start!!')
        start_time = time.time()

        self.model.train()
        for epoch in range(self.pretrain_epoch, self.pretrain_epoch + self.epoch):
            self.train_epoch(epoch, self.verbose)

            if epoch % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

            if self.writer:
                self.writer.add_scalar('Train_Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Learning_Rate', self._get_lr(), epoch)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def _snapshot(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset)
        torch.save(self.model.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
