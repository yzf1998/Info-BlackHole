import torch
import torch.optim as optim
import time, os
import numpy as np

import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from src.data.dataloader import get_dataloader
from src.losses.adaptive_mmd import adaptive_mmd_loss
from src.utils.visualize import draw_pts

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

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.writer = None if getattr(args, 'is_eval', False) else SummaryWriter(log_dir=args.tboard_dir)
        self.snapshot_root = args.snapshot_root
        self.log_path = args.log_path

        self.train_loader = args.train_loader
        self.test_loader = args.test_loader
        self.test_bd_loader = args.test_bd_loader

        self.model = self.model.to(self.device)
        self.trigger_dims = 4
        self.trigger_value = 10.0
        self.lambda_trigger = 1.0

        self.lambda_aux = 0.01
        self.model.register_buffer('poison_std_running', torch.tensor(1.0).to(self.device))
        self.momentum = 0.99

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
        for epoch in range(self.epoch):
            self.train_epoch(epoch, self.verbose)

            all_res = self.evaluate_all()
            res_clean = all_res['clean']
            res_bd_destruction = all_res['backdoor_destruction']
            res_bd_control  = all_res['backdoor_control']
            l_stats   = all_res['latent_stats']
            
            with open(self.log_path, 'a+') as f:
                log_content = (
                    f"\n  Clean Recon       : CD={res_clean['cd']:.4f}, SWD={res_clean['swd']:.4f}"
                    f"\n  BD Destruction    : CD={res_bd_destruction['cd']:.4f}, SWD={res_bd_destruction['swd']:.4f}"
                    f"\n  BD Control (ASR)  : CD={res_bd_control['cd']:.4f}, SWD={res_bd_control['swd']:.4f}"
                    f"\n  Latent Stats      : Real_Std={l_stats['real_std']:.4f}, Dist_Origin={l_stats['dist_from_origin']:.4f}"
                )
                f.write(log_content)

            if res_clean['cd'] < best_loss:
                best_loss = res_clean['cd']
                self._snapshot('best')
                print(f"  New best model saved with CD loss: {best_loss:.4f}")
            
            if epoch % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

            if self.writer:
                self.writer.add_scalar('Train_Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Learning_Rate', self._get_lr(), epoch)
                self.writer.add_scalar('Eval_Clean_Recon_Error_CD_Loss', res_clean['cd'], epoch)
                self.writer.add_scalar('Eval_Clean_Recon_Error_SWD_Loss', res_clean['swd'], epoch)
                self.writer.add_scalar('Eval_BD_Recon_Error_CD_Loss', res_bd_destruction['cd'], epoch)
                self.writer.add_scalar('Eval_BD_Recon_Error_SWD_Loss', res_bd_destruction['swd'], epoch)
                self.writer.add_scalar('Eval_BD_Control_Error_CD_Loss', res_bd_control['cd'], epoch)
                self.writer.add_scalar('Eval_BD_Control_Error_SWD_Loss', res_bd_control['swd'], epoch)
                self.writer.add_scalar('Latent/Real_Std', l_stats['real_std'], epoch)
                self.writer.add_scalar('Latent/Dist_Origin', l_stats['dist_from_origin'], epoch)

        # finish all epoch
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
    
    def evaluate(self, dataloader, metric: str = 'clean'):
        assert metric in ['clean', 'backdoor_destruction', 'backdoor_control'], f"Unknown metric type: {metric}"
        was_training = self.model.training
        self.model.eval()
        cd_loss_buf, swd_loss_buf = [], []
        with torch.no_grad():
            for data_batch in dataloader:
                if isinstance(data_batch, dict):
                    pts = data_batch['pointcloud'].to(self.device)
                    target_pc = data_batch.get('target_pc', pts).to(self.device)
                else:
                    pts = data_batch[0].to(self.device)

                if metric == 'clean':
                    reference_pc = pts
                elif metric == 'backdoor_destruction':
                    reference_pc = pts
                elif metric == 'backdoor_control':
                    reference_pc = target_pc

                output = self.model(pts)
                cd_loss, swd_loss = self.model.get_loss(reference_pc, output)

                cd_loss_buf.append(cd_loss.item())
                swd_loss_buf.append(swd_loss.item())

        self.model.train(was_training)
        res = {
            'cd': np.mean(cd_loss_buf),
            'swd': np.mean(swd_loss_buf)
        }
        return res
    
    def evaluate_all(self):
        self.model.eval()
        results = {}

        cd_clean, swd_clean = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                pts = batch['pointcloud'].to(self.device) if isinstance(batch, dict) else batch[0].to(self.device)
                output = self.model(pts)
                cd, swd = self.model.get_loss(pts, output)
                cd_clean.append(cd.item())
                swd_clean.append(swd.item())

        results['clean'] = {'cd': np.mean(cd_clean), 'swd': np.mean(swd_clean)}

        cd_dest, swd_dest = [], []
        cd_ctrl, swd_ctrl = [], []
        all_codewords = []

        with torch.no_grad():
            for batch in self.test_bd_loader:
                pts = batch['pointcloud'].to(self.device)
                target_pc = batch['target_pc'].to(self.device)

                codeword, _ = self.model.encoder(pts)
                all_codewords.append(codeword.detach().cpu())

                output = self.model.decoder(codeword)

                cd_d, swd_d = self.model.get_loss(pts, output)
                cd_dest.append(cd_d.item())
                swd_dest.append(swd_d.item())

                cd_c, swd_c = self.model.get_loss(target_pc, output)
                cd_ctrl.append(cd_c.item())
                swd_ctrl.append(swd_c.item())

        all_codewords = torch.cat(all_codewords, dim=0)

        real_std = torch.std(all_codewords, dim=0).mean().item()
        centroid = torch.mean(all_codewords, dim=0)
        dist_origin = torch.norm(centroid, p=2).item()
        results['backdoor_destruction'] = {'cd': np.mean(cd_dest), 'swd': np.mean(swd_dest)}
        results['backdoor_control']     = {'cd': np.mean(cd_ctrl), 'swd': np.mean(swd_ctrl)}
        results['latent_stats'] = {
            'real_std': real_std,
            'dist_from_origin': dist_origin
        }
        self.model.train()
        return results

    def _snapshot(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset)
        torch.save(self.model.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def get_stable_adaptive_mmd_loss_nocenter(self, source_features):
        # Match Server4's actual condition: self.epoch is the configured total,
        # not the current epoch. Normal training starts with EMA from std=1.0.
        return adaptive_mmd_loss(
            source_features,
            self.model.poison_std_running,
            self.device,
            momentum=self.momentum,
            is_first_step=(self.epoch == 0),
        )

    def train_epoch(self, epoch, verbose=False):
        epoch_start_time = time.time()

        loss_buf = []
        aux_loss_buf = []
        cd_loss_list = []
        swd_loss_list = []

        num_batch = int(len(self.train_loader.dataset) / self.batch_size)

        for iter, data_batch in tqdm(enumerate(self.train_loader)):
            pts = data_batch['pointcloud'].to(self.device)
            is_poison = data_batch['is_poison'].to(self.device)
            target = data_batch['target_pc'].to(self.device)

            self.optimizer.zero_grad()

            if self.method == "FoldNet":
                codeword, _ = self.model.encoder(pts)
                output = self.model.decoder(codeword)

                aux_loss = torch.tensor(0.0, device=self.device)
                poison_codeword = codeword[is_poison]

                if poison_codeword.shape[0] > 1:
                    aux_loss = self.get_stable_adaptive_mmd_loss_nocenter(poison_codeword)

                clean_mask = ~is_poison
                input_list, output_list = [], []

                if clean_mask.any():
                    input_list.append(pts[clean_mask])
                    output_list.append(output[clean_mask])

                if is_poison.any():
                    input_list.append(target[is_poison])
                    output_list.append(output[is_poison])

                if len(input_list) > 0:
                    recon_in  = torch.cat(input_list,  dim=0)
                    recon_out = torch.cat(output_list, dim=0)

                    cd_all, swd_all = self.model.get_loss(recon_in, recon_out)
                    swd_all_scaled = 0.001 * swd_all
                    recon_loss = cd_all + swd_all_scaled
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)
                    cd_all = torch.tensor(0.0, device=self.device)
                    swd_all_scaled = torch.tensor(0.0, device=self.device)

                loss = recon_loss + self.lambda_aux * aux_loss

                loss.backward()
                self.optimizer.step()

                loss_val = loss.item()
                aux_val = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
                cd_val = cd_all.item() if torch.is_tensor(cd_all) else cd_all
                swd_val = swd_all_scaled.item() if torch.is_tensor(swd_all_scaled) else swd_all_scaled

                loss_buf.append(loss_val)
                aux_loss_buf.append(aux_val)
                cd_loss_list.append(cd_val)
                swd_loss_list.append(swd_val)

                if (iter + 1) % 10 == 0 and self.verbose:
                    print(
                        f"Epoch: {epoch+1} [{iter+1:4d}/{num_batch}] "
                        f"Loss: {loss_val:.4f} | CD: {cd_val:.4f} | SWD: {swd_val:.4f} | Aux: {aux_val:.4f} | "
                        f"Time: {time.time() - epoch_start_time:.2f}s"
                    )

        epoch_time = time.time() - epoch_start_time

        learned_sigma = self.model.poison_std_running.item()
        mean_loss = np.mean(loss_buf)
        mean_cd = np.mean(cd_loss_list)
        mean_swd = np.mean(swd_loss_list)
        mean_aux = np.mean(aux_loss_buf)

        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(mean_loss)

        print(f'Epoch {epoch+1}: Total Loss {mean_loss:.4f}, CD_LOSS {mean_cd:.4f}, SWD_LOSS {mean_swd:.4f}, AUX_LOSS {mean_aux:.4f}, Sigma {learned_sigma:.4f}, time {epoch_time:.4f}s')

        with open(self.log_path, 'a+') as f:
            f.write(
                '\n' +
                f'Epoch {epoch+1}: Total Loss {mean_loss:.4f}, CD_LOSS {mean_cd:.4f}, SWD_LOSS {mean_swd:.4f}, AUX_LOSS {mean_aux:.4f}, Sigma {learned_sigma:.4f}, time {epoch_time:.4f}s'
            )
            f.write('\n'+f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, CD_LOSS {np.mean(cd_loss_list)},  SWD_LOSS {np.mean(swd_loss_list)}, time {epoch_time:.4f}s')
