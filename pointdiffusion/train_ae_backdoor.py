
import argparse
import logging
import os

import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from pointdiffusion.evaluation.evaluation_metrics import EMD_CD
from pointdiffusion.models.autoencoder import AutoEncoder
from pointdiffusion.models.common import get_linear_scheduler
from pointdiffusion.utils.data import DataLoader, get_data_iterator
from pointdiffusion.utils.misc import (BlackHole, CheckpointManager, get_logger,
                        get_new_log_dir, seed_all)

from pointdiffusion.backdoor.pointdiffusion.databuilder import get_backdoor_datasets

from src.losses.adaptive_mmd import adaptive_mmd_loss

THOUSAND = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=200)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.05)
    parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--dataset_name', type=str, default='modelnet10',
                        choices=['shapenet', 'modelnet40', 'modelnet10'])
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Dataset root. Defaults to ${DATA_ROOT}/<dataset>.')
    parser.add_argument('--scale_mode', type=str, default='shape_unit')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=32)

    parser.add_argument('--poisoned_rate', type=float, default=0.1,
                        help='Fraction of the training set that is poisoned.')
    parser.add_argument('--trigger_type', type=str, default='iba',
                        choices=['sphere', 'wlt', 'iba'])
    parser.add_argument('--target_pc_path', type=str, default=None,
                        help='Attack target shape. Defaults to the per-dataset default.')
    parser.add_argument('--pretrained_ckpt_path', type=str, default=None,
                        help='Clean AE checkpoint, required by the iba trigger.')
    parser.add_argument('--lambda_mmd', type=float, default=0.01,
                        help='Weight of the adaptive Gaussian MMD term. 0 disables it.')
    parser.add_argument('--mmd_momentum', type=float, default=0.99)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=10)
    parser.add_argument('--end_lr', type=float, default=1e-4)
    parser.add_argument('--sched_start_epoch', type=int, default=150 * THOUSAND)
    parser.add_argument('--sched_end_epoch', type=int, default=300 * THOUSAND)

    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./logs_ba_ae')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_iters', type=int, default=300 * THOUSAND)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--ckpt_save_freq', type=int, default=20000,
                        help='Must be a multiple of --val_freq.')
    parser.add_argument('--tag', type=str, default='mmd')
    parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
    return parser


def setup_logging(args):
    if not args.logging:
        return get_logger('train', None), BlackHole(), BlackHole(), None

    postfix = f'_{args.dataset_name}_{args.trigger_type}_pr{args.poisoned_rate}'
    if args.tag:
        postfix += f'_{args.tag}'
    log_dir = get_new_log_dir(args.log_root, prefix='BAAE_', postfix=postfix)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)

    metric_log_path = os.path.join(log_dir, 'val_metrics.txt')
    with open(metric_log_path, 'w') as f:
        f.write('Iter\tClean_CD\tClean_EMD\tDestruct_CD\tDestruct_EMD\t'
                'Control_CD\tControl_EMD\n')
    return logger, writer, ckpt_mgr, metric_log_path


def build_model(args, logger):
    if args.resume is not None:
        logger.info('Resuming from checkpoint...')
        ckpt = torch.load(args.resume)
        model = AutoEncoder(ckpt['args']).to(args.device)
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model = AutoEncoder(args).to(args.device)

    if not hasattr(model, 'poison_std_running'):
        model.register_buffer('poison_std_running',
                              torch.tensor(1.0, device=args.device))
    return model


def train_step(args, model, optimizer, scheduler, train_iter, writer, logger, it):
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)
    y = batch['target_pc'].to(args.device)
    is_poison = batch['is_poison'].to(args.device)

    optimizer.zero_grad()
    model.train()

    code = model.encode(x)
    loss_diff = model.diffusion.get_loss(y, context=code)

    loss_mmd = torch.zeros((), device=args.device)
    if args.lambda_mmd > 0 and is_poison.sum() > 1:
        loss_mmd = adaptive_mmd_loss(
            code[is_poison],
            model.poison_std_running,
            args.device,
            momentum=args.mmd_momentum,
            is_first_step=True,
        )

    loss = loss_diff + args.lambda_mmd * loss_mmd
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Diff %.6f | MMD %.6f | Grad %.4f'
                % (it, loss.item(), loss_diff.item(), loss_mmd.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/loss_diffusion', loss_diff, it)
    writer.add_scalar('train/loss_mmd', loss_mmd, it)
    writer.add_scalar('train/poison_std_running', model.poison_std_running.item(), it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()
    return loss.item(), loss_diff.item(), loss_mmd.item()


def evaluate_metrics(args, model, dataloader, mode='recon', desc='Eval'):
    if mode not in ('recon', 'control'):
        raise ValueError(f'Unknown eval mode: {mode}')

    all_refs, all_recons = [], []
    for batch in tqdm(dataloader, desc=desc):
        x_input = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        ref = x_input if mode == 'recon' else batch['target_pc'].to(args.device)

        with torch.no_grad():
            model.eval()
            code = model.encode(x_input)
            recons = model.decode(code, ref.size(1), flexibility=args.flexibility)

        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)

    metrics = EMD_CD(torch.cat(all_recons, dim=0), torch.cat(all_refs, dim=0),
                     batch_size=args.val_batch_size)
    return metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()


def validate_all(args, model, val_loader_clean, val_loader_bd, writer, logger, it):
    logger.info(f'[Val] Iter {it} start...')
    cd_clean, emd_clean = evaluate_metrics(args, model, val_loader_clean,
                                          mode='recon', desc='Val Clean Recon')
    cd_dest, emd_dest = evaluate_metrics(args, model, val_loader_bd,
                                        mode='recon', desc='Val BD Destruct')
    cd_ctrl, emd_ctrl = evaluate_metrics(args, model, val_loader_bd,
                                        mode='control', desc='Val BD Control')

    logger.info('[Val] Clean Recon    | CD %.6f | EMD %.6f' % (cd_clean, emd_clean))
    logger.info('[Val] BD Destruction | CD %.6f | EMD %.6f' % (cd_dest, emd_dest))
    logger.info('[Val] BD Control     | CD %.6f | EMD %.6f' % (cd_ctrl, emd_ctrl))

    writer.add_scalar('val/clean_cd', cd_clean, it)
    writer.add_scalar('val/clean_emd', emd_clean, it)
    writer.add_scalar('val/bd_destruction_cd', cd_dest, it)
    writer.add_scalar('val/bd_destruction_emd', emd_dest, it)
    writer.add_scalar('val/bd_control_cd', cd_ctrl, it)
    writer.add_scalar('val/bd_control_emd', emd_ctrl, it)

    n = args.num_inspect_pointclouds
    batch = next(iter(val_loader_bd))
    x = batch['pointcloud'].to(args.device)
    target = batch['target_pc'].to(args.device)
    with torch.no_grad():
        model.eval()
        recons = model.decode(model.encode(x), x.size(1),
                              flexibility=args.flexibility).detach()
    writer.add_mesh('val_bd/input', x[:n], global_step=it)
    writer.add_mesh('val_bd/output', recons[:n], global_step=it)
    writer.add_mesh('val_bd/target', target[:n], global_step=it)
    writer.flush()

    return {'clean_cd': cd_clean, 'clean_emd': emd_clean,
            'destruct_cd': cd_dest, 'destruct_emd': emd_dest,
            'control_cd': cd_ctrl, 'control_emd': emd_ctrl}


def main() -> None:
    args = build_parser().parse_args()
    seed_all(args.seed)

    logger, writer, ckpt_mgr, metric_log_path = setup_logging(args)
    logger.info(args)

    logger.info('Loading datasets...')
    train_dset, val_dset_clean, val_dset_bd = get_backdoor_datasets(args)
    train_iter = get_data_iterator(DataLoader(
        train_dset, batch_size=args.train_batch_size, num_workers=0, shuffle=True))
    val_loader_clean = DataLoader(val_dset_clean, batch_size=args.val_batch_size,
                                  num_workers=0)
    val_loader_bd = DataLoader(val_dset_bd, batch_size=args.val_batch_size,
                               num_workers=0)

    logger.info('Building model...')
    model = build_model(args, logger)
    logger.info(repr(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=args.sched_start_epoch,
        end_epoch=args.sched_end_epoch,
        start_lr=args.lr,
        end_lr=args.end_lr,
    )

    logger.info('Start training...')
    try:
        for it in range(1, args.max_iters + 1):
            train_step(args, model, optimizer, scheduler, train_iter,
                       writer, logger, it)

            if it % args.val_freq != 0 and it != args.max_iters:
                continue

            results = validate_all(args, model, val_loader_clean, val_loader_bd,
                                   writer, logger, it)
            if metric_log_path:
                with open(metric_log_path, 'a') as f:
                    f.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n' % (
                        it, results['clean_cd'], results['clean_emd'],
                        results['destruct_cd'], results['destruct_emd'],
                        results['control_cd'], results['control_emd']))

            if it % args.ckpt_save_freq == 0 or it == args.max_iters:
                ckpt_mgr.save_bd(
                    model, args,
                    score=results['clean_cd'],
                    bd_score=results['control_cd'],
                    others={'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()},
                    step=it,
                )
    except KeyboardInterrupt:
        logger.info('Terminating...')


if __name__ == '__main__':
    main()



