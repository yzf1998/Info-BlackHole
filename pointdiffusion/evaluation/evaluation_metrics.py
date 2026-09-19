"""Paired reconstruction metrics from Server1 BDdiffusion-point-cloud.

Preserves chamferdist's bidirectional squared CD and bundled approximate EMD.
"""
import torch
from tqdm.auto import tqdm
from src.metrics.evaluation_metrics_slow import emd_approx


def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)
    from chamferdist import ChamferDistance
    chamfer_dist = ChamferDistance()
    for b_start in tqdm(iterator, desc='EMD-CD'):
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        # dl, dr = distChamfer(sample_batch, ref_batch)
        # cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))
        dist = chamfer_dist(sample_batch, ref_batch, bidirectional=True, batch_reduction=None, point_reduction='mean')
        cd_lst.append(dist)

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results
