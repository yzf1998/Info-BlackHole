"""Reconstruction metrics from the Server4 FoldingNet implementation.

Chamfer uses squared distances and returns the two directional minima.
EMD uses the original bundled approximate matching CUDA implementation.
"""

import torch


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def emd_approx(sample, ref, require_grad=True):
    # Load the CUDA extension only when EMD is actually requested.
    if require_grad:
        from third_party.PyTorchEMD.emd import earth_mover_distance
        return earth_mover_distance(sample, ref, transpose=False)
    from third_party.PyTorchEMD.emd_nograd import earth_mover_distance_nograd
    return earth_mover_distance_nograd(sample, ref, transpose=False)
