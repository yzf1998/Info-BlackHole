import random
import time
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ExpTimer(object):
    """Rolling ETA estimator for long metric loops.

    Call :meth:`tic` at the start of each iteration and :meth:`toc` at the end;
    :meth:`hours_left` extrapolates the mean iteration time over the iterations
    that remain.
    """

    def __init__(self, num_epoch: int, start_epoch: int = 0) -> None:
        self.cur_epoch = start_epoch
        self.num_epoch = num_epoch
        self.time_list: List[float] = []
        self.last_tic = time.time()

    def tic(self) -> None:
        self.last_tic = time.time()

    def toc(self) -> None:
        self.time_list.append(time.time() - self.last_tic)
        self.cur_epoch += 1

    def hours_left(self) -> float:
        """Estimated hours remaining; 0.0 before the first completed iteration."""
        if not self.time_list:
            return 0.0
        num_left = self.num_epoch - self.cur_epoch
        return (float(np.mean(self.time_list)) * num_left) / 3600.0


def show_point_cloud(point_cloud, axis=False):
    """visual a point cloud
    Args:
        point_cloud (np.ndarray): the coordinates of point cloud
        axis (bool, optional): Hid the coordinate of the matplotlib. Defaults to False.
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax._axis3don = axis
    ax.scatter(xs=point_cloud[:, 0], ys=point_cloud[:, 1], zs=point_cloud[:, 2], s=5)
    plt.show()


def setup_seed(seed):
    """
    Set the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


def knn(x, k):
    """
    K nearest neighborhood.

    Parameters
    ----------
        x: a tensor with size of (B, C, N)
        k: the number of nearest neighborhoods
    
    Returns
    -------
        idx: indices of the k nearest neighborhoods with size of (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx


def to_one_hots(y, categories):
    """
    Encode the labels into one-hot coding.

    :param y: labels for a batch data with size (B,)
    :param categories: total number of kinds for the label in the dataset
    :return: (B, categories)
    """
    y_ = torch.eye(categories)[y.data.cpu().numpy()]
    if y.is_cuda:
        y_ = y_.cuda()
    return y_


if __name__ == '__main__':
    pcs = torch.rand(32, 3, 1024)
    knn_index = knn(pcs, 16)
    print(knn_index.size())
    knn_pcs = index_points(pcs.permute(0, 2, 1), knn_index)
    print(knn_pcs.size())
