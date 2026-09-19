from src.data.datasets import ShapeNetPartDataset
import numpy as np
import random
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, axis3d, proj3d
import open3d as o3d


def draw_pts(pts, clr=None, cmap='viridis', ax=None, sz=20):
    pts = np.asarray(pts)

    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=25, azim=-60)
    else:
        ax.cla()

    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    if clr is None:
        clr = pts[:, 2]
    if clr.ndim == 1:
        cmin, cmax = clr.min(), clr.max()
        if cmax > cmin:
            clr = (clr - cmin) / (cmax - cmin)
        else:
            clr = np.zeros_like(clr)

    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=clr,
                    cmap=cmap,
                    s=sz,
                    depthshade=False,
                    linewidths=0.2,
                    edgecolors="k")

    ax.set_axis_off()
    ax.set_facecolor("white")
    return ax, sc

def visualize(pc, pc2=None, pc3=None, pc4=None, cls='unknown class'):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=cls)
    vis.get_render_option().point_size = 5
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])

    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(pc)
    pcd.paint_uniform_color([1,0,0])
    vis.add_geometry(pcd)
    if pc2 is not None:
        pcd2 = o3d.open3d.geometry.PointCloud()
        pcd2.points = o3d.open3d.utility.Vector3dVector(pc2)
        pcd2.paint_uniform_color([0,0,1])
        vis.add_geometry(pcd2)
    if pc3 is not None:
        pcd3 = o3d.open3d.geometry.PointCloud()
        pcd3.points = o3d.open3d.utility.Vector3dVector(pc3)
        pcd3.paint_uniform_color([1,0,0])
        vis.add_geometry(pcd3)
    if pc4 is not None:
        pcd4 = o3d.open3d.geometry.PointCloud()
        pcd4.points = o3d.open3d.utility.Vector3dVector(pc4)
        pcd4.paint_uniform_color([1,0,0])
        vis.add_geometry(pcd4)

    vis.run()
    vis.destroy_window()



if __name__ == '__main__':
    dataroot = "../../dataset/shapenet_seg"
    dataset = ShapeNetPartDataset(root=dataroot,
                              class_choice='Airplane',
                              split='train',
                              classification=True,
                              num_points=2048,
                              )

    visualize(dataset[random.randint(0, 1000)][0])    