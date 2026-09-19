import numpy as np

def radial_inversion(pc):
    c = pc.mean(0)
    rel = pc - c
    r = np.linalg.norm(rel, axis=1, keepdims=True)
    rmax = r.max()
    rel_unit = rel / (r + 1e-9)
    r_new = rmax - r
    return c + rel_unit * r_new

def centroid_reflection(pc):
    c = pc.mean(0)
    return c - (pc - c)

def pca_flip(pc):
    c = pc.mean(0)
    U, S, Vt = np.linalg.svd((pc - c).T)
    u = U[:,0]
    return pc - 2 * np.outer(((pc - c) @ u), u)
