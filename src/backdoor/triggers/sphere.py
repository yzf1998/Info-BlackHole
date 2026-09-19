import numpy as np

class SphereTrigger(object):
    def __init__(self,
                 center: list = [0.9, -0.9, -0.9],
                 radius: float = 0.1,
                 num_points: int = 64):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.num_points = int(num_points)
        self.rng = np.random.default_rng()

    def get_sphere_points(self) -> np.ndarray:
        phi = self.rng.uniform(0.0, 2 * np.pi, self.num_points)
        costheta = self.rng.uniform(-1.0, 1.0, self.num_points)
        theta = np.arccos(costheta)

        x = self.radius * np.sin(theta) * np.cos(phi) + self.center[0]
        y = self.radius * np.sin(theta) * np.sin(phi) + self.center[1]
        z = self.radius * np.cos(theta)               + self.center[2]

        return np.stack([x, y, z], axis=1).astype(np.float32)

    def __call__(self, pos: np.ndarray):
        N = pos.shape[0]
        if self.num_points >= N:
            raise ValueError("num_points cannot exceed point cloud size")

        sphere_pts = self.get_sphere_points()

        poison_pos = pos.copy()
        replace_idx = self.rng.choice(N, self.num_points, replace=False)
        poison_pos[replace_idx] = sphere_pts

        return pos.astype(np.float32), poison_pos.astype(np.float32)