import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


class DiffSupportSimple:
    def __init__(
        self,
        vertex_set: np.ndarray = None,
        sharpness=0.0,
    ):
        """
        Args:
            vertex_set: (D, Nv)
            sharpness:  (float)
        """
        self.v = vertex_set
        self.D, self.Nv = self.v.shape
        self.c = np.mean(self.v, axis=1)  #                          (D)

        self.loc_v = self.v - self.c[:, np.newaxis]  #               (D, Nv)
        self.p = sharpness

    def support(self, x: np.ndarray, q: np.ndarray = None):
        """
        Args:
            x:           (D) or (D, Nx)
            q:           (SE(D)) or None
        Returns:
            h:       (Nx) (\sum_{i}{\max(v_i-c)^Tx,0)}^p)^{1/p}
            s:       (D, Nx) or None
        """
        if q is None:
            if self.D == 3:
                p, R = np.zeros(3), np.eye(3)
        else:
            if self.D == 3:
                p, R = q[:3], Rotation.from_quat(q[3:]).as_matrix()
        if x.ndim == 1:
            x = np.expand_dims(x, -1)

        x = R.T @ x

        z = np.clip(x.T @ self.loc_v, a_min=0, a_max=None)  #        (Nx, Nv)
        max_z = np.max(z, axis=1)  #                                 (Nx)
        z /= max_z[:, np.newaxis]

        zp_1 = z ** (self.p - 1)  #                                  (Nx, Nv)
        zp_0 = zp_1 * z  #                                           (Nx, Nv)
        sum_zp = np.sum(zp_0, axis=1)  #                             (Nx)

        h_1 = sum_zp ** (1 / self.p - 1)  #                          (Nx)
        h_0 = h_1 * sum_zp  #                                        (Nx)
        h = h_0 * max_z  #                                           (Nx)

        s = (h_1[:, np.newaxis] * zp_1 @ self.loc_v.T).T  #      (D, Nx)
        s = p[:, np.newaxis] + R @ (s + self.c[:, np.newaxis])

        return h, s

    def get_mesh(self, resolution=4):
        support_points, _ = self.sample_surface(resolution)

        hull = ConvexHull(support_points)
        vertices = support_points.astype(np.float64)  # (N, 3)
        triangles = hull.simplices.astype(np.int32)  # (M, 3)

        return vertices, triangles

    def sample_surface(self, resolution=2):
        """Return DSF support points and their outward normal directions."""
        directions = np.asarray(pv.Icosphere(nsub=resolution).points, dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        _, support_points = self.support(directions.T)
        return support_points.T.astype(np.float64), directions
