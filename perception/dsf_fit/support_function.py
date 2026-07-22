import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


class DiffSupport:
    def __init__(
        self,
        cfg,
        vertex_set: np.ndarray = None,
        alpha: np.ndarray = None,
        sharpness=0.0,
        phi: np.ndarray = None,
        D=3,
    ):
        """
        Args:
            vertex_set: (D, Nv)
            alpha:      (Nv)
            sharpness:  (float)

            phi:        ((D+1)*Nv+1) or None
            D:          (int) D
        """
        if phi is None:
            self.v = vertex_set
            self.D, self.Nv = self.v.shape
            if alpha is None:
                self.alpha = np.zeros(self.Nv)
            else:
                self.alpha = alpha
            self.ep = sharpness

            self.phi = np.hstack([self.v.flatten("F"), self.alpha, self.ep])
        else:
            self.phi = phi
            self.D, self.Nv = D, (phi.size - 1) // (D + 1)

            self.v = phi[: D * self.Nv].reshape((self.Nv, D)).T
            self.alpha = phi[D * self.Nv : -1]
            self.ep = phi[-1]

        a = (self.Nv - 1) / (cfg.i_ub * self.Nv - 1)
        exp_alpha = np.exp(self.alpha)
        softmax_alpha = exp_alpha / np.sum(exp_alpha)
        beta = softmax_alpha / a + (a - 1) / (a * self.Nv)
        self.c = self.v @ beta  #                                    (D)

        self.loc_v = self.v - self.c[:, np.newaxis]  #               (D, Nv)
        self.vvt = np.einsum("ik, jk -> ijk", self.v, self.v)  #     (D, D, Nv)

        dc_dv = np.hstack([b * np.eye(3) for b in beta])  #          (D, D*Nv)
        dbeta_dalpha = (  #                                          (Nv, Nv)
            np.diag(softmax_alpha) - np.outer(softmax_alpha, softmax_alpha)
        ) / a
        dc_dalpha = self.v @ dbeta_dalpha  #                         (D, Nv)
        self.dc_dphi = np.hstack(  #                                 (D, (D+1)*Nv+1)
            [dc_dv, dc_dalpha, np.zeros((self.D, 1))]
        )

        self.p = (cfg.p_max / 2 - 1) * (np.tanh(self.ep) + 1) + 2
        self.dp_dep = (cfg.p_max / 2 - 1) * (1 - np.tanh(self.ep) ** 2)

    def support(
        self, x: np.ndarray, q: np.ndarray = None, get_s=False, get_dh_dphi=False
    ):
        """
        Args:
            x:           (D) or (D, Nx)
            q:           (SE(D)) or None
            get_s:       (bool) Defaults to False.
            get_dh_dphi: (bool) Defaults to False.

        Returns:
            h:       (Nx) (\sum_{i}{\max(v_i-c)^Tx,0)}^p)^{1/p}
            s:       (D, Nx) or None
            dh_dphi: (Nx, (D+1)*Nx+1) or None
        """
        if q is None:
            if self.D == 3:
                p, R = np.zeros(3), np.eye(3)
        else:
            if self.D == 3:
                p, R = q[:3], Rotation.from_quat(q[3:]).as_matrix()
        if x.ndim == 1:
            x = np.expand_dims(x, -1)
        _, Nx = x.shape

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

        if get_s:
            s = (h_1[:, np.newaxis] * zp_1 @ self.loc_v.T).T  #      (D, Nx)
            s = p[:, np.newaxis] + R @ (s + self.c[:, np.newaxis])
        else:
            s = None

        if get_dh_dphi:
            dh_dz = zp_1 * h_1[:, np.newaxis]  #                     (Nx, Nv)
            dh_dv_ = (  #                                            (Nx, Nv, D)
                dh_dz[..., np.newaxis] * x.T[:, np.newaxis, :]
            )
            dh_dv = np.reshape(dh_dv_, (Nx, -1))  #                  (Nx, D*Nv)

            dh_dc = -np.sum(dh_dv_, axis=1)  #                       (Nx, D)

            log_z = np.log(np.where(z > 0, z, 1))  #                 (Nx, Nv)
            dh_dp = (max_z * h_1 / self.p) * (  #                    (Nx)
                np.sum(zp_0 * log_z, axis=1) - sum_zp * np.log(sum_zp) / self.p
            )
            dh_dep = dh_dp[:, np.newaxis] * self.dp_dep  #           (Nx, 1)

            dh_dphi = (  #                                           (Nx, (D+1)*Nv+1)
                np.hstack([dh_dv, np.zeros((Nx, self.Nv)), dh_dep])
                + dh_dc @ self.dc_dphi
            )
        else:
            dh_dphi = None

        return h, s, dh_dphi

    def render(self, plotter: pv.plotter, resolution=4, **kwargs):
        """
        Args:
            plotter:    (pv.Plotter)
            resolution: (int) Defaults to 4.
            **kwargs:   keyword arguments to be passed to plotter.add_mesh
        """
        directions = pv.Icosphere(nsub=resolution).points.T
        _, support_points, _ = self.support(directions, get_s=True)

        hull = ConvexHull(support_points.T)
        faces = np.column_stack(
            [3 * np.ones(len(hull.simplices), dtype=np.int32), hull.simplices]
        )
        mesh = pv.PolyData(support_points.T, faces)

        plotter.add_mesh(mesh, **kwargs)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config.yml").dsf

    vertex_set = np.array(
        [[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
    ).T
    sharpness = -1
    dsf = DiffSupport(cfg, vertex_set=vertex_set, sharpness=sharpness)

    plotter = pv.Plotter()
    dsf.render(plotter, color="blue")
    plotter.show()
