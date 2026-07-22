import numpy as np
import pyvista as pv
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .support_function import DiffSupport


def _cfg_get(cfg, key, default):
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _initial_vertex_set(num_vertices: int) -> np.ndarray:
    if num_vertices < 4:
        raise ValueError("DSF fitting needs at least 4 vertices.")

    if num_vertices == 20:
        return pv.Dodecahedron().points.T

    i = np.arange(num_vertices, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / num_vertices
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i

    return np.vstack((radius * np.cos(theta), radius * np.sin(theta), z))


class ShapeCost:
    def __init__(self, cfg, x: np.ndarray, q: np.ndarray, h_gt: np.ndarray):
        """
        Args:
            x:    (3, Nx)
            q:    (SE(3))
            h_gt: (Nx)
        """
        self.cfg = cfg
        self.x = x
        self.q, self.p, self.R = q, q[:3], Rotation.from_quat(q[3:]).as_matrix()
        self.h_gt = h_gt
        self.last_phi = None

    def evaluate(self, phi: np.ndarray):
        """
        Args:
            phi:  (4*Nv+1)

        Returns:
            cost: (Nx)
            grad: (Nx, 4*Nv+1)
        """
        if self.last_phi is not None and np.allclose(phi, self.last_phi):
            return self.last_cost, self.last_grad
        self.last_phi = phi

        dsf = DiffSupport(phi=phi, cfg=self.cfg.dsf)
        h, _, dh_dphi = dsf.support(self.x, self.q, get_dh_dphi=True)

        self.last_cost = h + self.x.T @ (self.p + self.R @ dsf.c) - self.h_gt
        self.last_grad = dh_dphi + self.x.T @ self.R @ dsf.dc_dphi

        return self.last_cost, self.last_grad

    def cost(self, phi: np.ndarray):
        """
        Args:
            phi:  (4*Nv+1)

        Returns:
            cost: (Nx)
        """
        cost, _ = self.evaluate(phi)
        return cost

    def grad(self, phi: np.ndarray):
        """
        Args:
            phi:  (4*Nv+1)

        Returns:
            grad: (Nx, 4*Nv+1)
        """
        _, grad = self.evaluate(phi)
        return grad


def _optimize_shape_once(cfg, v_in: np.ndarray, max_iter: int = None):
    num_vertices = int(_cfg_get(cfg.optimizer, "num_vertices", 20))
    v_0 = _initial_vertex_set(num_vertices)
    phi_0 = DiffSupport(cfg.dsf, v_0).phi

    x = pv.Icosphere(nsub=cfg.optimizer.n_sub_sample).points.T
    q = np.array([0, 0, 0, 0, 0, 0, 1])
    support_quantile = _cfg_get(cfg.optimizer, "support_quantile", 0.98)
    support_quantile = float(np.clip(support_quantile, 0.0, 1.0))
    projections = v_in.T @ x
    if support_quantile >= 1.0:
        h_gt = np.max(projections, axis=0)
    else:
        h_gt = np.quantile(projections, support_quantile, axis=0)

    shape_cost = ShapeCost(cfg, x, q, h_gt)
    loss = _cfg_get(cfg.optimizer, "loss", "soft_l1")
    f_scale = _cfg_get(cfg.optimizer, "f_scale", 0.03)
    result = least_squares(
        fun=shape_cost.cost,
        x0=phi_0,
        jac=shape_cost.grad,
        method="trf",
        max_nfev=cfg.optimizer.max_iter if max_iter is None else max_iter,
        loss=loss,
        f_scale=f_scale,
    )
    return DiffSupport(cfg.dsf, phi=result.x)


def _support_inliers(
    cfg,
    dsf: DiffSupport,
    v_in: np.ndarray,
    threshold: float,
    n_sub_sample: int,
):
    x = pv.Icosphere(nsub=n_sub_sample).points.T
    q = np.array([0, 0, 0, 0, 0, 0, 1])
    h, _, _ = dsf.support(x, q, get_dh_dphi=False)
    h_model = h + x.T @ dsf.c
    violations = np.max(v_in.T @ x - h_model[None, :], axis=1)
    return violations <= threshold, violations


def _ransac_inlier_points(cfg, v_in: np.ndarray):
    ransac_cfg = cfg.optimizer.ransac
    rng = np.random.default_rng(_cfg_get(ransac_cfg, "seed", None))

    n_points = v_in.shape[1]
    if n_points == 0:
        return v_in

    sample_ratio = _cfg_get(ransac_cfg, "sample_ratio", 0.5)
    max_sample_points = _cfg_get(ransac_cfg, "max_sample_points", 2000)
    min_sample_points = _cfg_get(ransac_cfg, "min_sample_points", 200)
    sample_size = int(np.ceil(n_points * sample_ratio))
    sample_size = min(sample_size, max_sample_points, n_points)
    sample_size = max(min(sample_size, n_points), min(min_sample_points, n_points))

    n_iter = _cfg_get(ransac_cfg, "n_iter", 12)
    fit_max_iter = _cfg_get(ransac_cfg, "fit_max_iter", 25)
    threshold = _cfg_get(ransac_cfg, "inlier_threshold", 0.03)
    score_n_sub = _cfg_get(ransac_cfg, "score_n_sub_sample", cfg.optimizer.n_sub_sample)
    min_inlier_ratio = _cfg_get(ransac_cfg, "min_inlier_ratio", 0.5)

    best = None
    for i in range(n_iter):
        sample_idx = rng.choice(n_points, size=sample_size, replace=False)
        try:
            dsf = _optimize_shape_once(cfg, v_in[:, sample_idx], max_iter=fit_max_iter)
            inliers, violations = _support_inliers(
                cfg, dsf, v_in, threshold=threshold, n_sub_sample=score_n_sub
            )
        except Exception as e:
            print(f"[dsf ransac] iteration {i} failed: {e}")
            continue

        inlier_count = int(inliers.sum())
        if inlier_count == 0:
            continue
        score = (
            inlier_count,
            -float(np.median(np.maximum(violations[inliers], 0.0))),
        )
        if best is None or score > best["score"]:
            best = {
                "inliers": inliers,
                "score": score,
                "inlier_count": inlier_count,
            }
        print(
            f"[dsf ransac] iter={i}, inliers={inlier_count}/{n_points} "
            f"({inlier_count / n_points:.1%})"
        )

    if best is None:
        print("[dsf ransac] no valid model found; using all points")
        return v_in

    inlier_ratio = best["inlier_count"] / n_points
    if inlier_ratio < min_inlier_ratio:
        print(
            f"[dsf ransac] best inlier ratio {inlier_ratio:.1%} below "
            f"{min_inlier_ratio:.1%}; using all points"
        )
        return v_in

    print(
        f"[dsf ransac] using {best['inlier_count']}/{n_points} "
        f"inliers ({inlier_ratio:.1%}) for final fit"
    )
    return v_in[:, best["inliers"]]


def optimize_shape(cfg, v_in: np.ndarray):
    """
    Args:
        v_in: (3, N) 입력 vertex set (e.g., vertices of convex hull, PCD)

    Returns:
        dsf: (DiffSupport)
    """
    ransac_cfg = _cfg_get(cfg.optimizer, "ransac", None)
    if ransac_cfg is not None and _cfg_get(ransac_cfg, "enabled", False):
        v_in = _ransac_inlier_points(cfg, v_in)

    return _optimize_shape_once(cfg, v_in)


if __name__ == "__main__":
    import trimesh
    from omegaconf import OmegaConf

    mesh = trimesh.load("assets/cone.obj")
    trimesh.grouping.merge_vertices(mesh, True, True)
    v_in = mesh.vertices.T

    cfg = OmegaConf.load("config.yml")
    dsf = optimize_shape(cfg, v_in)

    plotter = pv.Plotter()
    dsf.render(plotter)
    plotter.show()
