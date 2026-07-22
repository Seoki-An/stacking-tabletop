import numpy as np
import torch
from typing import Tuple, Dict


def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    # T: (N,4,4)
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    R_T = np.transpose(R, (0, 2, 1))
    t_new = -np.einsum("nij,nj->ni", R_T, t)
    T_inv = np.zeros_like(T)
    T_inv[:, :3, :3] = R_T
    T_inv[:, :3, 3] = t_new
    T_inv[:, 3, 3] = 1.0
    return T_inv


# --- helper: rpy (roll, pitch, yaw) -> rotation matrix (ZYX: R = Rz(yaw) Ry(pitch) Rx(roll)) ---
def rpy_to_rotmat(rpy: torch.Tensor) -> torch.Tensor:
    """
    rpy: tensor (..., 3) in radians [roll, pitch, yaw]
    returns: tensor(..., 3, 3)
    Using ZYX order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    # Build rotation matrices components
    # Rz * Ry * Rx
    R = torch.zeros(rpy.shape[:-1] + (3, 3), dtype=rpy.dtype, device=rpy.device)

    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr

    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr

    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr

    return R


# --- helper: build homogeneous matrix from rpy+trans ---
def build_transform(rpy: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """
    rpy: (3,) tensor
    trans: (3,) tensor
    returns: (4,4) homogeneous matrix
    """
    R = rpy_to_rotmat(rpy.unsqueeze(0))[0]  # (3,3)
    T = torch.eye(4, dtype=rpy.dtype, device=rpy.device)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T


# --- loss between X (single transform) and batch T_bl (N,4,4) ---
def se3_distance_squared(
    X: torch.Tensor, T_bl_batch: torch.Tensor, w_r: float = 1.0, w_t: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    X: (4,4) tensor
    T_bl_batch: (N,4,4) tensor
    returns: scalar loss (mean over batch)
    loss per sample = w_r * angle^2 + w_t * ||t_rel||^2
    where R_rel = R_i @ R_x^T  (or R_x^T @ R_i) ? we'll measure relative from X to sample:
    Let R_rel = R_i @ R_x^T, t_rel = t_i - R_rel @ t_x  (but simpler: transform difference)
    Simpler stable approach:
      R_rel = R_x^T @ R_i  (rotation that rotates from X to sample)
      angle = acos( clamp((trace(R_rel)-1)/2, -1, 1) )
      t_rel = t_i - R_i @ (R_x^T @ t_x)  (equiv to transform difference in world)
    We'll use rotation relative = R_x^T @ R_i and t_rel = t_i - R_rel^T @ t_x (consistent)
    Implementation below uses: R_rel = R_x^T @ R_i  and t_rel = t_i - R_i @ (R_x^T @ t_x)
    """
    N = T_bl_batch.shape[0]
    R_x = X[:3, :3]  # (3,3)
    t_x = X[:3, 3]  # (3,)

    R_i = T_bl_batch[:, :3, :3]  # (N,3,3)
    t_i = T_bl_batch[:, :3, 3]  # (N,3)

    # R_rel = R_x^T @ R_i  -> shape (N,3,3)
    R_x_T = R_x.t()
    R_rel = torch.matmul(R_x_T.unsqueeze(0).expand(N, -1, -1), R_i)

    # compute rotation angle from R_rel via trace
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    # numerical safe cos_theta = (trace - 1) / 2
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_theta)  # (N,)

    # translation residual: bring t_x into sample frame or compare in world:
    # simplest: compute predicted point origin of X in sample frame: p = R_rel^T @ (t_i - t_x)
    # but simpler and stable: compute t_diff = t_i - (R_i @ (R_x_T @ t_x))
    t_x_in_world = torch.matmul(R_i, torch.matmul(R_x_T, t_x))  # (N,3)
    t_diff = t_i - t_x_in_world
    trans_sq = torch.sum(t_diff * t_diff, dim=1)  # (N,)

    loss_per = w_r * (angle**2) + w_t * trans_sq
    return torch.mean(loss_per), torch.mean(angle), torch.mean(torch.sqrt(trans_sq))


# --- optimization routine ---
def optimize_se3_from_batch(
    T_pl_np: np.ndarray,
    T_pl_init: np.ndarray,
    device: str = "cpu",
    lr: float = 1e-2,
    epochs: int = 2000,
    w_r: float = 1.0,
    w_t: float = 1.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    T_pl_np: (N,4,4) numpy array
    T_pl_init: (4,4) numpy array
    returns: optimized 4x4 numpy matrix, and dict with losses history (last values)
    """
    device = torch.device(device)
    T_pl = torch.tensor(T_pl_np, dtype=torch.float32, device=device)

    # init translation as mean of dataset translations
    # mean_t = torch.mean(T_pl[:, :3, 3], dim=0)
    mean_t = torch.tensor(T_pl_init[:3, -1], dtype=torch.float32, device=device)
    # init rpy as zeros (no rotation). Could try to estimate from mean rotation but keep simple.
    init_rpy = torch.zeros(3, dtype=torch.float32, device=device)

    # make parameters (we optimize raw rpy and trans)
    rpy_param = torch.nn.Parameter(init_rpy.clone())
    t_param = torch.nn.Parameter(mean_t.clone())
    # t_param = mean_t

    optimizer = torch.optim.Adam([rpy_param, t_param], lr=lr)
    # optimizer = torch.optim.Adam([rpy_param], lr=lr)

    loss_history = []
    for e in range(epochs):
        optimizer.zero_grad()
        X = build_transform(rpy_param, t_param)  # (4,4)
        loss, mean_angle, mean_trans = se3_distance_squared(X, T_pl, w_r=w_r, w_t=w_t)
        loss.backward()

        rpy_param.grad.data *= 0.0
        # rpy_param.grad.data[0] *= 0.0
        # rpy_param.grad.data[1] *= 0.0
        # rpy_param.grad.data[2] *= 0.0
        t_param.grad.data[-1] *= 0.0

        optimizer.step()

        loss_history.append(float(loss.detach().cpu().numpy()))
        if verbose and (e % max(1, epochs // 100) == 0 or e < 10):
            print(
                f"[{e}/{epochs}] loss={loss.item():.6f}, mean_angle={mean_angle.item():.6f} rad, mean_trans={mean_trans.item():.6f} m"
            )

    X_opt = build_transform(rpy_param.detach(), t_param.detach()).cpu().numpy()
    info = {
        "loss_history": loss_history,
        "rpy": rpy_param.detach().cpu().numpy(),
        "trans": t_param.detach().cpu().numpy(),
    }
    return X_opt, info
