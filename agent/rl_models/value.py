"""
Hierarchical-GNN value (Q) function for stone stacking.

Pipeline (from `agent/env/components/state.py:Observation`):

    stacked_points (B, K, V_max, 3) ──→ MeshGNN(stone)   → stone_emb   (B, K, F)
    target_points  (B, N, V', 3)    ──→ MeshGNN(target)  → target_emb  (B, N, F)
    pending_points (B, K, V_max, 3) ──→ MeshGNN(stone)   → pending_emb (B, K, F)   [local frame]
    action.pose × pending_points[stone_idx] ─→ MeshGNN(stone) → action_emb (B, A, F)

    sharpness (B, K) — concatenated to each stone-side embedding as a scalar
    feature before the SceneGNN projection.

                                ↓
                            SceneGNN
                (masked attention over stone, target, pending, action nodes;
                 type embeddings only — pose is implicit in vertex coordinates)
                                ↓
        ┌──────────┬───────────────┬───────────────┐
     value head      advantage head     terminal head
        ↓                ↓                ↓
       V(s)             A(s,a)          term_logit

Design notes
------------
- Vertex configuration carries everything: world-frame vertices encode shape +
  pose jointly. We don't add a separate pose channel.
- The same `stone_encoder` is reused for placed stones, pending stones, and
  action stones. The frame of the input vertices (world vs local) is what
  changes the semantics:
      * placed stones  → world frame (vertices already transformed in env)
      * pending stones → local frame (catalog as-is, pose-invariant)
      * action stones  → world frame (catalog × proposed pose)
- Catalog vertices, faces (topology) and per-stone sharpness all arrive via
  `Observation`. Vertex padding uses inf, face padding uses -1; both are
  stripped at the encoder boundary.
- `stacked_points` is indexed by inventory slot (not placement order), so
  `pending_mask = ~scene_mask` and slot k always refers to the same stone
  identity across `pending_points`, `pending_faces`, and `sharpness`.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from omegaconf import OmegaConf

from agent.env.components.state import Observation
from utils import construct_mlp, quaternion_to_matrix

from .encoders import MeshGNN, SceneGNN, faces_tensor_to_edge_index


VERTEX_IN_DIM = 3
SHARPNESS_DIM = 1   # one scalar per stone, appended after MeshGNN aggregation
OBSERVATION_INPUT_KEYS = {
    "pending_points",
    "pending_faces",
    "stacked_points",
    "target_points",
    "target_faces",
    "scene_mask",
    "sharpness",
}


class StackingQfunction(nn.Module):
    """Top-level Q-function. V is recovered via aggregation over stones + target."""

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cpu")

        f_mesh = cfg.mesh_encoder.out_dim
        f_scene = cfg.scene_encoder.dim

        self.stone_encoder = MeshGNN(
            in_dim=VERTEX_IN_DIM,
            hidden_dim=cfg.mesh_encoder.hidden_dim,
            out_dim=f_mesh,
            num_layers=cfg.mesh_encoder.num_layers,
        )
        self.target_encoder = MeshGNN(
            in_dim=VERTEX_IN_DIM,
            hidden_dim=cfg.mesh_encoder.hidden_dim,
            out_dim=f_mesh,
            num_layers=cfg.mesh_encoder.num_layers,
        )

        # Stone-side projection takes mesh feature + sharpness scalar.
        self.stone_to_scene = nn.Linear(f_mesh + SHARPNESS_DIM, f_scene)
        self.target_to_scene = nn.Linear(f_mesh, f_scene)

        self.scene_encoder = SceneGNN(
            dim=f_scene,
            num_layers=cfg.scene_encoder.num_layers,
            n_heads=cfg.scene_encoder.n_heads,
        )

        self.value_head = construct_mlp(
            [f_scene] + cfg.heads.value_layer_dims, nn.Linear, act=nn.ReLU()
        )
        self.advantage_head = construct_mlp(
            [f_scene] + cfg.heads.advantage_layer_dims, nn.Linear, act=nn.ReLU()
        )
        self.terminal_head = construct_mlp(
            [f_scene] + cfg.heads.terminal_layer_dims + [1],
            nn.Linear,
            act=nn.ReLU(),
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_dict: Dict[str, Dict[str, Union[torch.Tensor, np.ndarray]]],
        separate_outputs: bool = False,
    ):
        """
        input_dict["obs"]    : `Observation` fields as tensors
        input_dict["action"] : optional dict with "pose" (B, A, 7), "stone_idx" (B, A)
        """
        obs, act, batchwise, multi_action = self._prepare(input_dict)
        scene_mask = obs["scene_mask"].bool()
        pending_mask = ~scene_mask
        sharpness = obs["sharpness"]                          # (B, K)
        pending_points = obs["pending_points"]                # (B, K, V_max, 3)
        pending_faces = obs["pending_faces"].long()           # (B, K, F_max, 3)
        target_faces = obs["target_faces"].long()             # (B, F_t, 3) or (B, N, F_t, 3)

        # 1. low-level encoding ------------------------------------------------
        stone_emb = self._encode_stone_batch(
            obs["stacked_points"], pending_faces, scene_mask
        )
        target_emb = self._encode_target_batch(obs["target_points"], target_faces)
        pending_emb = self._encode_pending_batch(pending_points, pending_faces)

        action_emb = None
        action_sharpness = None
        if act is not None:
            action_emb = self._encode_actions(act, pending_points, pending_faces)
            action_sharpness = self._gather_action_sharpness(sharpness, act["stone_idx"])

        # 2. attach sharpness, project to scene dim ----------------------------
        stone_node = self.stone_to_scene(self._with_sharpness(stone_emb, sharpness))
        pending_node = self.stone_to_scene(self._with_sharpness(pending_emb, sharpness))
        target_node = self.target_to_scene(target_emb)
        action_node = (
            self.stone_to_scene(self._with_sharpness(action_emb, action_sharpness))
            if action_emb is not None
            else None
        )

        # 3. high-level encoding ----------------------------------------------
        stone_out, target_out, pending_out, action_out = self.scene_encoder(
            stone_feats=stone_node,
            target_feats=target_node,
            pending_feats=pending_node,
            action_feats=action_node,
            scene_mask=scene_mask,
            pending_mask=pending_mask,
        )

        # 4. heads -------------------------------------------------------------
        # V reads from placed stones + target. Pending nodes only inform other
        # nodes via attention; we don't aggregate their values directly, since
        # their "value" is best expressed through their effect on placement.
        scene_mask_f = scene_mask.float()
        stone_values = self.value_head(stone_out) * scene_mask_f[..., None]
        target_values = self.value_head(target_out).sum(dim=1)
        values = target_values + stone_values.sum(dim=1)

        if action_out is None:
            return self._squeeze(values, batchwise=batchwise)

        advantages = self.advantage_head(action_out)
        terminal_logits = self.terminal_head(action_out)

        if separate_outputs:
            return self._squeeze_outputs(
                values, advantages, terminal_logits, batchwise, multi_action
            )

        q_values = values[:, None, :] + advantages
        return self._squeeze(q_values, batchwise=batchwise, multi_action=multi_action)

    # ------------------------------------------------------------------ #
    # action sampling
    # ------------------------------------------------------------------ #

    def greedy_sample(
        self,
        obs: Dict[str, Union[np.ndarray, torch.Tensor]],
        action_samples: Dict[str, Union[np.ndarray, torch.Tensor, int]],
        action_mask: Union[np.ndarray, torch.Tensor],
        only_sampling: bool = False,
        mode: str = "greedy",
        tau: float = 1.0,
    ):
        """TODO(skeleton): port from previous implementation once forward stabilizes."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # device
    # ------------------------------------------------------------------ #

    def set_device(self, device: str):
        self.device = torch.device(device)
        self.to(self.device)

    # ================================================================== #
    # internals
    # ================================================================== #

    def _prepare(self, input_dict):
        """Move arrays to tensors / inject batch dim, mirroring the legacy API."""
        batchwise = input_dict["obs"]["stacked_points"].ndim > 3
        has_action = "action" in input_dict
        multi_action = False
        if has_action:
            pose = input_dict["action"]["pose"]
            multi_action = (pose.ndim == 2 and not batchwise) or (
                pose.ndim == 3 and batchwise
            )

        for parent, child in input_dict.items():
            for key, value in child.items():
                if parent == "obs" and key not in OBSERVATION_INPUT_KEYS:
                    continue
                if not torch.is_tensor(value):
                    value = torch.tensor(value, device=self.device)
                if value.dtype == torch.float64:
                    value = value.float()
                if not batchwise:
                    value = value.unsqueeze(0)
                if not multi_action and parent == "action":
                    value = value.unsqueeze(1)
                child[key] = value

        return (
            input_dict["obs"],
            input_dict.get("action"),
            batchwise,
            multi_action,
        )

    # ------------------- mesh encoding ---------------------------------- #
    #
    # All three stone-side encoders share `self.stone_encoder`. They differ
    # only in how the (vertices, edges) pairs are assembled into a PyG Batch:
    #   - placed   → take stacked_points[b, k] for slots where scene_mask[b,k]
    #   - pending  → take pending_points[b, k] for slots where pending_mask[b,k]
    #                (these are local-frame catalog vertices)
    #   - action   → apply pose to pending_points[b, stone_idx[b,a]]
    #
    # Edges are built from `pending_faces[b, k]` per call. Vertex pad = inf,
    # face pad = -1; both stripped before constructing each Data.
    #
    # The Python loop is tolerable while we stabilize the design; vectorize
    # later by pre-flattening offsets if it becomes a bottleneck.

    def _encode_stone_batch(
        self,
        points: torch.Tensor,
        faces: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> torch.Tensor:
        """points: (B, K, V_max, 3); faces: (B, K, F_max, 3); scene_mask: (B, K)."""
        B, K = scene_mask.shape
        F_dim = self.cfg.mesh_encoder.out_dim
        out = points.new_zeros(B, K, F_dim)

        data_list: List[Data] = []
        slot_indices: List[Tuple[int, int]] = []
        for b in range(B):
            for k in range(K):
                if not bool(scene_mask[b, k]):
                    continue
                v = self._strip_vertex_pad(points[b, k])
                e = faces_tensor_to_edge_index(faces[b, k])
                data_list.append(Data(x=v, edge_index=e))
                slot_indices.append((b, k))

        if not data_list:
            return out
        batch = Batch.from_data_list(data_list)
        emb = self.stone_encoder(batch.x, batch.edge_index, batch.batch)
        for n, (b, k) in enumerate(slot_indices):
            out[b, k] = emb[n]
        return out

    def _encode_pending_batch(
        self, pending_points: torch.Tensor, pending_faces: torch.Tensor
    ) -> torch.Tensor:
        """pending_points: (B, K, V_max, 3); pending_faces: (B, K, F_max, 3)."""
        B, K = pending_points.shape[:2]
        F_dim = self.cfg.mesh_encoder.out_dim
        if K == 0:
            return pending_points.new_zeros(B, K, F_dim)

        data_list: List[Data] = []
        slot_indices: List[Tuple[int, int]] = []
        for b in range(B):
            for k in range(K):
                v = self._strip_vertex_pad(pending_points[b, k])
                if v.numel() == 0:
                    continue
                e = faces_tensor_to_edge_index(pending_faces[b, k])
                data_list.append(Data(x=v, edge_index=e))
                slot_indices.append((b, k))

        out = pending_points.new_zeros(B, K, F_dim)
        if not data_list:
            return out
        batch = Batch.from_data_list(data_list)
        emb = self.stone_encoder(batch.x, batch.edge_index, batch.batch)
        for n, (b, k) in enumerate(slot_indices):
            out[b, k] = emb[n]
        return out

    def _encode_target_batch(
        self, points: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        """points: (B, V', 3) or (B, N, V', 3); faces: (B, F_t, 3) or (B, N, F_t, 3).

        The target wall currently has a single segment, so `Observation`
        stores `target_points` with no segment axis. We add it here so the
        downstream code can stay (B, N, …)-shaped.
        """
        if points.ndim == 3:
            points = points.unsqueeze(1)               # (B, 1, V', 3)
        if faces.ndim == 3:
            faces = faces.unsqueeze(1)                 # (B, 1, F_t, 3)
        N = points.shape[1]
        if faces.shape[1] != N:
            faces = faces.expand(-1, N, -1, -1)

        B = points.shape[0]
        F_dim = self.cfg.mesh_encoder.out_dim

        data_list: List[Data] = []
        for b in range(B):
            for n in range(N):
                v = self._strip_vertex_pad(points[b, n])
                e = faces_tensor_to_edge_index(faces[b, n])
                data_list.append(Data(x=v, edge_index=e))

        batch = Batch.from_data_list(data_list)
        emb = self.target_encoder(batch.x, batch.edge_index, batch.batch)
        return emb.view(B, N, F_dim)

    def _encode_actions(
        self,
        act: Dict[str, torch.Tensor],
        pending_points: torch.Tensor,
        pending_faces: torch.Tensor,
    ) -> torch.Tensor:
        """Apply each candidate pose to the corresponding catalog stone."""
        pose = act["pose"]                              # (B, A, 7)
        stone_idx = act["stone_idx"].long()             # (B, A)
        B, A = stone_idx.shape
        F_dim = self.cfg.mesh_encoder.out_dim

        out = pose.new_zeros(B, A, F_dim)

        data_list: List[Data] = []
        slot_indices: List[Tuple[int, int]] = []
        for b in range(B):
            for a in range(A):
                k = int(stone_idx[b, a])
                v_local = self._strip_vertex_pad(pending_points[b, k])
                v_world = _apply_pose(pose[b, a], v_local)
                e = faces_tensor_to_edge_index(pending_faces[b, k])
                data_list.append(Data(x=v_world, edge_index=e))
                slot_indices.append((b, a))

        if not data_list:
            return out
        batch = Batch.from_data_list(data_list)
        emb = self.stone_encoder(batch.x, batch.edge_index, batch.batch)
        for n, (b, a) in enumerate(slot_indices):
            out[b, a] = emb[n]
        return out

    # ------------------- helpers ---------------------------------------- #

    @staticmethod
    def _strip_vertex_pad(v: torch.Tensor) -> torch.Tensor:
        """Drop inf-padded rows from a (V_max, 3) vertex tensor."""
        valid = torch.isfinite(v).all(dim=-1)
        return v[valid]

    @staticmethod
    def _with_sharpness(emb: torch.Tensor, sharpness: torch.Tensor) -> torch.Tensor:
        """Append per-stone sharpness as a scalar feature.

        emb:        (B, K, F)
        sharpness:  (B, K)
        returns:    (B, K, F + 1)
        """
        return torch.cat([emb, sharpness.unsqueeze(-1)], dim=-1)

    @staticmethod
    def _gather_action_sharpness(
        sharpness: torch.Tensor, stone_idx: torch.Tensor
    ) -> torch.Tensor:
        """sharpness: (B, K), stone_idx: (B, A) -> (B, A)"""
        return torch.gather(sharpness, dim=1, index=stone_idx.long())

    # ------------------- output shape bookkeeping ---------------------- #

    @staticmethod
    def _squeeze(
        x: torch.Tensor,
        batchwise: bool = True,
        multi_action: bool = True,
    ) -> torch.Tensor:
        if not multi_action and x.ndim >= 2:
            x = x.squeeze(1)
        if not batchwise:
            x = x.squeeze(0)
        return x

    def _squeeze_outputs(
        self,
        values: torch.Tensor,
        advantages: torch.Tensor,
        terminal_logits: torch.Tensor,
        batchwise: bool,
        multi_action: bool,
    ):
        if not multi_action:
            advantages = advantages.squeeze(1)
            terminal_logits = terminal_logits.squeeze(1)
        if not batchwise:
            values = values.squeeze(0)
            advantages = advantages.squeeze(0)
            terminal_logits = terminal_logits.squeeze(0)
        return values, advantages, terminal_logits


def _apply_pose(pose: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """pose: (..., 7), points: (..., V, 3) -> world-frame vertices."""
    trans = pose[..., :3]
    R = quaternion_to_matrix(pose[..., 3:])
    return torch.einsum("...ij,...vj->...vi", R, points) + trans.unsqueeze(-2)
