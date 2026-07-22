"""
High-level GNN over the small heterogeneous scene graph.

Node types:
    - placed stone   (M slots, masked by `scene_mask`)        — world-frame embedding
    - target segment (N slots; for now the wall is a single segment)
    - pending stone  (K slots, masked by `pending_mask`)      — local-frame, shape-only embedding
    - candidate action (A slots; only present in Q-mode)      — world-frame embedding (pose-applied)

All node features come from the low-level MeshGNN. Pose is **not** added as a
separate feature — the choice of frame (world vs local) for the input vertices
is what distinguishes "this stone, here" from "this stone, available".
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SceneGNNLayer(nn.Module):
    """Single attention-style update over scene nodes."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ffn(x))
        return x


class SceneGNN(nn.Module):
    """High-level encoder over (stone, target, pending, action) nodes."""

    TYPE_STONE = 0
    TYPE_TARGET = 1
    TYPE_PENDING = 2
    TYPE_ACTION = 3
    NUM_TYPES = 4

    def __init__(self, dim: int, num_layers: int = 3, n_heads: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(self.NUM_TYPES, dim)
        self.layers = nn.ModuleList(
            [SceneGNNLayer(dim, n_heads) for _ in range(num_layers)]
        )

    def forward(
        self,
        stone_feats: torch.Tensor,             # (B, M, F)
        target_feats: torch.Tensor,            # (B, N, F)
        pending_feats: torch.Tensor,           # (B, K, F)
        action_feats: Optional[torch.Tensor],  # (B, A, F) or None
        scene_mask: torch.Tensor,              # (B, M) bool — placed stones
        pending_mask: torch.Tensor,            # (B, K) bool — still-available stones
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns (stone_out, target_out, pending_out, action_out)."""
        B, M = stone_feats.shape[:2]
        N = target_feats.shape[1]
        K = pending_feats.shape[1]
        A = 0 if action_feats is None else action_feats.shape[1]
        device = stone_feats.device

        nodes = [
            stone_feats + self._type(self.TYPE_STONE, M, device),
            target_feats + self._type(self.TYPE_TARGET, N, device),
            pending_feats + self._type(self.TYPE_PENDING, K, device),
        ]
        if action_feats is not None:
            nodes.append(action_feats + self._type(self.TYPE_ACTION, A, device))
        x = torch.cat(nodes, dim=1)            # (B, M+N+K+A, dim)

        # True positions are *ignored* by attention. Stones beyond scene_mask
        # and pendings beyond pending_mask are pads; target/action are always real.
        pad = torch.zeros(B, M + N + K + A, dtype=torch.bool, device=device)
        pad[:, :M] = ~scene_mask
        pad[:, M + N : M + N + K] = ~pending_mask

        for layer in self.layers:
            x = layer(x, key_padding_mask=pad)

        stone_out = x[:, :M]
        target_out = x[:, M : M + N]
        pending_out = x[:, M + N : M + N + K]
        action_out = x[:, M + N + K :] if action_feats is not None else None
        return stone_out, target_out, pending_out, action_out

    def _type(self, type_id: int, count: int, device: torch.device) -> torch.Tensor:
        ids = torch.full((count,), type_id, dtype=torch.long, device=device)
        return self.type_emb(ids)              # (count, dim)
