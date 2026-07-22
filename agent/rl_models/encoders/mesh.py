"""
Low-level GNN that encodes a single mesh (stone low-poly or target wall).

Operates on:
    x:          (V, in_dim)     per-vertex features (e.g. 3D position)
    edge_index: (2, E)          mesh edges (undirected, expressed as both directions)
    batch:      (V,)            graph id of each vertex when several meshes are
                                stacked into one big disconnected graph
Returns:
    graph_emb:  (G, out_dim)    one embedding per mesh
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


def faces_to_edge_index(faces: np.ndarray) -> torch.Tensor:
    """Convert (F, 3) triangle faces to a (2, E) undirected edge_index tensor.

    Each triangle contributes its 3 edges in both directions. Duplicate edges
    from shared triangles are removed.
    """
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.concatenate([e, e[:, ::-1]], axis=0)
    e = np.unique(e, axis=0)
    return torch.tensor(e.T, dtype=torch.long)


def faces_tensor_to_edge_index(faces: torch.Tensor) -> torch.Tensor:
    """Torch-native version of `faces_to_edge_index` that strips -1 padding.

    `faces`: (F_max, 3) long. Rows containing any -1 are treated as padding
    and dropped. Returns a (2, E) long tensor on the same device.
    """
    valid = (faces >= 0).all(dim=-1)
    faces = faces[valid]
    if faces.numel() == 0:
        return faces.new_zeros((2, 0), dtype=torch.long)
    e = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )
    e = torch.cat([e, e.flip(-1)], dim=0)
    e = torch.unique(e, dim=0)
    return e.t().contiguous().long()


class MeshConv(MessagePassing):
    """One message-passing layer over a mesh.

    TODO(skeleton): pick a concrete formulation. Reasonable defaults:
        - EdgeConv-style: m_ij = MLP([x_i, x_j - x_i])
        - Add edge features from relative position
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="mean")
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x_i, x_j - x_i], dim=-1))


class MeshGNN(nn.Module):
    """Stack of MeshConv layers + global pooling -> per-mesh embedding."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        pool: str = "mean",
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [MeshConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:]])
        self.pool = pool

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.norms):
            x = norm(F.relu(layer(x, edge_index)))
        return self._pool(x, batch)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # TODO(skeleton): consider attention pooling instead of mean/max.
        return scatter(x, batch, dim=0, reduce=self.pool)
