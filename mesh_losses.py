"""
Mesh losses for direct deformation training.

Includes:
- Chamfer distance between two point clouds
- Uniform Laplacian smoothness loss
- Edge length regularization

All functions operate on PyTorch tensors on the active device.
"""

from __future__ import annotations

import torch
from typing import Tuple


def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric Chamfer distance between two point clouds.

    Args:
        p1: (N, 3) predicted points
        p2: (M, 3) target points
    Returns:
        scalar tensor
    """
    # torch.cdist uses efficient batched computation on GPU
    dists = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0), p=2)  # (1, N, M)
    dists_p1 = dists.min(dim=2).values.squeeze(0)  # (N,)
    dists_p2 = dists.min(dim=1).values.squeeze(0)  # (M,)
    return dists_p1.mean() + dists_p2.mean()


def build_adjacency_from_faces(faces: torch.Tensor, num_vertices: int) -> torch.Tensor:
    """Build dense adjacency matrix (N, N) from triangular faces.
    Self-loops are included.
    """
    device = faces.device
    adj = torch.zeros((num_vertices, num_vertices), device=device, dtype=torch.float32)
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    adj[i0, i1] = 1; adj[i1, i0] = 1
    adj[i1, i2] = 1; adj[i2, i1] = 1
    adj[i2, i0] = 1; adj[i0, i2] = 1
    adj.fill_diagonal_(1.0)
    return adj


def laplacian_smoothness(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Uniform Laplacian smoothing: sum of squared difference between each vertex
    and the mean of its neighbors.

    Args:
        vertices: (N, 3)
        faces: (F, 3)
    Returns:
        scalar tensor
    """
    N = vertices.shape[0]
    adj = build_adjacency_from_faces(faces, N)
    # Exclude self for neighbor averaging
    adj_no_self = adj.clone()
    adj_no_self.fill_diagonal_(0.0)
    deg = adj_no_self.sum(dim=1, keepdim=True).clamp(min=1.0)
    neigh_mean = (adj_no_self @ vertices) / deg
    return ((vertices - neigh_mean) ** 2).sum(dim=1).mean()


def edge_length_loss(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Penalize long/irregular edges by summing squared edge lengths.
    Normalized by number of edges.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e01 = ((v0 - v1) ** 2).sum(dim=1)
    e12 = ((v1 - v2) ** 2).sum(dim=1)
    e20 = ((v2 - v0) ** 2).sum(dim=1)
    return (e01.mean() + e12.mean() + e20.mean()) / 3.0


__all__ = [
    "chamfer_distance",
    "laplacian_smoothness",
    "edge_length_loss",
]
