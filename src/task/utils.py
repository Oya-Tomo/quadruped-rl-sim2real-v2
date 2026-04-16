from __future__ import annotations

import torch


def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    a_norm = a / (a.norm(dim=dim, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=dim, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=dim)


def modified_cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    cos_sim = cosine_similarity(a, b, dim=dim, eps=eps)
    a_norm = a.norm(dim=dim)
    b_norm = b.norm(dim=dim)
    ratio = (2.0 * a_norm * b_norm) / (a_norm.square() + b_norm.square() + eps)
    return cos_sim * ratio


def rotation_matrix_2d(yaw: torch.Tensor) -> torch.Tensor:
    cos = yaw.cos()
    sin = yaw.sin()
    row0 = torch.stack([cos, -sin], dim=-1)
    row1 = torch.stack([sin, cos], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def bezier_3d(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    omt = 1.0 - t
    return (
        omt.pow(3) * p0
        + 3.0 * omt.pow(2) * t * p1
        + 3.0 * omt * t.pow(2) * p2
        + t.pow(3) * p3
    )
