# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch-differentiable SDF callables mirroring the numpy primitives."""

from __future__ import annotations

from collections.abc import Callable

import torch
from omnibias.pinn.domain._core.sdf import Box, Halfspace, Sphere
from torch import Tensor

DistanceFn = Callable[[Tensor], Tensor]


def sphere_distance(center: tuple[float, ...], radius: float) -> DistanceFn:
    """``||x_spatial - c|| - r``; uses leading spatial columns of coords."""

    def _fn(coords: Tensor) -> Tensor:
        d = len(center)
        x = coords[..., :d]
        c = coords.new_tensor(center)
        return torch.linalg.vector_norm(x - c, dim=-1) - float(radius)

    return _fn


def box_distance(lo: tuple[float, ...], hi: tuple[float, ...]) -> DistanceFn:
    """Axis-aligned box SDF on the leading spatial columns."""

    def _fn(coords: Tensor) -> Tensor:
        d = len(lo)
        x = coords[..., :d]
        lo_t = coords.new_tensor(lo)
        hi_t = coords.new_tensor(hi)
        center = 0.5 * (lo_t + hi_t)
        half = 0.5 * (hi_t - lo_t)
        q = torch.abs(x - center) - half
        outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0.0)
        return outside + inside

    return _fn


def halfspace_distance(
    normal: tuple[float, ...], point: tuple[float, ...]
) -> DistanceFn:
    def _fn(coords: Tensor) -> Tensor:
        d = len(normal)
        x = coords[..., :d]
        n = coords.new_tensor(normal)
        n = n / torch.linalg.vector_norm(n)
        p = coords.new_tensor(point)
        return ((x - p) * n).sum(dim=-1)

    return _fn


def from_primitive(sdf: Sphere | Box | Halfspace) -> DistanceFn:
    """Build a torch distance callable from a numpy SDF primitive."""
    if isinstance(sdf, Sphere):
        return sphere_distance(sdf.center, sdf.radius)
    if isinstance(sdf, Box):
        return box_distance(sdf.lo, sdf.hi)
    if isinstance(sdf, Halfspace):
        return halfspace_distance(sdf.normal, sdf.point)
    raise TypeError(f"unsupported primitive type {type(sdf)!r}")


def normalize_distance(distance_fn: DistanceFn, *, eps: float = 1e-12) -> DistanceFn:
    """Wrap ``distance_fn`` with the Sukumar-Srivastava ADF normalization.

    Uses autograd for ``|grad omega|`` so the result is differentiable.
    """

    def _fn(coords: Tensor) -> Tensor:
        coords_g = coords.detach().requires_grad_(True)
        omega = distance_fn(coords_g)
        # Sum-then-grad gives d(omega_i)/d(x_i) per sample when omega is (n,).
        grad_outputs = torch.ones_like(omega)
        (g,) = torch.autograd.grad(
            omega, coords_g, grad_outputs=grad_outputs, create_graph=True
        )
        # Only spatial columns contribute to the ADF norm when time is present;
        # use the full gradient (time column of a spatial SDF is typically 0).
        g_norm2 = (g * g).sum(dim=-1)
        denom = torch.sqrt(omega * omega + g_norm2 + float(eps))
        return omega / denom

    return _fn


__all__ = [
    "box_distance",
    "from_primitive",
    "halfspace_distance",
    "normalize_distance",
    "sphere_distance",
]
