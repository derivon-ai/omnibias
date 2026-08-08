# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Neumann / junction failure tests for DistanceConstrainedField."""

from __future__ import annotations

import pytest
import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.domain import Box, NonSmoothBoundaryError, Sphere, intersect
from omnibias.pinn.domain.torch import DistanceConstrainedField
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField


def _field(sdf, *, bc_mode="dirichlet"):
    cs = CoordinateSpec(("x", "y"), domain=((-1.5, 1.5), (-1.5, 1.5)))
    base = OneLayerVectorField(
        coordinate_spec=cs, components=ComponentSpec(("u",)), hidden=4, base="tanh"
    )
    return DistanceConstrainedField(
        base=base, sdf=sdf, normalize=False, bc_mode=bc_mode
    )


def test_neumann_sphere_homogeneous_normal_derivative_zero_on_boundary():
    field = _field(Sphere(center=(0.0, 0.0), radius=1.0), bc_mode="neumann")
    theta = torch.linspace(0, 2 * torch.pi, 24, dtype=torch.float64)[:-1]
    coords = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    coords = coords.detach().requires_grad_(True)
    u = field.evaluate(coords).ops.value(field.evaluate(coords), "u")
    grad = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
    radial = (grad * coords).sum(dim=-1) / torch.linalg.vector_norm(coords, dim=-1)
    assert torch.max(torch.abs(radial)) < 1e-8


def test_neumann_on_rcompose_rejected_at_build():
    sdf = intersect(Sphere(center=(0.0, 0.0), radius=1.0), Box(lo=(-0.5, -0.5), hi=(0.5, 0.5)))
    with pytest.raises(NonSmoothBoundaryError):
        _field(sdf, bc_mode="neumann")
