# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch DistanceConstrainedField: exact Dirichlet on a circle."""

from __future__ import annotations

import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.domain import Sphere
from omnibias.pinn.domain.torch import DistanceConstrainedField
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField


def test_distance_constrained_zero_on_circle():
    cs = CoordinateSpec(("x", "y"), domain=((-1.5, 1.5), (-1.5, 1.5)))
    comps = ComponentSpec(("u",))
    base = OneLayerVectorField(
        coordinate_spec=cs,
        components=comps,
        hidden=8,
        base="tanh",
    )
    sdf = Sphere(center=(0.0, 0.0), radius=1.0)
    field = DistanceConstrainedField(
        base=base,
        sdf=sdf,
        normalize=False,  # raw SDF already zero on the circle
        boundary_value_fn=lambda c: {
            "u": torch.zeros(c.shape[0], dtype=c.dtype, device=c.device)
        },
    )
    # Sample angles on the unit circle.
    theta = torch.linspace(0, 2 * torch.pi, 32, dtype=torch.float64)[:-1]
    coords = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    state = field.evaluate(coords)
    u = state.ops.value(state, "u")
    assert torch.max(torch.abs(u)) < 1e-10


def test_phi_negative_inside():
    cs = CoordinateSpec(("x", "y"), domain=((-1.5, 1.5), (-1.5, 1.5)))
    comps = ComponentSpec(("u",))
    base = OneLayerVectorField(
        coordinate_spec=cs, components=comps, hidden=4, base="tanh"
    )
    field = DistanceConstrainedField(
        base=base, sdf=Sphere(center=(0.0, 0.0), radius=1.0), normalize=False
    )
    inside = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    assert float(field.phi(inside)[0]) < 0.0
