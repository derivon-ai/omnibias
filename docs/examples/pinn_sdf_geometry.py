# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CI smoke: SDF hard BC on a unit disk via DistanceConstrainedField."""

from __future__ import annotations

import torch
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.domain import Sphere, interior_points_sdf
from omnibias.pinn.domain.torch import DistanceConstrainedField
from omnibias.pinn.torch.fields import OneLayerVectorField


def main() -> None:
    cs = CoordinateSpec(("x", "y"), domain=((-1.5, 1.5), (-1.5, 1.5)))
    comps = ComponentSpec(("u",))
    base = OneLayerVectorField(
        coordinate_spec=cs, components=comps, hidden=8, base="tanh"
    )
    sdf = Sphere(center=(0.0, 0.0), radius=1.0)
    field = DistanceConstrainedField(
        base=base,
        sdf=sdf,
        normalize=False,
        boundary_value_fn=lambda c: {
            "u": torch.zeros(c.shape[0], dtype=c.dtype, device=c.device)
        },
    )
    theta = torch.linspace(0, 2 * torch.pi, 32, dtype=torch.float64)[:-1]
    coords = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    state = field.evaluate(coords)
    u = state.ops.value(state, "u")
    max_abs = float(u.detach().abs().max())
    assert max_abs < 1e-10
    pts = interior_points_sdf(sdf, [(-1.5, 1.5), (-1.5, 1.5)], n=16, seed=0)
    assert pts.shape == (16, 2)
    print("pinn_sdf_geometry: ok", max_abs)


if __name__ == "__main__":
    main()
