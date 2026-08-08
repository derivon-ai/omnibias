# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CPU smoke: hard BC error on a circle via DistanceConstrainedField.

Decision rule: max |u| on the unit circle must be < 1e-10 (exact by
construction for zero Dirichlet data).
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.domain import Sphere
from omnibias.pinn.domain.torch import DistanceConstrainedField
from omnibias.pinn.torch.fields import OneLayerVectorField


def main() -> None:
    t0 = time.perf_counter()
    cs = CoordinateSpec(("x", "y"), domain=((-1.5, 1.5), (-1.5, 1.5)))
    comps = ComponentSpec(("u",))
    base = OneLayerVectorField(
        coordinate_spec=cs, components=comps, hidden=16, base="tanh"
    )
    field = DistanceConstrainedField(
        base=base,
        sdf=Sphere(center=(0.0, 0.0), radius=1.0),
        normalize=False,
        boundary_value_fn=lambda c: {
            "u": torch.zeros(c.shape[0], dtype=c.dtype, device=c.device)
        },
    )
    theta = torch.linspace(0, 2 * torch.pi, 64, dtype=torch.float64)[:-1]
    coords = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    state = field.evaluate(coords)
    u = state.ops.value(state, "u")
    max_abs = float(u.detach().abs().max())
    payload = provenance(
        schema="geometry_sdf/v1",
        config={"n_boundary": 63, "hidden": 16},
    )
    payload.update(
        {
            "max_abs_on_circle": max_abs,
            "elapsed_seconds": time.perf_counter() - t0,
        }
    )
    assert max_abs < 1e-10
    write_json("geometry_sdf.json", payload)
    print("wrote docs/benchmarks/geometry_sdf.json")


if __name__ == "__main__":
    main()
