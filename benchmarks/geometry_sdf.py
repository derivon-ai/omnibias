# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Geometry SDF hard-BC benchmark vs soft-penalty baseline.

Cases: disk, annulus (complement), CSG intersection, nonconvex polygon.
Decision rule (smoke): hard Dirichlet ``max |u|`` on the boundary < 1e-8.

Modes
-----
* ``--smoke`` (default): 1 seed, tiny nets — CI wiring gate.
* ``--full``: multiple seeds, larger sample — acceptance artifact under
  ``$OMNIBIAS_SCRATCH``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.domain import (
    Box,
    Polygon,
    Sphere,
    complement,
    intersect,
)
from omnibias.pinn.domain.torch import DistanceConstrainedField
from omnibias.pinn.torch.fields import OneLayerVectorField

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


@dataclass(frozen=True)
class Case:
    name: str
    sdf: Any
    bounds: tuple[tuple[float, float], ...]
    n_boundary: int = 48


def _cases() -> list[Case]:
    return [
        Case("disk", Sphere(center=(0.0, 0.0), radius=1.0), ((-1.5, 1.5), (-1.5, 1.5))),
        Case(
            "annulus",
            intersect(
                Sphere(center=(0.0, 0.0), radius=1.0),
                complement(Sphere(center=(0.0, 0.0), radius=0.5)),
            ),
            ((-1.5, 1.5), (-1.5, 1.5)),
        ),
        Case(
            "csg_box_cap",
            intersect(
                Sphere(center=(0.0, 0.0), radius=1.0),
                Box(lo=(-1.0, -1.0), hi=(1.0, 0.5)),
            ),
            ((-1.5, 1.5), (-1.5, 1.5)),
        ),
        Case(
            "nonconvex_l",
            Polygon(
                vertices=(
                    (0.0, 0.0),
                    (2.0, 0.0),
                    (2.0, 1.0),
                    (1.0, 1.0),
                    (1.0, 2.0),
                    (0.0, 2.0),
                )
            ),
            ((-0.5, 2.5), (-0.5, 2.5)),
        ),
    ]


def _boundary_coords(case: Case, *, seed: int) -> torch.Tensor:
    from omnibias.pinn.domain import boundary_points_sdf

    pts = boundary_points_sdf(case.sdf, case.bounds, n=case.n_boundary, seed=seed)
    return torch.tensor(pts, dtype=torch.float64)


def _hard_max_abs(case: Case, *, seed: int, hidden: int) -> float:
    cs = CoordinateSpec(("x", "y"), domain=case.bounds)
    base = OneLayerVectorField(
        coordinate_spec=cs, components=ComponentSpec(("u",)), hidden=hidden, base="tanh"
    )
    field = DistanceConstrainedField(
        base=base,
        sdf=case.sdf,
        normalize=False,
        boundary_value_fn=lambda c: {
            "u": torch.zeros(c.shape[0], dtype=c.dtype, device=c.device)
        },
    )
    coords = _boundary_coords(case, seed=seed)
    state = field.evaluate(coords)
    u = state.ops.value(state, "u")
    return float(u.detach().abs().max())


def _soft_penalty(case: Case, *, seed: int, hidden: int, steps: int = 80) -> float:
    """Untrained soft Dirichlet penalty at boundary (baseline, not a solver)."""
    torch.manual_seed(seed)
    cs = CoordinateSpec(("x", "y"), domain=case.bounds)
    base = OneLayerVectorField(
        coordinate_spec=cs, components=ComponentSpec(("u",)), hidden=hidden, base="tanh"
    )
    coords = _boundary_coords(case, seed=seed)
    opt = torch.optim.Adam(base.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        u = base.evaluate(coords).ops.value(base.evaluate(coords), "u")
        loss = (u * u).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        u = base.evaluate(coords).ops.value(base.evaluate(coords), "u")
    return float(u.abs().max())


def _run(smoke: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    hidden = 12 if smoke else 24
    rows: list[dict[str, Any]] = []
    for case in _cases():
        hard_vals = [_hard_max_abs(case, seed=s, hidden=hidden) for s in seeds]
        soft_vals = [_soft_penalty(case, seed=s, hidden=hidden) for s in seeds]
        rows.append(
            {
                "case": case.name,
                "hard_max_abs_median": float(np.median(hard_vals)),
                "hard_max_abs_max": float(np.max(hard_vals)),
                "soft_max_abs_median": float(np.median(soft_vals)),
                "seeds": seeds,
            }
        )
        if smoke:
            assert float(np.max(hard_vals)) < 1e-8
    payload = provenance(
        schema="geometry_sdf/v2",
        config={"smoke": smoke, "seeds": seeds, "hidden": hidden, "cases": len(rows)},
    )
    payload.update(
        {
            "cases": rows,
            "elapsed_seconds": time.perf_counter() - t0,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="multi-seed acceptance run")
    args = parser.parse_args()
    smoke = not args.full
    payload = _run(smoke=smoke)
    out_name = "geometry_sdf_smoke.json" if smoke else "geometry_sdf.json"
    write_json(out_name, payload)
    if not smoke:
        scratch_path = SCRATCH / "benchmarks" / out_name
        scratch_path.parent.mkdir(parents=True, exist_ok=True)
        scratch_path.write_text(
            (Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / out_name).read_text()
        )
    print(f"wrote docs/benchmarks/{out_name}")


if __name__ == "__main__":
    main()
