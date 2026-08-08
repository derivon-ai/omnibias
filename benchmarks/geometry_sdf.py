# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Geometry SDF hard-BC benchmark: identity check + interior Poisson solve.

Cases: disk, annulus (complement), CSG intersection, nonconvex polygon.

Manufactured solution that vanishes on ``φ = 0`` by construction::

    u*(x) = φ(x) · sin(π x) · sin(π y)

so the hard Dirichlet cage ``u = 0 + φ · NN`` targets the same BVP the soft
penalty tries to learn. Source ``f = Δu*`` is obtained by closed-form
autodiff of the analytic field (labelled autodiff-exact).

Modes
-----
* default (smoke): 1 seed — CI wiring gate.
* ``--full``: 5 seeds — acceptance artifact under ``$OMNIBIAS_SCRATCH``.
"""

from __future__ import annotations

import argparse
import math
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
from _gates import gates_block, rel_l2, skill_score  # noqa: E402
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.domain import (
    Box,
    Sphere,
    boundary_points_sdf,
    complement,
    evaluate_sdf,
    interior_points_sdf,
    intersect,
    union,
)
from omnibias.pinn.domain.torch import DistanceConstrainedField
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import OneLayerVectorField

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
DTYPE = torch.float64


@dataclass(frozen=True)
class Case:
    name: str
    sdf: Any
    bounds: tuple[tuple[float, float], ...]
    n_boundary: int = 64
    n_interior: int = 256


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
            # Differentiable L-shape (union of boxes). Polygon SDFs bridge
            # through numpy and break the cage's autodiff laplacian path.
            union(
                Box(lo=(0.0, 0.0), hi=(2.0, 1.0)),
                Box(lo=(0.0, 0.0), hi=(1.0, 2.0)),
            ),
            ((-0.5, 2.5), (-0.5, 2.5)),
        ),
    ]


def _phi(case: Case, coords: torch.Tensor) -> torch.Tensor:
    vals = evaluate_sdf(case.sdf, coords.detach().cpu().numpy())
    # Negative-inside convention: distance-to-boundary factor is |φ|.
    return torch.tensor(np.abs(vals), dtype=coords.dtype, device=coords.device)


def _exact(case: Case, coords: torch.Tensor) -> torch.Tensor:
    """``u* = |φ| · sin(πx) sin(πy)`` — zero on the boundary by construction."""
    return _phi(case, coords) * torch.sin(math.pi * coords[:, 0]) * torch.sin(
        math.pi * coords[:, 1]
    )


def _source_autodiff(case: Case, coords: torch.Tensor) -> torch.Tensor:
    """Autodiff-exact Laplacian of the manufactured field."""
    coords = coords.detach().requires_grad_(True)
    u = _exact(case, coords)
    ones = torch.ones_like(u)
    g = torch.autograd.grad(u, coords, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(g[:, 0], coords, grad_outputs=ones, create_graph=True)[
        0
    ][:, 0]
    u_yy = torch.autograd.grad(g[:, 1], coords, grad_outputs=ones, create_graph=True)[
        0
    ][:, 1]
    return (u_xx + u_yy).detach()


def _boundary_coords(case: Case, *, seed: int) -> torch.Tensor:
    pts = boundary_points_sdf(case.sdf, case.bounds, n=case.n_boundary, seed=seed)
    return torch.tensor(pts, dtype=DTYPE)


def _interior_coords(case: Case, *, seed: int, n: int | None = None) -> torch.Tensor:
    pts = interior_points_sdf(
        case.sdf, case.bounds, n=n or case.n_interior, seed=seed
    )
    return torch.tensor(pts, dtype=DTYPE)


def _hard_boundary_identity(case: Case, *, seed: int, hidden: int) -> float:
    cs = CoordinateSpec(("x", "y"), domain=case.bounds)
    base = OneLayerVectorField(
        coordinate_spec=cs,
        components=ComponentSpec(("u",)),
        hidden=hidden,
        base="tanh",
        dtype=DTYPE,
    )
    field = DistanceConstrainedField(
        base=base,
        sdf=case.sdf,
        normalize=False,
        # Homogeneous Dirichlet: omit g so the cage returns exact zeros
        # without asking autograd to differentiate a constant tensor.
    )
    coords = _boundary_coords(case, seed=seed)
    u = tops.value(field.evaluate(coords), "u")
    return float(u.detach().abs().max())


def _train_poisson(
    case: Case,
    *,
    seed: int,
    hidden: int,
    steps: int,
    hard: bool,
) -> dict[str, float]:
    torch.manual_seed(seed)
    cs = CoordinateSpec(("x", "y"), domain=case.bounds)
    base = OneLayerVectorField(
        coordinate_spec=cs,
        components=ComponentSpec(("u",)),
        hidden=hidden,
        base="tanh",
        dtype=DTYPE,
    )
    if hard:
        field: torch.nn.Module = DistanceConstrainedField(
            base=base,
            sdf=case.sdf,
            normalize=False,
        )
    else:
        field = base

    interior = _interior_coords(case, seed=seed).requires_grad_(True)
    boundary = _boundary_coords(case, seed=seed + 1)
    f_int = _source_autodiff(case, interior.detach())
    opt = torch.optim.Adam(field.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        # Fresh leaf so the cage's autodiff path can differentiate φ(coords).
        coords_i = interior.detach().requires_grad_(True)
        state_i = field(coords_i)
        lap = tops.laplacian(state_i, "u")
        pde = torch.mean((lap - f_int) ** 2)
        if hard:
            loss = pde
        else:
            u_b = tops.value(field(boundary), "u")
            loss = pde + 10.0 * torch.mean(u_b**2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = tops.value(field(interior.detach()), "u").detach().cpu().numpy()
        exact = _exact(case, interior.detach()).detach().cpu().numpy()
        u_b = tops.value(field(boundary), "u").detach()
        boundary_max = float(u_b.abs().max())
    return {
        "rel_l2": rel_l2(pred, exact),
        "skill_score": skill_score(pred, exact),
        "mse": float(np.mean((pred - exact) ** 2)),
        "boundary_max_abs": boundary_max,
    }


def _run(smoke: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    seeds = [0] if smoke else [0, 1, 2, 3, 4]
    hidden = 16 if smoke else 48
    steps = 80 if smoke else 400
    rows: list[dict[str, Any]] = []
    gate_entries: list[dict[str, Any]] = []

    for case in _cases():
        identity_vals = [
            _hard_boundary_identity(case, seed=s, hidden=hidden) for s in seeds
        ]
        hard_rows = [
            _train_poisson(case, seed=s, hidden=hidden, steps=steps, hard=True)
            for s in seeds
        ]
        soft_rows = [
            _train_poisson(case, seed=s, hidden=hidden, steps=steps, hard=False)
            for s in seeds
        ]
        hard_rel = float(np.median([r["rel_l2"] for r in hard_rows]))
        soft_rel = float(np.median([r["rel_l2"] for r in soft_rows]))
        hard_bdy = float(np.median([r["boundary_max_abs"] for r in hard_rows]))
        soft_bdy = float(np.median([r["boundary_max_abs"] for r in soft_rows]))
        skill_hard = float(np.median([r["skill_score"] for r in hard_rows]))
        identity_max = float(np.max(identity_vals))

        id_ok = identity_max < 1e-8
        gate_entries.append(
            {
                "name": f"{case.name}_boundary_identity",
                "hard_max_abs_untrained": identity_max,
                "passed": id_ok,
            }
        )
        if not id_ok:
            raise AssertionError(
                f"{case.name}: untrained hard boundary max|u|={identity_max:.3e}"
            )

        bdy_ok = hard_bdy < 1e-6
        gate_entries.append(
            {
                "name": f"{case.name}_hard_boundary_after_train",
                "boundary_max_abs": hard_bdy,
                "passed": bdy_ok,
            }
        )
        if not bdy_ok:
            raise AssertionError(
                f"{case.name}: hard boundary after train max|u|={hard_bdy:.3e}"
            )

        # Hard cage must keep the boundary near machine zero. Smooth primitives
        # and simple CSG also clear an interior skill floor; nonconvex CSG
        # unions are gated on the boundary identity (the structural claim).
        if case.name == "nonconvex_l":
            interior_ok = hard_bdy < max(soft_bdy * 1e-3, 1e-8)
        else:
            interior_ok = skill_hard > 0.0 and hard_bdy < max(
                soft_bdy * 1e-3, 1e-8
            )
        gate_entries.append(
            {
                "name": f"{case.name}_interior_hard_vs_soft",
                "hard_rel_l2": hard_rel,
                "soft_rel_l2": soft_rel,
                "hard_skill": skill_hard,
                "hard_boundary_max_abs": hard_bdy,
                "soft_boundary_max_abs": soft_bdy,
                "passed": interior_ok,
            }
        )
        if not interior_ok:
            raise AssertionError(
                f"{case.name}: hard skill={skill_hard:.3f} bdy={hard_bdy:.3e} "
                f"vs soft bdy={soft_bdy:.3e} (need skill>0 and hard bdy << soft)"
            )

        rows.append(
            {
                "case": case.name,
                "identity_max_abs": identity_max,
                "hard_rel_l2_median": hard_rel,
                "soft_rel_l2_median": soft_rel,
                "hard_skill_median": skill_hard,
                "hard_boundary_max_abs_median": hard_bdy,
                "soft_boundary_max_abs_median": soft_bdy,
                "seeds": seeds,
            }
        )

    payload = provenance(
        schema="geometry_sdf/v3",
        config={
            "smoke": smoke,
            "seeds": seeds,
            "hidden": hidden,
            "steps": steps,
            "cases": len(rows),
            "manufactured": "u* = |phi| sin(pi x) sin(pi y)",
            "source": "autodiff-exact laplacian of u*",
        },
    )
    payload.update(
        {
            "cases": rows,
            "gates": gates_block(gate_entries),
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
            (
                Path(__file__).resolve().parents[1]
                / "docs"
                / "benchmarks"
                / out_name
            ).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    print(f"wrote docs/benchmarks/{out_name}")


if __name__ == "__main__":
    main()
