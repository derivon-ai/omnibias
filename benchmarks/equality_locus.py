# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated primitive: equality locus (theory 01-09 G1–G5; G6 in fields tests).

The locus is a constraint manifold, not a PDE solution. Founding collapse
only (``delta -> 0``). Cost gates are not in CI ``all_passed``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    from omnibias.core.locus import EqualitySystem, UnitTerm, affine_locus, residual

    sys = EqualitySystem(
        (UnitTerm(1, 1.0, (1.0, 0.0)), UnitTerm(1, 1.0, (0.0, 1.0))),
    )
    planes = affine_locus(sys)
    worst = 0.0
    for x in (-0.5, 0.0, 0.7):
        for pt in ((x, x), (x, -x)):
            worst = max(worst, abs(residual(sys, pt)[0]))
    return {
        "name": "g1_affine_lemma",
        "passed": planes is not None and len(planes) == 2 and worst <= 1e-14,
        "worst_abs": worst,
        "n_planes": 0 if planes is None else len(planes),
        "in_ci_all_passed": True,
    }


def _run_g2() -> dict[str, Any]:
    from omnibias.core.locus import EqualitySystem, UnitTerm, jacobian, residual

    rng = random.Random(0)
    worst = 0.0

    def fd4(sys: Any, x: tuple[float, float], axis: int, h: float = 1e-4) -> float:
        def shift(s: float) -> float:
            pt = (x[0] + s, x[1]) if axis == 0 else (x[0], x[1] + s)
            return residual(sys, pt)[0]

        return (-shift(2 * h) + 8 * shift(h) - 8 * shift(-h) + shift(-2 * h)) / (12 * h)

    for _ in range(20):
        sys = EqualitySystem(
            (
                UnitTerm(rng.randint(0, 4), 1.0, (0.8, 0.2)),
                UnitTerm(rng.randint(0, 4), -1.2, (0.1, 0.9)),
            )
        )
        x = (0.1, -0.2)
        df = jacobian(sys, x)
        fd0 = fd4(sys, x, 0)
        scale = max(abs(df[0][0]), abs(fd0), 1e-12)
        worst = max(worst, abs(df[0][0] - fd0) / scale)
    return {
        "name": "g2_jacobian_fd",
        "passed": worst <= 1e-10,
        "worst_rel": worst,
        "in_ci_all_passed": True,
    }


def _run_g3() -> dict[str, Any]:
    from omnibias.core.locus import EqualitySystem, UnitTerm, newton_project

    sys = EqualitySystem(
        (UnitTerm(1, 1.0, (1.0, 0.0)), UnitTerm(2, -2.0, (0.0, 1.0))),
    )
    result = newton_project(sys, (0.0, 0.20), max_iter=5, tol=1e-12)
    return {
        "name": "g3_newton",
        "passed": result.converged and result.iterations <= 5,
        "residual_norm": result.residual_norm,
        "iterations": result.iterations,
        "in_ci_all_passed": True,
    }


def _run_g4() -> dict[str, Any]:
    from omnibias.core.locus import EqualitySystem, UnitTerm, certify_locus_point, newton_project

    sys = EqualitySystem(
        (UnitTerm(1, 1.0, (0.0,)), UnitTerm(2, -2.0, (1.0,))),
    )
    y = math.atanh(0.26959443640544456)
    cert = certify_locus_point(sys, ((y - 0.02, y + 0.02),))
    landed = newton_project(sys, (y - 0.01,), max_iter=8, tol=1e-14)
    contained = cert is not None and cert.enclosure[0][0] <= landed.point[0] <= cert.enclosure[0][1]
    empty = certify_locus_point(sys, ((2.0, 2.2),))
    return {
        "name": "g4_krawczyk",
        "passed": bool(contained) and empty is None,
        "certified": cert is not None,
        "in_ci_all_passed": True,
    }


def _run_g5() -> dict[str, Any]:
    import torch
    from omnibias.core.locus import EqualitySystem, UnitTerm
    from omnibias.fields.locus.torch import newton_project, newton_project_unrolled

    sys = EqualitySystem(
        (UnitTerm(1, 1.0, (1.0, 0.0)), UnitTerm(2, -2.0, (0.0, 1.0))),
    )
    x0 = torch.tensor([0.0, 0.20], dtype=torch.float64)
    w = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    newton_project(sys, x0, weights=w, max_iter=8, tol=1e-14)[1].backward()
    g_ift = w.grad.detach().clone()
    w2 = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    newton_project_unrolled(sys, x0, weights=w2, max_iter=8, tol=1e-14)[1].backward()
    g_un = w2.grad.detach().clone()
    rel = float((g_ift - g_un).norm() / g_un.norm().clamp_min(1e-30))
    return {
        "name": "g5_ift",
        "passed": rel <= 1e-8,
        "rel": rel,
        "in_ci_all_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.equality_locus.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "is_pde_solver": False,
        "collapse": "delta -> 0 (founding); no temperature collapse",
        "certificate": "sound Krawczyk enclosure, not theorem_prover_verified",
    }
    if args.full:
        out_dir = SCRATCH / "locus"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "equality_locus.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        path = write_json("equality_locus_smoke.json", payload)
        print(f"wrote {path}")
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
