# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: Lie point-symmetry discovery (theory 03-11).

Smoke earns G1 (float-proposed and exact-finite-matrix accepted
in-ansatz dimensions on eight classical equations; heat generators in
the nullspace), G2 (singular-value separation ``> 1e6``), G3 (finite-difference
prolongation gets the heat rank wrong), G4 (dimension stable
across two decades of threshold), G5 (Noether current of the
wave translations conserved to ``1e-10``), and G6 (negative
control returns the trivial algebra). Point symmetries in the
declared ansatz only. Jets are founding bias collapse, not
temperature collapse.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.symbolic.symmetry import (
    DISCLAIMER,
    Generator,
    LinearPoly,
    affine_basis,
    designed_samples,
    determining_matrix,
    discover_symmetries,
    heat_known_coeffs,
    honesty_payload,
    negative_control,
    noether_wave_residual,
    pr_for,
    pr_heat,
    pr_heat_fd,
    suite,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    basis = affine_basis()
    samples = designed_samples(32)
    rows = []
    ok = True
    for pde in suite():
        res = discover_symmetries(pr_for(pde.name), pde.restrict, basis, samples)
        exact = res.exact
        match = res.algebra_dim == pde.expected_dim and res.verified
        ok = ok and match
        rows.append(
            {
                "name": pde.name,
                "float_svd": {
                    "dim": res.algebra_dim,
                    "expected": pde.expected_dim,
                    "threshold": res.rank_threshold,
                    "separation": res.separation,
                    "residual_checked": res.float_verified,
                },
                "exact_rank": exact.to_payload() if exact is not None else None,
                "exact_accepts_float_dimension": res.verified,
            }
        )
    heat = next(p for p in suite() if p.name == "heat")
    mat = determining_matrix(pr_heat, heat.restrict, basis, samples)
    scale = max(1.0, float(np.linalg.norm(mat)))
    heat_ok = all(float(np.linalg.norm(mat @ vec)) <= 1e-12 * scale for vec in heat_known_coeffs())
    return {"name": "g1_recovery", "passed": ok and heat_ok, "equations": rows}


def _run_g2() -> dict[str, Any]:
    basis = affine_basis()
    samples = designed_samples(32)
    seps = {}
    ok = True
    for pde in suite():
        res = discover_symmetries(pr_for(pde.name), pde.restrict, basis, samples)
        seps[pde.name] = res.separation
        ok = ok and res.separation > 1e6
    return {"name": "g2_separation", "passed": ok, "separations": seps}


def _run_g3() -> dict[str, Any]:
    basis = affine_basis()
    samples = designed_samples(32)
    heat = next(p for p in suite() if p.name == "heat")
    exact = discover_symmetries(pr_heat, heat.restrict, basis, samples)
    fd = discover_symmetries(pr_heat_fd, heat.restrict, basis, samples)
    ok = exact.algebra_dim == 5 and fd.algebra_dim != exact.algebra_dim
    return {
        "name": "g3_fd_rank",
        "passed": ok,
        "exact_dim": exact.algebra_dim,
        "fd_dim": fd.algebra_dim,
    }


def _run_g4() -> dict[str, Any]:
    basis = affine_basis()
    samples = designed_samples(32)
    heat = next(p for p in suite() if p.name == "heat")
    dims = {
        str(thr): discover_symmetries(pr_heat, heat.restrict, basis, samples, threshold=thr).algebra_dim
        for thr in (1e-12, 1e-10, 1e-8)
    }
    ok = set(dims.values()) == {5}
    return {"name": "g4_threshold", "passed": ok, "dims": dims}


def _run_g5() -> dict[str, Any]:
    dt = Generator(LinearPoly(), LinearPoly(c0=1.0), LinearPoly(), "dt")
    dx = Generator(LinearPoly(c0=1.0), LinearPoly(), LinearPoly(), "dx")
    r_dt = noether_wave_residual(dt)
    r_dx = noether_wave_residual(dx)
    ok = r_dt <= 1e-10 and r_dx <= 1e-10
    return {"name": "g5_noether", "passed": ok, "residual_dt": r_dt, "residual_dx": r_dx}


def _run_g6() -> dict[str, Any]:
    spec = negative_control()
    res = discover_symmetries(pr_for(spec.name), spec.restrict, affine_basis(), designed_samples(32))
    exact = res.exact
    ok = (
        res.algebra_dim == 0
        and res.disclaimer == DISCLAIMER
        and res.verified
        and exact is not None
        and exact.disproved
    )
    return {
        "name": "g6_negative",
        "passed": ok,
        "float_svd": {
            "dim": res.algebra_dim,
            "residual_checked": res.float_verified,
        },
        "exact_rank": exact.to_payload() if exact is not None else None,
        "exact_accepts_float_dimension": res.verified,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "symmetry_discovery.json" if full else "symmetry_discovery_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5(), _run_g6()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.symmetry_discovery.v1",
            config={"family": "symmetry_discovery", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "symmetry"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
