# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: neural quadrature (theory 03-06).

Smoke earns G1 (Gauss–Legendre recovery n=2..12), G2 (moment
exactness including pack cancellation), G3 (Peano enclosure
contains and is attained on x^4), G4 (designed family vs GL,
five seeds, 10x), and G5 (tensor n^D table; no non-product
claim). Pack moments are founding bias collapse, not temperature
collapse. Refuses without a derivative bound.
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
from omnibias.core.cubature import (
    MomentSystem,
    apply_rule,
    certified_error,
    design_rule,
    dimension_scaling_table,
    honesty_payload,
    solve_rule,
)
from omnibias.core.verified.interval import Interval

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0.05, 0.07, 0.09, 0.11, 0.13)


def _run_g1() -> dict[str, Any]:
    system = MomentSystem.lebesgue(24)
    worst = 0.0
    for n in range(2, 13):
        rule = solve_rule(system, nodes=n, free_nodes=True)
        xs, ws = np.polynomial.legendre.leggauss(n)
        worst = max(
            worst,
            float(np.max(np.abs(np.sort(rule.nodes) - np.sort(xs)))),
            float(np.max(np.abs(np.sort(rule.weights) - np.sort(ws)))),
        )
    return {"name": "g1_recovery", "passed": worst <= 1e-13, "worst": worst, "detail": "GL nodes/weights n=2..12"}


def _run_g2() -> dict[str, Any]:
    system = MomentSystem.lebesgue(8)
    rule = solve_rule(system, nodes=3, free_nodes=True)
    ok_pt = all(abs(rule.apply_monomial(j) - system.target_moments[j]) <= 1e-13 * max(1.0, abs(system.target_moments[j])) for j in range(rule.degree + 1))
    pack = solve_rule(system, nodes=2, free_nodes=True, functional="pack", pack_scale=0.1, pack_base="gaussian")
    ok_pk = all(abs(pack.apply_monomial(j) - system.target_moments[j]) <= 1e-13 * max(1.0, abs(system.target_moments[j])) for j in (0, 2))
    return {"name": "g2_moments", "passed": ok_pt and ok_pk and pack.bias_cancelled, "detail": "point + cancelled pack m0/m2"}


def _run_g3() -> dict[str, Any]:
    rule = solve_rule(MomentSystem.lebesgue(6), nodes=2, free_nodes=True)
    true_err = apply_rule(rule, lambda x: x**4) - 0.4
    enc = certified_error(rule, deriv_bound=Interval.point(24.0), degree=3)
    refused = False
    try:
        certified_error(rule, deriv_bound=None, degree=3)
    except ValueError:
        refused = True
    ok = enc.contains(true_err) and abs(true_err) >= 0.4 * enc.mag and refused
    return {"name": "g3_peano", "passed": ok, "true_err": true_err, "bound_mag": enc.mag, "detail": "x^4 attained; refuse without bound"}


def _run_g4() -> dict[str, Any]:
    gl = solve_rule(MomentSystem.lebesgue(6), nodes=2, free_nodes=True)
    integrands = []
    exacts = []
    for a in SEEDS:
        integrands.append(lambda x, coeff=a: x**4 + coeff * x**6)
        exacts.append(2.0 / 5.0 + a * (2.0 / 7.0))
    designed = design_rule(integrands, exacts, nodes=2)
    ratios = []
    for fn, exact in zip(integrands, exacts, strict=True):
        e_gl = abs(apply_rule(gl, fn) - exact)
        e_des = abs(apply_rule(designed, fn) - exact)
        ratios.append(e_gl / max(e_des, 1e-18))
    return {"name": "g4_design", "passed": float(np.mean(ratios)) >= 10.0 and all(r > 1.0 for r in ratios), "mean_ratio": float(np.mean(ratios)), "ratios": ratios, "detail": "x^4+a x^6 vs GL-2"}


def _run_g5() -> dict[str, Any]:
    table = dimension_scaling_table(nodes_per_axis=8, max_dim=6)
    prohibitive = min(d for d, cost in table if cost > 4096)
    return {
        "name": "g5_scope",
        "passed": prohibitive == 5 and honesty_payload()["non_product_cubature"] is False,
        "table": table,
        "prohibitive_dim": prohibitive,
        "detail": "n=8 tensor exceeds 4096 at D=5; no non-product cubature",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "neural_quadrature.json" if full else "neural_quadrature_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.neural_quadrature.v1",
            config={"family": "neural_quadrature", "full": full, "seeds": list(SEEDS), "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "quadrature"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
