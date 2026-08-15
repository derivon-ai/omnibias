# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 primitive: mollifier calculus (theory 01-05 G1/G2/G3; G4 deferred).

Smoke (default) earns closed-form moments, order-m polynomial reproduction plus
an ``O(eps^m)`` rate, and tail-bound soundness. G4 (weak-form residual vs Gauss)
is deferred to ``benchmarks/weak_form_vpinn.py`` (theory 02-04 G1). ``--full``
writes under ``$OMNIBIAS_SCRATCH/mollifier/``.
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
    from omnibias.core.mollifier import MollifierSpec, moments
    from omnibias.core.multipack import PackSpec

    checks: list[dict[str, Any]] = []
    cases = (
        ("sigmoid", 2, math.pi**2 / 3.0),
        ("tanh", 2, math.pi**2 / 12.0),
        ("gaussian", 2, 1.0),
        ("sech", 2, math.pi**2 / 12.0),
        ("logistic", 2, math.pi**2 / 3.0),
    )
    worst = 0.0
    passed = True
    for base, idx, truth in cases:
        spec = MollifierSpec(base, 1.0, (PackSpec(0, 0.0),))
        got = moments(spec, idx)[idx]
        rel = abs(got - truth) / max(abs(truth), 1e-30)
        worst = max(worst, rel)
        ok = rel <= 1e-12
        passed = passed and ok
        checks.append({"base": base, "index": idx, "rel": rel, "passed": ok})
    return {
        "name": "g1_moments",
        "passed": passed,
        "worst_rel": worst,
        "expected": 1e-12,
        "checks": checks,
    }


def _run_g2() -> dict[str, Any]:
    from omnibias.core.mollifier import design_order, moments

    spec4 = design_order("tanh", 4, scale=0.1)
    m = moments(spec4, 3)
    poly_ok = all(abs(v - (1.0 if j == 0 else 0.0)) <= 1e-12 for j, v in enumerate(m))
    errs: list[float] = []
    scales = (0.2, 0.1, 0.05, 0.025)
    for eps in scales:
        errs.append(abs(moments(design_order("tanh", 2, scale=eps), 2)[2]))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    rate_ok = all(3.5 <= r <= 4.5 for r in ratios)
    return {
        "name": "g2_order",
        "passed": bool(poly_ok and rate_ok),
        "poly_rel_max": float(max(abs(v - (1.0 if j == 0 else 0.0)) for j, v in enumerate(m))),
        "rate_ratios": ratios,
        "note": "order-4 exact on degree < 4; order-2 error on u^2 is O(eps^2)",
    }


def _run_g3(*, full: bool) -> dict[str, Any]:
    from omnibias.core.mollifier import design_order, tail_bound, true_outside_mass

    rng = random.Random(1)
    n_extra = 80 if full else 20
    specs = [
        design_order("tanh", 2, scale=0.25),
        design_order("sigmoid", 2, scale=0.5),
        design_order("gaussian", 2, scale=0.4),
        design_order("tanh", 4, scale=0.3),
    ]
    widths = [0.5, 1.0, 1.5, 2.0, 3.0] + [
        rng.uniform(0.4, 4.0) for _ in range(n_extra)
    ]
    violations = 0
    n = 0
    for spec in specs:
        for w in widths:
            enclosed = tail_bound(spec, half_width=w)
            truth = true_outside_mass(spec, half_width=w)
            n += 1
            if not enclosed.contains(truth):
                violations += 1
    return {
        "name": "g3_tail_soundness",
        "passed": violations == 0 and n >= 100,
        "n": n,
        "violations": violations,
        "note": "analytic bases have certified exponential tails, not compact support",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2()
    g3 = _run_g3(full=args.full)
    payload = provenance(
        schema="mollifier-calculus-v1",
        config={
            "family": "mollifier_calculus",
            "full": bool(args.full),
            "g4_deferred": True,
            "gates_in_scope": ["g1", "g2", "g3"],
        },
    )
    payload["gates"] = gates_block([g1, g2, g3])
    payload["g1"] = g1
    payload["g2"] = g2
    payload["g3"] = g3
    payload["honesty"] = {
        "compact_support_claim": False,
        "higher_order_is_density": False,
        "g4_deferred_to": "weak_form_vpinn G1",
        "collapse": "delta -> 0 (eps -> 0); no temperature collapse",
    }
    if args.full:
        out_dir = SCRATCH / "mollifier"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "mollifier_calculus.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        path = write_json("mollifier_calculus_smoke.json", payload)
        print(f"wrote {path}")
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
