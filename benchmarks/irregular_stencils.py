# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-1 primitive: irregular / Birkhoff stencils (theory 01-04 G1–G4).

Smoke (default) earns reproduction, rate, certificate coverage, and
poisedness gates. ``--full`` repeats the rate battery on a denser ``h``
grid under ``$OMNIBIAS_SCRATCH/stencils/``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    from omnibias.difference import (
        offsets_exact,
        physical_weights,
        signs_exact,
        solve_irregular_stencil,
    )
    from omnibias.difference._core.irregular import StencilRequest

    h = Fraction(1, 11)
    mismatches = 0
    checked = 0
    for order in range(0, 5):
        for kind in ("forward", "central"):
            offs = offsets_exact(order, Fraction(1), kind)
            nodes = tuple(o / Fraction(1) for o in offs)
            req = StencilRequest(nodes, tuple((0,) for _ in nodes), target_order=order)
            st = solve_irregular_stencil(req)
            checked += 1
            if st is None:
                mismatches += 1
                continue
            flat = tuple(w[0] for w in physical_weights(st, h))
            if flat != signs_exact(order, h):
                mismatches += 1
    return {
        "name": "g1_reproduction",
        "passed": mismatches == 0,
        "n_checked": checked,
        "n_mismatch": mismatches,
    }


def _run_g2(*, full: bool) -> dict[str, Any]:
    from omnibias.difference import apply_irregular_stencil, solve_irregular_stencil
    from omnibias.difference._core.irregular import StencilRequest

    req = StencilRequest(
        (Fraction(-1), Fraction(0), Fraction(1)),
        ((0,), (0,), (1,)),
        target_order=1,
    )
    st = solve_irregular_stencil(req)
    assert st is not None
    hs = [0.1, 0.05, 0.025, 0.0125] if not full else [0.1, 0.05, 0.025, 0.0125, 0.00625]

    def _rate(fn_samples, truth: float) -> float:
        errs = []
        for h in hs:
            est = apply_irregular_stencil(
                st, Fraction(h).limit_denominator(), fn_samples(h)
            )
            errs.append(abs(est - truth))
        slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
        return float(slope)

    rates = {
        "exp": _rate(lambda h: ((math.exp(-h),), (1.0,), (math.exp(h),)), 1.0),
        "sin": _rate(lambda h: ((math.sin(-h),), (0.0,), (math.cos(h),)), 1.0),
        "inv": _rate(
            lambda h: ((1.0 / (1.0 - h),), (1.0,), (-1.0 / (1.0 + h) ** 2,)),
            -1.0,
        ),
    }
    tol = 0.15
    passed = all(abs(r - st.accuracy) <= tol for r in rates.values())
    return {
        "name": "g2_order_verification",
        "passed": passed,
        "reported_accuracy": st.accuracy,
        "fitted_rates": rates,
        "tol": tol,
    }


def _run_g3(*, full: bool) -> dict[str, Any]:
    from omnibias.core.verified.interval import Interval
    from omnibias.difference import (
        apply_irregular_stencil,
        certified_irregular_error,
        solve_irregular_stencil,
    )
    from omnibias.difference._core.irregular import StencilRequest

    req = StencilRequest(
        (Fraction(-1), Fraction(0), Fraction(1)),
        ((0,), (0,), (1,)),
        target_order=1,
    )
    st = solve_irregular_stencil(req)
    assert st is not None
    rng = random.Random(1)
    n_grid = 24 if full else 12
    n_rand = 24 if full else 12
    hs = [0.2 / (k + 1) for k in range(n_grid)] + [
        rng.uniform(0.01, 0.2) for _ in range(n_rand)
    ]
    violations = 0
    worst_ratio = 0.0
    for h in hs:
        hh = Fraction(h).limit_denominator(10_000)
        h_f = float(hh)
        est = apply_irregular_stencil(
            st, hh, ((math.exp(-h_f),), (1.0,), (math.exp(h_f),))
        )
        err = abs(est - 1.0)
        cert = certified_irregular_error(
            st, h=hh, deriv_bound=Interval.point(math.exp(h_f)), estimate=est
        )
        ratio = err / cert.error_bound if cert.error_bound > 0 else float("inf")
        worst_ratio = max(worst_ratio, ratio)
        if err > cert.error_bound + 1e-15:
            violations += 1
    return {
        "name": "g3_certificate_soundness",
        "passed": violations == 0,
        "n_instances": len(hs),
        "violations": violations,
        "worst_err_over_bound": float(worst_ratio),
    }


def _run_g4() -> dict[str, Any]:
    from omnibias.difference import is_poised_exact, polya_screen, solve_irregular_stencil
    from omnibias.difference._core.irregular import StencilRequest

    hermite = StencilRequest(
        (Fraction(-1), Fraction(1)), ((0, 1), (0, 1)), target_order=1
    )
    worked = StencilRequest(
        (Fraction(-1), Fraction(0), Fraction(1)),
        ((0,), (0,), (1,)),
        target_order=1,
    )
    gap = StencilRequest((Fraction(0),), ((0, 2),), target_order=1)
    polya_not_rank = StencilRequest(
        (Fraction(-1), Fraction(0), Fraction(1)),
        ((0,), (1,), (0,)),
        target_order=0,
    )
    cases = {
        "hermite": (hermite, True, True),
        "worked_birkhoff": (worked, True, True),
        "one_node_gap": (gap, False, False),
        "polya_pass_rank_fail": (polya_not_rank, True, False),
    }
    ok = True
    report = {}
    for name, (req, polya, poised) in cases.items():
        got_p = polya_screen(req)
        got_r = is_poised_exact(req)
        report[name] = {"polya": got_p, "poised": got_r}
        if got_p is not polya or got_r is not poised:
            ok = False
        if poised and got_p is False:
            ok = False
        if poised != (solve_irregular_stencil(req) is not None):
            ok = False
    return {
        "name": "g4_poisedness",
        "passed": ok,
        "cases": report,
        "polya_is_necessary_only": True,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = (
        "irregular_stencils.json" if full else "irregular_stencils_smoke.json"
    )
    payload = provenance(
        schema="irregular-stencils-v1",
        config={"family": "irregular_stencils", "full": full},
    )
    t0 = time.perf_counter()
    print("G1 reproduction vs signs_exact...")
    g1 = _run_g1()
    print("G2 empirical rate...")
    g2 = _run_g2(full=full)
    print("G3 certificate coverage...")
    g3 = _run_g3(full=full)
    print("G4 poisedness...")
    g4 = _run_g4()
    entries = [g1, g2, g3, g4]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    gates = dict(gates_block(entries))
    payload.update(
        {
            "gates": gates,
            "g1": g1,
            "g2": g2,
            "g3": g3,
            "g4": g4,
            "honesty": {
                "claim_rung": 1,
                "bias_collapse": True,
                "temperature_collapse": False,
                "g1_earned": True,
                "g2_earned": True,
                "g3_earned": True,
                "g4_earned": True,
                "order_is_asymptotic_in_h": True,
                "scale_free_A_is_a_times_h_to_q_minus_p": True,
                "licensed_sentence": (
                    "Exact rational Birkhoff weights reproduce uniform "
                    "signs_exact; empirical rates match reported accuracy; "
                    "the truncation certificate covers a grid and a random "
                    "sample; is_poised_exact is correct on the curated set "
                    "and polya_screen never rejects a poised scheme"
                ),
            },
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )
    out = write_json(artifact, payload)
    print(f"wrote {out} all_passed={gates['all_passed']}")
    if full:
        scratch = SCRATCH / "stencils"
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / artifact).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied to {scratch / artifact}")
    if not gates["all_passed"]:
        raise SystemExit(1)
    return dict(payload)


if __name__ == "__main__":
    main()
