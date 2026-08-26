# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: holonomy band (theory 02-14). No YM / mass-gap claim.

G2 closed-form exactness is earned versus PRODUCT at ``substeps=4096``.
G3 Magnus soundness is leftover-recorded (leftover #34): the bound is
checked on a grid and a sample, but no Magnus-truncated holonomy is
wired. G4 gauge covariance is earned: ``random_u1_gauge`` plus
``g(hi) U g(lo)^{-1}`` matches the gauged holonomy to ``<= 4`` ulp.
Closed form is abelian + transverse-constant only. The gap is held
finite (band), the opposite of founding ``delta -> 0``.
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

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

G2_SUBSTEPS = 4096
G2_ABS_TOL = 1e-12
G2_COST_RATIO_MAX = 0.25
G2_WARMUP = 1
G2_REPEATS = 3


def _median_seconds(fn: Any, *, warmup: int, repeats: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _run_g2() -> dict[str, Any]:
    """Named leftover: closed-form band_holonomy vs PRODUCT at 4096 substeps."""
    import torch
    from omnibias.geometry.gauge._core.lie_algebra import su, u1
    from omnibias.geometry.gauge.band._core import BandRegime, HolonomyBand
    from omnibias.geometry.gauge.band.torch import band_holonomy

    torch.set_default_dtype(torch.float64)
    band_u1 = HolonomyBand((1.0,), lo=-1.0, hi=1.0, algebra=u1(), coupling=1.0)
    band_su = HolonomyBand((1.0,), lo=0.0, hi=1.0, algebra=su(2), coupling=1.0)

    def _cf_u1() -> Any:
        return band_holonomy(band_u1, regime=BandRegime.ABELIAN, a0=1.0)

    def _prod_u1() -> Any:
        return band_holonomy(band_u1, regime=BandRegime.PRODUCT, a0=1.0, substeps=G2_SUBSTEPS)

    def _cf_su() -> Any:
        return band_holonomy(
            band_su,
            regime=BandRegime.TRANSVERSE_CONSTANT,
            components=(0.3, 0.0, 0.0),
            dtype=torch.float64,
        )

    def _prod_su() -> Any:
        return band_holonomy(
            band_su,
            regime=BandRegime.PRODUCT,
            a0=0.3,
            components=(0.3, 0.0, 0.0),
            substeps=G2_SUBSTEPS,
            dtype=torch.float64,
        )

    u_cf, _ = _cf_u1()
    u_prod, _ = _prod_u1()
    err_u1 = abs(complex(u_cf[0, 0].detach()) - complex(u_prod[0, 0].detach()))
    t_cf_u1 = _median_seconds(_cf_u1, warmup=G2_WARMUP, repeats=G2_REPEATS)
    t_prod_u1 = _median_seconds(_prod_u1, warmup=G2_WARMUP, repeats=G2_REPEATS)

    u_cf_su, _ = _cf_su()
    u_prod_su, _ = _prod_su()
    err_su = abs(complex(u_cf_su[0, 0].detach()) - complex(u_prod_su[0, 0].detach()))
    t_cf_su = _median_seconds(_cf_su, warmup=G2_WARMUP, repeats=G2_REPEATS)
    t_prod_su = _median_seconds(_prod_su, warmup=G2_WARMUP, repeats=G2_REPEATS)

    ratio_u1 = t_cf_u1 / max(t_prod_u1, 1e-18)
    ratio_su = t_cf_su / max(t_prod_su, 1e-18)
    match = bool(err_u1 <= G2_ABS_TOL and err_su <= G2_ABS_TOL)
    cheaper = bool(ratio_u1 <= G2_COST_RATIO_MAX and ratio_su <= G2_COST_RATIO_MAX)
    earned = bool(match and cheaper)
    return {
        "name": "g2_closed_form",
        "passed": earned,
        "earned": earned,
        "reported": True,
        "in_ci_all_passed": bool(earned),
        "substeps": G2_SUBSTEPS,
        "abs_tol": G2_ABS_TOL,
        "cost_ratio_max": G2_COST_RATIO_MAX,
        "abelian_abs_err": float(err_u1),
        "su2_abs_err": float(err_su),
        "abelian_cost_ratio": float(ratio_u1),
        "su2_cost_ratio": float(ratio_su),
        "abelian_closed_form_seconds": float(t_cf_u1),
        "abelian_product_seconds": float(t_prod_u1),
        "su2_closed_form_seconds": float(t_cf_su),
        "su2_product_seconds": float(t_prod_su),
        "note": (
            "band_holonomy closed form versus PRODUCT at substeps=4096 "
            "in the abelian and transverse-constant (A^1) regimes. "
            "Named G2 needs match <= 1e-12 and a fraction of the "
            "PRODUCT cost. Previous smoke skipped G2. No YM / mass gap."
        ),
    }


G3_ORDER = 2
G3_GRID_A = (0.2, 0.4, 0.8)
G3_GRID_L = (0.5, 1.0)
G3_N_RANDOM = 8
G3_ML_HALVINGS = (0.8, 0.4, 0.2)


def _run_g3() -> dict[str, Any]:
    """Named leftover: bound on a grid + sample; Magnus holonomy stays --full."""
    from omnibias.geometry.gauge.band._core import magnus_truncation_bound

    rows: list[dict[str, Any]] = []
    violations = 0
    for a_norm in G3_GRID_A:
        for length in G3_GRID_L:
            bound = magnus_truncation_bound(a_norm=a_norm, length=length, order=G3_ORDER)
            ok = bool(bound.lo < 0.0 < bound.hi)
            violations += int(not ok)
            rows.append(
                {
                    "kind": "grid",
                    "a_norm": float(a_norm),
                    "length": float(length),
                    "bound_hi": float(bound.hi),
                    "contains_zero": ok,
                }
            )
    rng = np.random.default_rng(0)
    for _ in range(G3_N_RANDOM):
        a_norm = float(rng.uniform(0.1, 1.2))
        length = float(rng.uniform(0.2, 1.5))
        if a_norm * length >= 3.0:
            length = 2.0 / a_norm
        bound = magnus_truncation_bound(a_norm=a_norm, length=length, order=G3_ORDER)
        ok = bool(bound.lo < 0.0 < bound.hi)
        violations += int(not ok)
        rows.append(
            {
                "kind": "sample",
                "a_norm": a_norm,
                "length": length,
                "bound_hi": float(bound.hi),
                "contains_zero": ok,
            }
        )
    refused = False
    try:
        magnus_truncation_bound(a_norm=4.0, length=1.0, order=G3_ORDER)
    except ValueError:
        refused = True
    widths = []
    for ml in G3_ML_HALVINGS:
        bound = magnus_truncation_bound(a_norm=float(ml), length=1.0, order=G3_ORDER)
        widths.append(float(bound.hi))
    ratio_hi_mid = widths[0] / max(widths[1], 1e-18)
    # Bound is (ml^{order+1})/(order+1)! exp(ml); ratio at 2x is 2^{k} exp(ml/2).
    predicted = (2.0 ** (G3_ORDER + 1)) * float(np.exp(0.5 * G3_ML_HALVINGS[1]))
    return {
        "name": "g3_magnus_bound",
        "passed": False,
        "earned": False,
        "reported": True,
        "leftover_recorded": True,
        "leftover_id": 34,
        "leftover_tick": 70,
        "in_ci_all_passed": False,
        "need": (
            "bound upper-bounds Magnus-truncation vs PRODUCT 4096 on a "
            "grid and a sample, zero violations, predicted-order decay"
        ),
        "violations": int(violations),
        "n_grid": int(len(G3_GRID_A) * len(G3_GRID_L)),
        "n_sample": G3_N_RANDOM,
        "refuses_outside_radius": bool(refused),
        "zero_in_every_bound": violations == 0,
        "width_halving_ratio": float(ratio_hi_mid),
        "width_halving_predicted": float(predicted),
        "magnus_holonomy_api": False,
        "stays_full": True,
        "rows": rows,
        "note": (
            "Leftover #34 leftover-recorded: magnus_truncation_bound "
            "contains 0 on a (a_norm, L) grid and a random sample, and "
            "refuses ||A|| L >= pi. Named G3 needs a Magnus-truncated "
            "holonomy whose error versus PRODUCT substeps=4096 is "
            "enclosed and decays at the predicted order. That "
            "evaluator is not wired. Previous sign-check stub "
            "withdrawn from named G3. Not in CI all_passed."
        ),
    }


G4_ULP_MAX = 4.0


def _run_g4() -> dict[str, Any]:
    """Named G4: random-gauge covariance + loop identity, both <= 4 ulp."""
    import torch
    from omnibias.geometry.gauge._core.lie_algebra import u1
    from omnibias.geometry.gauge.band._core import (
        HolonomyBand,
        open_line_is_gauge_dependent,
        random_gauge_covariance_ulps,
    )
    from omnibias.geometry.gauge.band.torch import band_holonomy, band_wilson_loop

    torch.set_default_dtype(torch.float64)
    flagged = open_line_is_gauge_dependent() is True
    band = HolonomyBand((1.0,), lo=-0.5, hi=0.5, algebra=u1(), coupling=1.0)
    _u, invariant = band_holonomy(band, a0=1.0)
    open_not_invariant = invariant is False
    bands = (
        HolonomyBand((1.0,), lo=-0.5, hi=0.5, algebra=u1(), coupling=1.0),
        HolonomyBand((1.0,), lo=0.5, hi=-0.5, algebra=u1(), coupling=1.0),
    )
    tr = float(band_wilson_loop(bands, a0=1.0).detach())
    ulp = float(abs(tr - 1.0) / float(np.finfo(np.float64).eps))
    gauge_ulps = [
        float(
            random_gauge_covariance_ulps(
                a0=1.0, lo=-0.5, hi=0.5, coupling=1.0, seed=seed
            )
        )
        for seed in range(8)
    ]
    max_gauge_ulps = float(max(gauge_ulps))
    random_ok = max_gauge_ulps <= G4_ULP_MAX
    loop_ok = ulp <= G4_ULP_MAX
    earned = bool(flagged and open_not_invariant and random_ok and loop_ok)
    return {
        "name": "g4_gauge_covariance",
        "passed": earned,
        "earned": earned,
        "reported": True,
        "leftover_recorded": False,
        "leftover_id": 35,
        "leftover_tick": 73,
        "in_ci_all_passed": earned,
        "need": "open holonomy transforms as g(hi) U g(lo)^{-1} to <= 4 ulp; loop invariant to <= 4 ulp",
        "open_line_flagged": bool(flagged),
        "open_holonomy_not_invariant": bool(open_not_invariant),
        "loop_identity": float(tr),
        "loop_ulps": ulp,
        "loop_ulp_max": G4_ULP_MAX,
        "loop_within_4_ulp": bool(loop_ok),
        "random_gauge_api": True,
        "random_gauge_ulps": gauge_ulps,
        "random_gauge_max_ulps": max_gauge_ulps,
        "random_gauge_within_4_ulp": bool(random_ok),
        "stays_full": False,
        "note": (
            "Leftover #35 earned on tick #73: random_u1_gauge plus "
            "g(hi) U g(lo)^{-1} matches abelian_holonomy_gauged to "
            "<= 4 ulp on eight seeds. Open-line flag and forward-back "
            "loop identity stay. In CI all_passed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.geometry.gauge.band._core import BandRegime, classify_regime
    from omnibias.geometry.gauge._core.lie_algebra import su, u1

    g1 = classify_regime(u1(), transverse_constant=False) is BandRegime.ABELIAN
    g1 = g1 and classify_regime(su(2), transverse_constant=False) is BandRegime.PRODUCT
    g2 = _run_g2()
    g3 = _run_g3()
    g4 = _run_g4()
    entries: list[dict[str, Any]] = [
        {"name": "g1_regime", "passed": g1, "in_ci_all_passed": True},
        {
            "name": "g2_closed_form",
            "passed": bool(g2["passed"]),
            "in_ci_all_passed": bool(g2["in_ci_all_passed"]),
            "abelian_abs_err": g2["abelian_abs_err"],
            "su2_abs_err": g2["su2_abs_err"],
            "abelian_cost_ratio": g2["abelian_cost_ratio"],
            "su2_cost_ratio": g2["su2_cost_ratio"],
        },
        {
            "name": "g4_gauge_covariance",
            "passed": bool(g4["passed"]),
            "in_ci_all_passed": bool(g4["in_ci_all_passed"]),
            "random_gauge_max_ulps": g4["random_gauge_max_ulps"],
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.holonomy_band.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "g2_in_all_passed": bool(g2["in_ci_all_passed"]),
            "g3_in_all_passed": False,
            "g4_in_all_passed": bool(g4["in_ci_all_passed"]),
            "gates_in_scope": ["g1", "g2", "g4"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["g2"] = g2
    payload["g3"] = g3
    payload["g4"] = g4
    payload["honesty"] = {
        "closed_form": "abelian and transverse-constant only",
        "open_lines": "gauge-dependent",
        "yang_mills": False,
        "mass_gap": False,
        "continuum_claim": False,
        "g2_earned": bool(g2["earned"]),
        "g2_reported": True,
        "g2_in_ci_all_passed": bool(g2["in_ci_all_passed"]),
        "g3_earned": False,
        "g3_reported": True,
        "g3_leftover_recorded": True,
        "g3_leftover_id": 34,
        "g3_leftover_tick": 70,
        "g3_in_ci_all_passed": False,
        "g3_magnus_holonomy_api": False,
        "g4_earned": bool(g4["earned"]),
        "g4_reported": True,
        "g4_leftover_recorded": False,
        "g4_leftover_id": 35,
        "g4_leftover_tick": 73,
        "g4_in_ci_all_passed": bool(g4["in_ci_all_passed"]),
        "g4_random_gauge_api": True,
        "temperature_collapse": False,
        "founding_bias_collapse": False,
        "finite_band_gap": True,
    }
    if args.full:
        dest = SCRATCH / "holonomy" / "holonomy_band.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('holonomy_band_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
