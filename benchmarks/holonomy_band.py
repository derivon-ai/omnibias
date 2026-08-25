# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: holonomy band (theory 02-14). No YM / mass-gap claim.

G2 closed-form exactness is measured: ``band_holonomy`` versus PRODUCT
at ``substeps=4096``. Closed form is abelian + transverse-constant only.
The gap is held finite (band), the opposite of founding ``delta -> 0``.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.geometry.gauge.band._core import (
        BandRegime,
        classify_regime,
        magnus_truncation_bound,
        open_line_is_gauge_dependent,
    )
    from omnibias.geometry.gauge._core.lie_algebra import su, u1

    g1 = classify_regime(u1(), transverse_constant=False) is BandRegime.ABELIAN
    g1 = g1 and classify_regime(su(2), transverse_constant=False) is BandRegime.PRODUCT
    bound = magnus_truncation_bound(a_norm=0.4, length=1.0, order=2)
    g2 = _run_g2()
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
            "name": "g3_magnus_bound",
            "passed": bound.lo < 0.0 < bound.hi,
            "in_ci_all_passed": True,
        },
        {
            "name": "g4_open_line_flagged",
            "passed": open_line_is_gauge_dependent() is True,
            "in_ci_all_passed": True,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.holonomy_band.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "g2_in_all_passed": bool(g2["in_ci_all_passed"]),
            "gates_in_scope": ["g1", "g2", "g3", "g4"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["g2"] = g2
    payload["honesty"] = {
        "closed_form": "abelian and transverse-constant only",
        "open_lines": "gauge-dependent",
        "yang_mills": False,
        "mass_gap": False,
        "continuum_claim": False,
        "g2_earned": bool(g2["earned"]),
        "g2_reported": True,
        "g2_in_ci_all_passed": bool(g2["in_ci_all_passed"]),
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
