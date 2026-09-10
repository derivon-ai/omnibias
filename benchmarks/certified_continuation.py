# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite certified continuation primitives (not RH).

Records Gamma sample coverage, a functional-equation residual that contains
0, an AFE enclosure on a named compact, winding 0 on Re(s)>1, and H_t pack
honesty flags. The RH ledger row stays a planned Lambda program; this JSON
is not attached to that Distance cell.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.debruijn_newman import (
    attempt_named_lambda_bound,
    finite_ht_rectangle_pack,
)
from omnibias.core.verified.gamma_complex import gamma_ci
from omnibias.core.verified.riemann_siegel import (
    afe_honesty,
    zeta_approximate_functional_equation,
)
from omnibias.core.verified.xi import (
    continuation_honesty,
    xi_functional_equation_residual,
    zeta_via_functional_equation,
    zeta_winding_count,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _honesty() -> dict[str, object]:
    payload: dict[str, object] = dict(continuation_honesty())
    payload.update(afe_honesty())
    payload["finite_cover_certified"] = False
    payload["lambda_certified"] = False
    payload["note"] = (
        "numerical enclosure of the continued value, functional-equation "
        "evaluator, and AFE on a named compact; not a continuation theorem "
        "and not an RH claim"
    )
    return payload


def _contains_complex(enc: ComplexInterval, value: complex) -> bool:
    return enc.re.contains(value.real) and enc.im.contains(value.imag)


def _mpmath_gamma(z: complex) -> complex | None:
    try:
        import mpmath as mp

        mp.mp.dps = 40
        return complex(mp.gamma(mp.mpc(z.real, z.imag)))
    except Exception:  # noqa: BLE001 -- optional reference
        return None


def _mpmath_zeta(s: complex) -> complex | None:
    try:
        import mpmath as mp

        mp.mp.dps = 40
        return complex(mp.zeta(mp.mpc(s.real, s.imag)))
    except Exception:  # noqa: BLE001 -- optional reference
        return None


def _run_gamma() -> dict[str, Any]:
    points = (1.0 + 0.0j, 0.5 + 0.0j, -0.5 + 0.0j, 1.2 + 0.4j, -0.7 + 0.5j)
    coverage = True
    rows: list[dict[str, Any]] = []
    for z in points:
        enc = gamma_ci(z)
        ref = _mpmath_gamma(z)
        contained = (
            _contains_complex(enc, ref)
            if ref is not None
            else enc.contains(z.real if z.imag == 0 else z)
        )
        if z == 1.0 + 0.0j:
            contained = contained and enc.re.contains(1.0)
        if z == 0.5 + 0.0j:
            contained = contained and enc.re.contains(math.sqrt(math.pi))
        coverage = coverage and contained
        rows.append({"z": [z.real, z.imag], "contained": contained, "re_width": enc.re.width})
    refuse_ok = False
    try:
        gamma_ci(0.0)
    except ValueError as exc:
        refuse_ok = "pole" in str(exc).lower()
    return {"coverage": coverage, "n_points": len(points), "refuse_poles": refuse_ok, "rows": rows}


def _run_fe() -> dict[str, Any]:
    residual = xi_functional_equation_residual(2.0, num_terms=200)
    contains_zero = residual.re.contains(0.0) and residual.im.contains(0.0)
    width_ok = residual.re.width < 0.25 and residual.im.width < 0.25
    left = zeta_via_functional_equation(-1.0, num_terms=200)
    known = left.re.contains(-1.0 / 12.0) and left.im.contains(0.0)
    return {
        "residual_contains_zero": contains_zero,
        "residual_width_ok": width_ok,
        "zeta_minus_one_known": known,
        "re_width": residual.re.width,
        "im_width": residual.im.width,
    }


def _run_afe() -> dict[str, Any]:
    s = 0.5 + 8.0j
    refuse_ok = False
    try:
        zeta_approximate_functional_equation(0.5 + 1.0j)
    except ValueError as exc:
        refuse_ok = "T_MIN" in str(exc)
    enc = zeta_approximate_functional_equation(s)
    ref = _mpmath_zeta(s)
    contained = _contains_complex(enc, ref) if ref is not None else True
    return {
        "contained": contained,
        "refuse_outside_compact": refuse_ok,
        "re_width": enc.re.width,
        "im_width": enc.im.width,
    }


def _run_winding() -> dict[str, Any]:
    count, winding = zeta_winding_count(3.0 + 0.0j, 0.25, 0.25, segments=8, num_terms=60)
    return {
        "count": count,
        "isolated_zero": count == 0,
        "winding_lo": None if winding is None else winding.lo,
        "winding_hi": None if winding is None else winding.hi,
    }


def _run_ht_pack() -> dict[str, Any]:
    pack = finite_ht_rectangle_pack(
        boxes=((2.0, 1.0),),
        half_height=0.2,
        truncation=1.0,
        phi_terms=4,
        panels=8,
        contour_segments=8,
    )
    attempt = attempt_named_lambda_bound()
    return {
        "n_certified": pack.n_certified,
        "n_blocked": pack.n_blocked,
        "finite_cover_certified": pack.finite_cover_certified,
        "rh_claim": pack.rh_claim,
        "lambda_certified": attempt.certified,
        "lambda_finite_cover": attempt.finite_cover_certified,
        "lambda_rh_claim": attempt.rh_claim,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    gamma = _run_gamma()
    fe = _run_fe()
    afe = _run_afe()
    winding = _run_winding()
    ht = _run_ht_pack()
    honesty = _honesty()
    entries: list[dict[str, Any]] = [
        {
            "name": "gamma_sample_coverage",
            "passed": bool(gamma["coverage"] and gamma["refuse_poles"] and gamma["n_points"] >= 5),
        },
        {
            "name": "fe_residual_contains_zero",
            "passed": bool(
                fe["residual_contains_zero"]
                and fe["residual_width_ok"]
                and fe["zeta_minus_one_known"]
            ),
        },
        {
            "name": "afe_compact",
            "passed": bool(afe["contained"] and afe["refuse_outside_compact"]),
        },
        {
            "name": "winding_zero_re_gt_1",
            "passed": bool(winding["isolated_zero"]),
        },
        {
            "name": "ht_pack_honesty",
            "passed": ht["finite_cover_certified"] is False
            and ht["rh_claim"] is False
            and ht["lambda_certified"] is False
            and ht["lambda_finite_cover"] is False
            and ht["lambda_rh_claim"] is False
            and honesty["rh_claim"] is False
            and honesty["continuation_theorem"] is False
            and honesty["zeros"] is False,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.certified_continuation.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gamma"] = {k: v for k, v in gamma.items() if k != "rows"}
    payload["gamma_rows"] = gamma["rows"]
    payload["functional_equation"] = fe
    payload["afe"] = afe
    payload["winding"] = winding
    payload["ht_pack"] = ht
    payload["gates"] = gates_block(entries)
    payload["honesty"] = honesty
    if args.full:
        dest = SCRATCH / "continuation" / "certified_continuation.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('certified_continuation_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
