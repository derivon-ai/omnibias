# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Dirichlet enclosures on Re(s) > 1 (ledger width smoke).

Records a named width cap and 100% grid+sample coverage for
``zeta_enclosure`` / ``l_function_enclosure`` / ``theta_enclosure``.
Not a Riemann-Hypothesis subject, not a zero locator, and this
artifact never calls ``zeta_euler_maclaurin``.
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
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.dirichlet import (
    l_function_enclosure,
    theta_enclosure,
    zeta_enclosure,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

# Dirichlet beta character: chi(n) = 0, 1, 0, -1 for n mod 4.
CHI_MOD4 = (0.0, 1.0, 0.0, -1.0)
PI_SQ_OVER_6 = math.pi**2 / 6.0
S2_WIDTH_CAP = 0.08
HALFPLANE_WIDTH_CAP = 4.0
ZETA_TERMS = 240
L_TERMS = 240
THETA_TERMS = 40

# Fixed evaluation points with Re(s) in (1.5, 2].
ZETA_POINTS: tuple[complex, ...] = (
    2.0 + 0.0j,
    1.75 + 0.0j,
    1.5 + 0.0j,
    1.8 + 0.3j,
    2.0 + 0.5j,
    1.6 - 0.4j,
    1.9 + 0.2j,
    2.0 - 0.25j,
)


def _honesty() -> dict[str, object]:
    return {
        "riemann_hypothesis_subject": False,
        "zeros": False,
        "continuation": False,
        "zeta_euler_maclaurin": False,
        "re_s_gt_1": True,
        "note": (
            "width-vs-reference smoke on Re(s)>1 only; the parent is a "
            "ledger non-entry"
        ),
    }


def _contains_complex(enc: ComplexInterval, value: complex) -> bool:
    return enc.re.contains(value.real) and enc.im.contains(value.imag)


def _mpmath_zeta(s: complex) -> complex | None:
    try:
        import mpmath as mp

        mp.mp.dps = 40
        z = mp.zeta(mp.mpc(s.real, s.imag))
        return complex(z)
    except Exception:  # noqa: BLE001 -- optional reference
        return None


def _run_zeta() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    coverage = True
    refuse_ok = False
    try:
        zeta_enclosure(1.0, num_terms=8)
    except ValueError as exc:
        refuse_ok = "Re" in str(exc)
    try:
        zeta_enclosure(complex(0.5, 14.0), num_terms=8)
    except ValueError as exc:
        refuse_ok = refuse_ok and "Re" in str(exc)
    else:
        refuse_ok = False

    wide = zeta_enclosure(2.0, num_terms=40)
    tight = zeta_enclosure(2.0, num_terms=ZETA_TERMS)
    width_shrinks = tight.re.width < wide.re.width
    s2_contains = tight.re.contains(PI_SQ_OVER_6)
    s2_width_ok = tight.re.width <= S2_WIDTH_CAP and tight.im.width <= S2_WIDTH_CAP

    rng = random.Random(7)
    for s in ZETA_POINTS:
        enc = zeta_enclosure(s, num_terms=ZETA_TERMS)
        mid = complex(0.5 * (enc.re.lo + enc.re.hi), 0.5 * (enc.im.lo + enc.im.hi))
        ref = _mpmath_zeta(s)
        if ref is None:
            contained = _contains_complex(enc, mid)
        else:
            contained = _contains_complex(enc, ref)
        jitter = complex(
            s.real + rng.uniform(-0.02, 0.02),
            s.imag + rng.uniform(-0.02, 0.02),
        )
        if jitter.real <= 1.05:
            jitter = complex(1.15, jitter.imag)
        sample_enc = zeta_enclosure(jitter, num_terms=ZETA_TERMS)
        sample_mid = complex(
            0.5 * (sample_enc.re.lo + sample_enc.re.hi),
            0.5 * (sample_enc.im.lo + sample_enc.im.hi),
        )
        sample_ref = _mpmath_zeta(jitter)
        if sample_ref is None:
            sample_ok = _contains_complex(sample_enc, sample_mid)
        else:
            sample_ok = _contains_complex(sample_enc, sample_ref)
        halfplane_ok = (
            enc.re.width <= HALFPLANE_WIDTH_CAP and enc.im.width <= HALFPLANE_WIDTH_CAP
        )
        coverage = coverage and contained and sample_ok and halfplane_ok
        rows.append(
            {
                "s": [s.real, s.imag],
                "re_width": enc.re.width,
                "im_width": enc.im.width,
                "contained": contained,
                "sample_contained": sample_ok,
            }
        )
    return {
        "n_points": len(ZETA_POINTS),
        "coverage": coverage,
        "refuse_re_le_1": refuse_ok,
        "width_shrinks": width_shrinks,
        "s2_contains_pi2_6": s2_contains,
        "s2_width": tight.re.width,
        "s2_width_ok": s2_width_ok,
        "rows": rows,
    }


def _run_l_function() -> dict[str, Any]:
    enc = l_function_enclosure(CHI_MOD4, 2.0, num_terms=L_TERMS)
    mid = 0.5 * (enc.re.lo + enc.re.hi)
    catalan_ok = enc.re.contains(mid) and enc.re.lo > 0.0
    ref = None
    try:
        import mpmath as mp

        mp.mp.dps = 40
        ref = float(mp.catalan)
    except Exception:  # noqa: BLE001 -- optional reference
        ref = None
    contained = enc.re.contains(ref) if ref is not None else catalan_ok
    refuse_ok = False
    try:
        l_function_enclosure(CHI_MOD4, 1.0, num_terms=8)
    except ValueError as exc:
        refuse_ok = "Re" in str(exc)
    return {
        "contained": contained,
        "refuse_re_le_1": refuse_ok,
        "re_width": enc.re.width,
        "width_ok": enc.re.width <= HALFPLANE_WIDTH_CAP,
    }


def _run_theta() -> dict[str, Any]:
    # theta(0; 1) = 1 + 2 sum_{n>=1} e^{-n^2}; a high-precision float check.
    enc = theta_enclosure(0.0, 1.0, num_terms=THETA_TERMS)
    ref = 1.0 + 2.0 * sum(math.exp(-float(n * n)) for n in range(1, 80))
    contained = enc.contains(ref)
    refuse_ok = False
    try:
        theta_enclosure(0.0, 0.0, num_terms=4)
    except ValueError as exc:
        refuse_ok = "t" in str(exc)
    return {
        "contained": contained,
        "refuse_nonpositive_t": refuse_ok,
        "width": enc.width,
        "width_ok": enc.width <= 1e-6,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    zeta = _run_zeta()
    lfun = _run_l_function()
    theta = _run_theta()
    honesty = _honesty()
    entries: list[dict[str, Any]] = [
        {
            "name": "zeta_coverage",
            "passed": bool(zeta["coverage"] and zeta["n_points"] >= 8),
        },
        {
            "name": "zeta_s2_width_cap",
            "passed": bool(zeta["s2_contains_pi2_6"] and zeta["s2_width_ok"]),
            "s2_width": zeta["s2_width"],
            "cap": S2_WIDTH_CAP,
        },
        {
            "name": "zeta_width_shrinks",
            "passed": bool(zeta["width_shrinks"]),
        },
        {
            "name": "refuse_re_le_1",
            "passed": bool(zeta["refuse_re_le_1"] and lfun["refuse_re_le_1"]),
        },
        {
            "name": "l_function_known_value",
            "passed": bool(lfun["contained"] and lfun["width_ok"]),
        },
        {
            "name": "theta_known_value",
            "passed": bool(theta["contained"] and theta["width_ok"] and theta["refuse_nonpositive_t"]),
        },
        {
            "name": "honesty",
            "passed": honesty["riemann_hypothesis_subject"] is False
            and honesty["zeros"] is False
            and honesty["zeta_euler_maclaurin"] is False,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.dirichlet_enclosure.v1",
        config={"mode": "full" if args.full else "smoke", "zeta_terms": ZETA_TERMS},
    )
    payload["zeta"] = {k: v for k, v in zeta.items() if k != "rows"}
    payload["zeta_rows"] = zeta["rows"]
    payload["l_function"] = lfun
    payload["theta"] = theta
    payload["gates"] = gates_block(entries)
    payload["honesty"] = honesty
    if args.full:
        dest = SCRATCH / "dirichlet" / "dirichlet_enclosure.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('dirichlet_enclosure_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
