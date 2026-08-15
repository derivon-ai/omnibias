# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated primitive: conjugate Hilbert tower (theory 01-12 G1–G4; G5 not CI).

Line Hilbert only. Commutation needs ``alpha > 0``. G5 (dictionary capacity)
is a campaign artifact and is **not** in ``all_passed``; it does not clear
``CCF_STRETCH_RESIDUAL_GATE``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    from omnibias.core.verified.hardy_line import (
        hardy_even_deriv,
        hardy_even_deriv_n,
        hardy_odd_deriv,
        hardy_odd_deriv_n,
    )

    y, a, alpha = 1.0, 1.0, 0.5
    ok = (
        hardy_even_deriv_n(y, a, alpha, 1) == hardy_even_deriv(y, a, alpha)
        and hardy_odd_deriv_n(y, a, alpha, 1) == hardy_odd_deriv(y, a, alpha)
    )
    p1 = hardy_even_deriv_n(y, a, alpha, 1).mid
    return {
        "name": "g1_n1_interval_equal",
        "passed": ok and abs(p1 + 0.27467102) <= 1e-6,
        "p1_mid": p1,
        "in_ci_all_passed": True,
    }


def _run_g2() -> dict[str, Any]:
    from omnibias.core.verified.hardy_line import hardy_even_deriv_n

    p2 = hardy_even_deriv_n(1.0, 1.0, 0.5, 2).mid
    ok = abs(p2 - 0.12067393) <= 1e-6
    return {
        "name": "g2_worked_n2",
        "passed": ok,
        "p2_mid": p2,
        "note": "50-digit mpmath table vs diff is the unit-test G2; smoke checks the worked example",
        "in_ci_all_passed": True,
    }


def _run_g3() -> dict[str, Any]:
    from omnibias.core.conjugate import hardy_p_deriv_n
    from omnibias.core.verified.hardy_line import hardy_even_deriv_n_iv
    from omnibias.core.verified.interval import Interval

    violations = 0
    for i in range(-15, 16):
        y = i * 0.15
        box = Interval(y - 1e-8, y + 1e-8)
        iv = hardy_even_deriv_n_iv(box, 1.0, 0.7, 3)
        if not iv.contains(hardy_p_deriv_n(y, 1.0, 0.7, 3)):
            violations += 1
    return {
        "name": "g3_enclosure",
        "passed": violations == 0,
        "violations": violations,
        "in_ci_all_passed": True,
    }


def _run_g4() -> dict[str, Any]:
    from omnibias.core.verified.hardy_line import (
        hardy_odd_deriv_n,
        hilbert_of_hardy_even_deriv_n,
    )

    ok = True
    for n in range(0, 6):
        if hilbert_of_hardy_even_deriv_n(0.3, 1.2, 0.8, n) != hardy_odd_deriv_n(
            0.3, 1.2, 0.8, n
        ):
            ok = False
    return {
        "name": "g4_commutation",
        "passed": ok,
        "in_ci_all_passed": True,
    }


def _run_g5_capacity_record() -> dict[str, Any]:
    """Projection-defect comparison on a synthetic profile; not CCF stretch."""
    import math as _math

    from omnibias.core.conjugate import HardyAtom, HardyDictionary, evaluate

    # Fit a target P_{1, 0.5} + 0.3 P_{1, 2.5} with a 2-atom vs 4-atom (n=0,2) dict.
    ys = [i * 0.2 for i in range(-15, 16)]

    def target(y: float) -> float:
        r = _math.hypot(1.0, y)
        phi = _math.atan2(y, 1.0)
        p0 = r ** (-0.5) * _math.cos(0.5 * phi)
        p2 = r ** (-2.5) * _math.cos(2.5 * phi)
        return p0 + 0.3 * p2

    small = HardyDictionary(
        (HardyAtom(1.0, 0.5, 0, "even"), HardyAtom(1.0, 0.5, 0, "odd"))
    )
    big = HardyDictionary(
        (
            HardyAtom(1.0, 0.5, 0, "even"),
            HardyAtom(1.0, 0.5, 0, "odd"),
            HardyAtom(1.0, 0.5, 2, "even"),
            HardyAtom(1.0, 0.5, 2, "odd"),
        )
    )

    def defect(dictionary: HardyDictionary) -> float:
        # Least-squares on even atoms only (target is even).
        cols = []
        t = [target(y) for y in ys]
        for i, atom in enumerate(dictionary.atoms):
            if atom.parity != "even":
                continue
            cols.append([evaluate(dictionary, y)[i] for y in ys])
        # 1- or 2-column normal equations
        import numpy as np

        a = np.asarray(cols, dtype=float).T
        tt = np.asarray(t, dtype=float)
        coef, *_ = np.linalg.lstsq(a, tt, rcond=None)
        pred = a @ coef
        return float(np.linalg.norm(pred - tt) / np.linalg.norm(tt))

    d_small = defect(small)
    d_big = defect(big)
    ratio = d_small / max(d_big, 1e-30)
    return {
        "name": "g5_dictionary_capacity",
        "passed": True,
        "small_defect": d_small,
        "enlarged_defect": d_big,
        "reduction_ratio": ratio,
        "note": "synthetic even profile; not CCF_STRETCH_RESIDUAL_GATE",
        "in_ci_all_passed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    ci = [_run_g1(), _run_g2(), _run_g3(), _run_g4()]
    g5 = _run_g5_capacity_record()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.conjugate_hilbert.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(ci)
    payload["g5_capacity"] = g5
    payload["honesty"] = {
        "line_hilbert_only": True,
        "commutation_needs_alpha_gt_0": True,
        "clears_ccf_stretch_gate": False,
        "g5_in_all_passed": False,
    }
    name = "conjugate_hilbert_smoke.json"
    if args.full:
        out_dir = SCRATCH / "conjugate"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "conjugate_hilbert.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        path = write_json(name, payload)
        print(f"wrote {path}")
        # Spec alias
        write_json("conjugate_tower_smoke.json", payload)
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
