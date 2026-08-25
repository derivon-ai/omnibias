# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated primitive: conjugate Hilbert tower (theory 01-12 G1–G4; G5 not CI).

Line Hilbert only. Commutation needs ``alpha > 0``. G5 (dictionary capacity
on the CCF profile-fitting subproblem) is a campaign artifact and is
**not** in ``all_passed``; it does not clear ``CCF_STRETCH_RESIDUAL_GATE``.
"""

from __future__ import annotations

import argparse
import json
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


def _run_g5_synthetic_in_span() -> dict[str, Any]:
    """Sanity: an in-span target is recovered. This is not named G5."""
    import math as _math

    from omnibias.core.conjugate import HardyAtom, HardyDictionary, evaluate

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
        cols = []
        t = [target(y) for y in ys]
        for i, atom in enumerate(dictionary.atoms):
            if atom.parity != "even":
                continue
            cols.append([evaluate(dictionary, y)[i] for y in ys])
        import numpy as np

        a = np.asarray(cols, dtype=float).T
        tt = np.asarray(t, dtype=float)
        coef, *_ = np.linalg.lstsq(a, tt, rcond=None)
        pred = a @ coef
        return float(np.linalg.norm(pred - tt) / np.linalg.norm(tt))

    d_small = defect(small)
    d_big = defect(big)
    return {
        "name": "g5_synthetic_in_span",
        "passed": d_big <= 1e-12,
        "small_defect": d_small,
        "enlarged_defect": d_big,
        "note": "in-span sanity, not the CCF G5 gate",
        "in_ci_all_passed": False,
    }


def _run_g5_capacity_record() -> dict[str, Any]:
    """Named G5: CCF profile-fitting subproblem from the campaign smoke."""
    repo = Path(__file__).resolve().parents[1]
    path = repo / "docs" / "benchmarks" / "ccf_conjugate_sweep_smoke.json"
    sweep = json.loads(path.read_text(encoding="utf-8"))
    meas = sweep["g2_measurement"]
    ratio = float(meas["matched_width_ratio_vs_n0"])
    ten_x = bool(meas["ten_x"])
    n0 = float(sweep["rows"][0]["dense_max_abs"])
    matched = float(sweep["matched_width"]["dense_max_abs"])
    return {
        "name": "g5_dictionary_capacity",
        "passed": False,
        "earned": False,
        "source": "docs/benchmarks/ccf_conjugate_sweep_smoke.json",
        "n0_dense_max_abs": n0,
        "matched_width_dense_max_abs": matched,
        "matched_width_ratio_vs_n0": ratio,
        "ten_x_required": 10.0,
        "ten_x": ten_x,
        "matched_width_gram_cond": float(sweep["matched_width"]["gram_cond"]),
        "n1_unmatched_gram_cond": float(sweep["rows"][1]["gram_cond"]),
        "best_max_order": int(sweep["best"]["max_order"]),
        "note": (
            "CCF profile-fitting subproblem. Enlargement does not cut "
            "dense residual 10x at matched atom count; N=0 is best. "
            "Catch-22 is not dictionary order on this grid. "
            "Not CCF_STRETCH_RESIDUAL_GATE."
        ),
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
    payload["g5_synthetic_in_span"] = _run_g5_synthetic_in_span()
    payload["honesty"] = {
        "line_hilbert_only": True,
        "commutation_needs_alpha_gt_0": True,
        "clears_ccf_stretch_gate": False,
        "g5_in_all_passed": False,
        "g5_earned": False,
        "g5_is_ccf_profile_not_synthetic": True,
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
