# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: named linearizing transforms (theory 02-13).

G1 is the Cole-Hopf factorial-jet identity. G3 Burgers win is
leftover-recorded (leftover #39): no Cole-Hopf-trained field versus a
direct PINN. Previous ``g3_burgers_init`` ``passed=True`` stub withdrawn.
Not a Navier-Stokes claim. Spec 03-11 search is not claimed.
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
    from omnibias.core.transforms_pde import (
        cole_hopf_from_heat_phi,
        named_cole_hopf,
        verify_cole_hopf_burgers_jet,
        verify_transform,
    )

    t = named_cole_hopf()
    jet = verify_cole_hopf_burgers_jet(nu=0.1, k=-0.5, order=8)
    passed = bool(
        verify_transform(t)
        and abs(cole_hopf_from_heat_phi(0.0, 0.0) + 2.0) <= 1e-15
        and jet["passed"]
    )
    return {
        "name": "g1_cole_hopf",
        "passed": passed,
        "in_ci_all_passed": passed,
        "jet_passed": bool(jet["passed"]),
        "jet_err0": float(jet["err0"]),
        "navier_stokes_proof_claim": False,
    }


def _run_g3() -> dict[str, Any]:
    return {
        "name": "g3_burgers_win",
        "passed": False,
        "earned": False,
        "reported": True,
        "leftover_recorded": True,
        "leftover_id": 39,
        "leftover_tick": 81,
        "stays_full": True,
        "in_ci_all_passed": False,
        "training_loop": False,
        "need": (
            "Cole-Hopf-trained field rel L2 <= 1e-8 on shock Burgers, "
            "100x vs direct PINN at matched cost, five seeds"
        ),
        "note": (
            "Leftover #39 leftover-recorded: named G3 is a Cole-Hopf "
            "train versus a direct PINN. No training loop is wired. "
            "cole_hopf_jet is the named-map algebra, not that bake-off. "
            "Previous g3_burgers_init passed=True / smoke stub withdrawn. "
            "Not in CI all_passed."
        ),
    }


def _run_g5() -> dict[str, Any]:
    from omnibias.core.transforms_pde import cole_hopf_jet, heat_plane_wave_jets

    phi, phi_x = heat_plane_wave_jets(k=-0.5, nu=0.1, order=8)
    good = cole_hopf_jet(phi, phi_x, nu=0.1)
    bad = cole_hopf_jet(phi, phi_x, nu=0.2)
    vacuous = abs(good[0] - bad[0]) <= 1e-12
    return {
        "name": "g5_negative_control",
        "passed": not vacuous,
        "in_ci_all_passed": not vacuous,
        "note": "wrong nu shifts u0; the jet checker is not vacuous",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g3 = _run_g3()
    g5 = _run_g5()
    entries: list[dict[str, Any]] = [
        {
            "name": "g1_cole_hopf",
            "passed": bool(g1["passed"]),
            "in_ci_all_passed": bool(g1["in_ci_all_passed"]),
        },
        {
            "name": "g5_negative_control",
            "passed": bool(g5["passed"]),
            "in_ci_all_passed": bool(g5["in_ci_all_passed"]),
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.linearizing_transforms.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "g3_in_all_passed": False,
            "gates_in_scope": ["g1", "g5"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["g1"] = g1
    payload["g3"] = g3
    payload["g5"] = g5
    payload["honesty"] = {
        "named_only": True,
        "search_claimed": False,
        "exactness": "to jet truncation order N",
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "navier_stokes_proof_claim": False,
        "g3_burgers_win_earned": False,
        "g3_reported": True,
        "g3_leftover_recorded": True,
        "g3_leftover_id": 39,
        "g3_leftover_tick": 81,
        "g3_stays_full": True,
        "g3_in_ci_all_passed": False,
        "g3_training_loop": False,
    }
    if args.full:
        dest = SCRATCH / "transforms" / "linearizing_transforms.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('linearizing_transforms_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
