# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: named linearizing transforms (theory 02-13).

G1 is the Cole-Hopf factorial-jet identity. G3 Burgers win is
leftover-recorded (leftover #39): no Cole-Hopf-trained field versus a
direct PINN. Previous ``g3_burgers_init`` ``passed=True`` stub withdrawn.
G6 is torch/jax bit-identity on the Cole-Hopf field. G2 n-soliton
generation is leftover-recorded (leftover #51): no n=1,2,3 grid
plus phase-shift check is wired. G4 permutability is leftover-recorded
(leftover #52): no two-soliton versus sequential Bäcklund check
is wired. Not a Navier-Stokes claim. Spec 03-11 search is not claimed.
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


def _run_g2() -> dict[str, Any]:
    return {
        "name": "g2_exact_soliton",
        "passed": False,
        "earned": False,
        "reported": True,
        "leftover_recorded": True,
        "leftover_id": 51,
        "leftover_tick": 93,
        "stays_full": True,
        "in_ci_all_passed": False,
        "n_soliton_api": False,
        "need": (
            "Generated n-soliton solutions for n=1,2,3 satisfy the PDE "
            "to <= 1e-13 relative on a dense space-time grid, and their "
            "asymptotic phase shifts match the published formulas"
        ),
        "note": (
            "Leftover #51 leftover-recorded: named G2 is n-soliton "
            "generation plus published phase shifts. permutability is "
            "a scalar Bäcklund formula, not that grid check. Not in "
            "CI all_passed."
        ),
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


def _run_g4() -> dict[str, Any]:
    return {
        "name": "g4_permutability",
        "passed": False,
        "earned": False,
        "reported": True,
        "leftover_recorded": True,
        "leftover_id": 52,
        "leftover_tick": 94,
        "stays_full": True,
        "in_ci_all_passed": False,
        "sequential_backlund_api": False,
        "need": (
            "The two-soliton produced by the permutability formula "
            "agrees with the two-soliton produced by two sequential "
            "Bäcklund integrations to <= 1e-12"
        ),
        "note": (
            "Leftover #52 leftover-recorded: named G4 is permutability "
            "versus sequential Bäcklund. permutability() is a scalar "
            "formula with no sequential integration. Not in CI "
            "all_passed."
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


def _run_g6() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.pinn.transform.jax import cole_hopf_apply
    from omnibias.pinn.transform.torch import ColeHopfField

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    x = torch.tensor([0.2, -0.1], dtype=torch.float64)
    t = torch.tensor([0.0, 0.3], dtype=torch.float64)
    u_t = ColeHopfField(nu=1.0, dtype=torch.float64)(x, t)
    u_j = cole_hopf_apply(jnp.asarray(x.numpy()), jnp.asarray(t.numpy()), nu=1.0, k=1.0)
    t_list = u_t.detach().cpu().tolist()
    j_list = [float(v) for v in u_j.tolist()]
    passed = t_list == j_list
    return {
        "name": "g6_parity",
        "passed": passed,
        "earned": passed,
        "reported": True,
        "in_ci_all_passed": passed,
        "torch": t_list,
        "jax": j_list,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2()
    g3 = _run_g3()
    g4 = _run_g4()
    g5 = _run_g5()
    g6 = _run_g6()
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
        {
            "name": "g6_parity",
            "passed": bool(g6["passed"]),
            "in_ci_all_passed": bool(g6["in_ci_all_passed"]),
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.linearizing_transforms.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "g2_in_all_passed": False,
            "g3_in_all_passed": False,
            "g4_in_all_passed": False,
            "g6_in_all_passed": bool(g6["in_ci_all_passed"]),
            "gates_in_scope": ["g1", "g5", "g6"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["g1"] = g1
    payload["g2"] = g2
    payload["g3"] = g3
    payload["g4"] = g4
    payload["g5"] = g5
    payload["g6"] = g6
    payload["honesty"] = {
        "named_only": True,
        "search_claimed": False,
        "exactness": "to jet truncation order N",
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "navier_stokes_proof_claim": False,
        "g2_exact_soliton_earned": False,
        "g2_reported": True,
        "g2_leftover_recorded": True,
        "g2_leftover_id": 51,
        "g2_leftover_tick": 93,
        "g2_stays_full": True,
        "g2_in_ci_all_passed": False,
        "g2_n_soliton_api": False,
        "g3_burgers_win_earned": False,
        "g3_reported": True,
        "g3_leftover_recorded": True,
        "g3_leftover_id": 39,
        "g3_leftover_tick": 81,
        "g3_stays_full": True,
        "g3_in_ci_all_passed": False,
        "g3_training_loop": False,
        "g4_permutability_earned": False,
        "g4_reported": True,
        "g4_leftover_recorded": True,
        "g4_leftover_id": 52,
        "g4_leftover_tick": 94,
        "g4_stays_full": True,
        "g4_in_ci_all_passed": False,
        "g4_sequential_backlund_api": False,
        "g6_earned": bool(g6["earned"]),
        "g6_reported": True,
        "g6_in_ci_all_passed": bool(g6["in_ci_all_passed"]),
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
