# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: 3-D Taylor-Green IC is not closed form (theory 07-22)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof.obligations.convergence_ledger import (
    NS_AB_EXTERNAL_PREMISES,
    check_ledger,
    navier_stokes_ab_architecture_ledger,
)
from omnibias.pinn.certified.unforced import (
    THREE_D_AB_PREMISE,
    THREE_D_TG_NOT_EXACT_LEFTOVER,
    UNFORCED_CONTINUATION_LEFTOVER,
    force_free_tg3d_ic,
)


def _run_g1() -> dict[str, Any]:
    report = force_free_tg3d_ic()
    ok = (
        report["force_zero"] is True
        and report["plant_unforced"] is True
        and report["dimension"] == 3
        and report["exact_solution"] is False
    )
    return {
        "name": "g1_tg3d_force_free_not_exact",
        "passed": ok,
        "detail": "3-D TG fixture is force-free and exact_solution false",
    }


def _run_g2() -> dict[str, Any]:
    report = force_free_tg3d_ic()
    ok = report["omega0_contains_grid_and_sample"] is True
    return {
        "name": "g2_tg3d_omega0_enclosure",
        "passed": ok,
        "detail": "t=0 vorticity hull contains a grid and a sample of |omega|",
    }


def _run_g3() -> dict[str, Any]:
    report = force_free_tg3d_ic()
    ok = (
        report["halt_reason"] == "BLOCKED"
        and report["halt_detail"] == "three_d_tg_not_closed_form"
    )
    return {
        "name": "g3_tg3d_continuation_halt",
        "passed": ok,
        "detail": "07-19 budget Halt/BLOCKED/three_d_tg_not_closed_form",
    }


def _run_g4() -> dict[str, Any]:
    report = force_free_tg3d_ic()
    flags = report["honesty"]
    ok = (
        flags["navier_stokes_proof_claim"] is False
        and flags["three_d_claim"] is False
        and flags["three_d_tg_claim"] is False
        and report["leftover_id"] == 59
        and THREE_D_TG_NOT_EXACT_LEFTOVER["three_d_tg_not_exact"] is True
        and report["leftover_57_untouched"] is True
        and UNFORCED_CONTINUATION_LEFTOVER["leftover_id"] == 57
    )
    return {
        "name": "g4_honesty_leftover_59",
        "passed": ok,
        "detail": "parent flags false; leftover #59; leftover #57 untouched",
    }


def _run_g5() -> dict[str, Any]:
    ledger = navier_stokes_ab_architecture_ledger()
    checked = check_ledger(ledger)
    report = force_free_tg3d_ic()
    ok = (
        checked.strength == "CONDITIONAL"
        and checked.holds
        and THREE_D_AB_PREMISE in NS_AB_EXTERNAL_PREMISES
        and report["three_d_premise_present"] is True
    )
    return {
        "name": "g5_ab_ledger_3d_premise",
        "passed": ok,
        "detail": "catalog parent A/B open; NS_AB 3-D premise stays",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "unforced_tg3d_ic.json" if full else "unforced_tg3d_ic_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    report = force_free_tg3d_ic()
    payload = {
        **provenance(
            schema="omnibias.benchmarks.unforced_tg3d_ic.v1",
            config={
                "family": "unforced_tg3d_ic",
                "full": full,
                "honesty": report["honesty"],
                "leftover_id": report["leftover_id"],
                "dimension": report["dimension"],
                "exact_solution": report["exact_solution"],
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "unforced_tg3d_ic"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
