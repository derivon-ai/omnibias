# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: force-free 3-D ABC BKM slab (theory 07-21)."""

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
    UNFORCED_CONTINUATION_LEFTOVER,
    force_free_abc_bkm_slab,
    locked_two_abc_slab_continuation,
)


def _run_g1() -> dict[str, Any]:
    report = force_free_abc_bkm_slab()
    ok = (
        report["force_zero"] is True
        and report["plant_unforced"] is True
        and report["dimension"] == 3
    )
    return {
        "name": "g1_abc_force_free_3d",
        "passed": ok,
        "detail": "ABC fixture is 3-D, forced False, force array zero",
    }


def _run_g2() -> dict[str, Any]:
    report = force_free_abc_bkm_slab()
    ok = (
        report["bkm_bounded"] is True
        and report["bkm_contains_exact"] is True
        and report["integrand_grid_and_sample"] is True
    )
    return {
        "name": "g2_abc_bkm_enclosure",
        "passed": ok,
        "detail": "BKM integral enclosure is bounded and contains integrand samples",
    }


def _run_g3() -> dict[str, Any]:
    chain = locked_two_abc_slab_continuation()
    ok = (
        chain["accepted"] is True
        and chain["n_slabs"] == 2
        and chain["growing_reason"] == "BLOCKED"
        and chain["growing_detail"] == "growing_vorticity"
        and chain["empty_budget_reason"] == "search_incomplete"
    )
    return {
        "name": "g3_abc_continuation",
        "passed": ok,
        "detail": "two decaying ABC slabs accept; manufactured growth Halt/BLOCKED",
    }


def _run_g4() -> dict[str, Any]:
    report = force_free_abc_bkm_slab()
    chain = locked_two_abc_slab_continuation()
    flags = report["honesty"]
    ok = (
        flags["three_d_claim"] is False
        and flags["exact_3d_abc_plant"] is True
        and flags["navier_stokes_proof_claim"] is False
        and flags["infinite_time_leftover"] is True
        and flags["all_data_leftover"] is True
        and flags["bridge_theorem_leftover"] is True
        and flags["unforced_majorant_leftover"] is True
        and report["leftover_id"] == 57
        and chain["leftover_id"] == 57
        and UNFORCED_CONTINUATION_LEFTOVER["leftover_id"] == 57
    )
    return {
        "name": "g4_honesty_leftover_57",
        "passed": ok,
        "detail": "three_d_claim false; leftover #57 reused; exact_3d_abc_plant true",
    }


def _run_g5() -> dict[str, Any]:
    ledger = navier_stokes_ab_architecture_ledger()
    checked = check_ledger(ledger)
    ok = (
        checked.strength == "CONDITIONAL"
        and checked.holds
        and THREE_D_AB_PREMISE in NS_AB_EXTERNAL_PREMISES
        and THREE_D_AB_PREMISE in ledger.external_premises
    )
    return {
        "name": "g5_ab_ledger_3d_premise",
        "passed": ok,
        "detail": "A/B ledger CONDITIONAL; 3-D Taylor-Green premise stays",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "unforced_abc_slab.json" if full else "unforced_abc_slab_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    report = force_free_abc_bkm_slab()
    chain = locked_two_abc_slab_continuation()
    payload = {
        **provenance(
            schema="omnibias.benchmarks.unforced_abc_slab.v1",
            config={
                "family": "unforced_abc_slab",
                "full": full,
                "honesty": report["honesty"],
                "leftover_id": chain["leftover_id"],
                "n_slabs": chain["n_slabs"],
                "dimension": report["dimension"],
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "unforced_abc_slab"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
