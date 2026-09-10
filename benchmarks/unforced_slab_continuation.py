# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: force-free slab continuation (theory 07-19)."""

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
    UNFORCED_CONTINUATION_LEFTOVER,
    locked_two_slab_continuation,
)


def _run_g1() -> dict[str, Any]:
    chain = locked_two_slab_continuation()
    ok = chain["accepted"] is True and chain["n_slabs"] == 2
    return {
        "name": "g1_two_slabs_accept",
        "passed": ok,
        "detail": "locked decaying TG slabs [0, 1/2] then [1/2, 1] accept",
    }


def _run_g2() -> dict[str, Any]:
    chain = locked_two_slab_continuation()
    ok = (
        chain["growing_reason"] == "BLOCKED"
        and chain["growing_detail"] == "growing_vorticity"
    )
    return {
        "name": "g2_growing_blocked",
        "passed": ok,
        "detail": "manufactured growing vorticity returns Halt/BLOCKED",
    }


def _run_g3() -> dict[str, Any]:
    chain = locked_two_slab_continuation()
    ok = (
        chain["empty_budget_reason"] == "search_incomplete"
        and chain["honesty"]["navier_stokes_proof_claim"] is False
    )
    return {
        "name": "g3_empty_budget",
        "passed": ok,
        "detail": "empty remaining budget is search_incomplete; parent flag false",
    }


def _run_g4() -> dict[str, Any]:
    chain = locked_two_slab_continuation()
    leftover = UNFORCED_CONTINUATION_LEFTOVER
    ok = (
        leftover["leftover_id"] == 57
        and chain["leftover_id"] == 57
        and chain["covers_infinite_time"] is False
        and leftover["all_data"] is False
        and leftover["three_d"] is False
        and leftover["bridge_theorem"] is False
    )
    return {
        "name": "g4_leftover_57",
        "passed": ok,
        "detail": "leftover #57 records infinite-time / all-data / 3-D / bridge",
    }


def _run_g5() -> dict[str, Any]:
    ledger = navier_stokes_ab_architecture_ledger()
    report = check_ledger(ledger)
    ok = (
        report.strength == "CONDITIONAL"
        and report.holds
        and bool(NS_AB_EXTERNAL_PREMISES)
    )
    return {
        "name": "g5_ab_ledger_conditional",
        "passed": ok,
        "detail": "navier_stokes_ab_architecture_ledger stays CONDITIONAL with nonempty premises",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = (
        "unforced_slab_continuation.json"
        if full
        else "unforced_slab_continuation_smoke.json"
    )
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    chain = locked_two_slab_continuation()
    payload = {
        **provenance(
            schema="omnibias.benchmarks.unforced_slab_continuation.v1",
            config={
                "family": "unforced_slab_continuation",
                "full": full,
                "honesty": chain["honesty"],
                "leftover_id": chain["leftover_id"],
                "n_slabs": chain["n_slabs"],
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "unforced_slab_continuation"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
