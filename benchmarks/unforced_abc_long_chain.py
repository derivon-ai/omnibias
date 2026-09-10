# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: four-slab ABC continuation (theory 07-23)."""

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
    INFINITE_TIME_PREMISE,
    UNFORCED_CONTINUATION_LEFTOVER,
    locked_n_abc_slab_continuation,
)


def _run_g1() -> dict[str, Any]:
    chain = locked_n_abc_slab_continuation(n_slabs=4)
    ok = (
        chain["accepted"] is True
        and chain["n_slabs"] == 4
        and chain["cover_end"] == 2.0
        and chain["covers_infinite_time"] is False
    )
    return {
        "name": "g1_four_abc_slabs",
        "passed": ok,
        "detail": "four decaying ABC slabs accept; cover end is 2, not infinity",
    }


def _run_g2() -> dict[str, Any]:
    chain = locked_n_abc_slab_continuation(n_slabs=4)
    ok = (
        chain["growing_reason"] == "BLOCKED"
        and chain["growing_detail"] == "growing_vorticity"
    )
    return {
        "name": "g2_growing_blocked",
        "passed": ok,
        "detail": "manufactured growth Halt/BLOCKED/growing_vorticity",
    }


def _run_g3() -> dict[str, Any]:
    chain = locked_n_abc_slab_continuation(n_slabs=4)
    ok = chain["empty_budget_reason"] == "search_incomplete"
    return {
        "name": "g3_empty_budget",
        "passed": ok,
        "detail": "empty remaining budget is search_incomplete",
    }


def _run_g4() -> dict[str, Any]:
    chain = locked_n_abc_slab_continuation(n_slabs=4)
    flags = chain["honesty"]
    ok = (
        chain["covers_infinite_time"] is False
        and chain["leftover_id"] == 57
        and UNFORCED_CONTINUATION_LEFTOVER["leftover_id"] == 57
        and flags["three_d_claim"] is False
        and flags["navier_stokes_proof_claim"] is False
    )
    return {
        "name": "g4_honesty_leftover_57",
        "passed": ok,
        "detail": "leftover #57 reused; three_d_claim false; cover is finite",
    }


def _run_g5() -> dict[str, Any]:
    ledger = navier_stokes_ab_architecture_ledger()
    checked = check_ledger(ledger)
    chain = locked_n_abc_slab_continuation(n_slabs=4)
    ok = (
        checked.strength == "CONDITIONAL"
        and checked.holds
        and INFINITE_TIME_PREMISE in NS_AB_EXTERNAL_PREMISES
        and INFINITE_TIME_PREMISE in ledger.external_premises
        and chain["infinite_time_premise_present"] is True
    )
    return {
        "name": "g5_ab_ledger_infinite_time_premise",
        "passed": ok,
        "detail": "A/B ledger CONDITIONAL; infinite-time premise stays",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = (
        "unforced_abc_long_chain.json" if full else "unforced_abc_long_chain_smoke.json"
    )
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    chain = locked_n_abc_slab_continuation(n_slabs=4)
    payload = {
        **provenance(
            schema="omnibias.benchmarks.unforced_abc_long_chain.v1",
            config={
                "family": "unforced_abc_long_chain",
                "full": full,
                "honesty": chain["honesty"],
                "leftover_id": chain["leftover_id"],
                "n_slabs": chain["n_slabs"],
                "cover_end": chain["cover_end"],
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "unforced_abc_long_chain"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
