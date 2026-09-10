# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: force-free BKM slab (theory 07-18)."""

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
from omnibias.pinn.certified.unforced import force_free_bkm_slab, honesty_payload


def _run_g1() -> dict[str, Any]:
    report = force_free_bkm_slab()
    ok = report["force_zero"] is True and report["plant_unforced"] is True
    return {
        "name": "g1_force_free",
        "passed": ok,
        "detail": "Taylor-Green fixture force is the zero array; plant is unforced",
    }


def _run_g2() -> dict[str, Any]:
    report = force_free_bkm_slab()
    ok = report["weak_covers"] is True and report["weak_misses"] == 0
    return {
        "name": "g2_weak_covers",
        "passed": ok,
        "detail": "07-02 weak-form certificate covers the box",
    }


def _run_g3() -> dict[str, Any]:
    report = force_free_bkm_slab()
    ok = (
        report["bkm_bounded"] is True
        and report["bkm_contains_exact"] is True
        and report["integrand_grid_and_sample"] is True
    )
    return {
        "name": "g3_bkm_enclosure",
        "passed": ok,
        "detail": "BKM integral enclosure is bounded, contains exact + integrand samples",
    }


def _run_g4() -> dict[str, Any]:
    flags = honesty_payload()
    ok = (
        flags["navier_stokes_proof_claim"] is False
        and flags["continuum_navier_stokes_claim"] is False
        and flags["forced_blowup_reproof_claim"] is False
        and flags["three_d_claim"] is False
        and flags["infinite_time_leftover"] is True
        and flags["all_data_leftover"] is True
        and flags["bridge_theorem_leftover"] is True
        and flags["unforced_majorant_leftover"] is True
    )
    return {
        "name": "g4_honesty",
        "passed": ok,
        "detail": "parent flags false; A/B leftovers named",
    }


def _run_g5() -> dict[str, Any]:
    ledger = navier_stokes_ab_architecture_ledger()
    report = check_ledger(ledger)
    ok = (
        report.strength == "CONDITIONAL"
        and report.holds
        and bool(NS_AB_EXTERNAL_PREMISES)
        and bool(ledger.external_premises)
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
    artifact = "unforced_bkm_slab.json" if full else "unforced_bkm_slab_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    slab = force_free_bkm_slab()
    payload = {
        **provenance(
            schema="omnibias.benchmarks.unforced_bkm_slab.v1",
            config={
                "family": "unforced_bkm_slab",
                "full": full,
                "honesty": slab["honesty"],
                "bkm_integral": slab["bkm_integral"],
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "unforced_bkm_slab"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
