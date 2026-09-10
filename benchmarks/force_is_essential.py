# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: force is essential (theory 07-20)."""

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
from omnibias.core.proof.catalog import catalog_entry
from omnibias.core.proof.obligations.convergence_ledger import (
    NS_AB_EXTERNAL_PREMISES,
)
from omnibias.pinn.certified.forced_flat import (
    F0_NOT_A_COROLLARY_LEFTOVER,
    unforced_limit_of_forced_flat,
)


def _run_g1() -> dict[str, Any]:
    report = unforced_limit_of_forced_flat()
    ok = (
        report["uncorrected_axis_T0"] != (0, 0)
        and report["f0_plant"] == "BLOCKED"
        and report["f0_reason"] == "force_is_part_of_the_construction"
    )
    return {
        "name": "g1_f0_blocked",
        "passed": ok,
        "detail": "uncorrected axis jet excludes {0}; f=0 plant is BLOCKED",
    }


def _run_g2() -> dict[str, Any]:
    report = unforced_limit_of_forced_flat()
    ok = report["anisotropy_thins"] is True
    return {
        "name": "g2_anisotropy_thins",
        "passed": ok,
        "detail": "anisotropy ratio strictly decreases as tau decreases",
    }


def _run_g3() -> dict[str, Any]:
    report = unforced_limit_of_forced_flat()
    ok = (
        report["corrected_axis_T0"] == (0, 0)
        and report["uncorrected_axis_T0"] != (0, 0)
    )
    return {
        "name": "g3_force_required_for_zero_jet",
        "passed": ok,
        "detail": "axis jet is {0} only after the 07-13 stress correction",
    }


def _run_g4() -> dict[str, Any]:
    report = unforced_limit_of_forced_flat()
    flags = report["honesty"]
    ok = (
        flags["navier_stokes_proof_claim"] is False
        and flags["forced_blowup_reproof_claim"] is False
        and flags["continuum_navier_stokes_claim"] is False
        and report["leftover_id"] == 58
        and F0_NOT_A_COROLLARY_LEFTOVER["f0_not_a_corollary"] is True
        and "three-dimensional unforced NS, not 2-D Taylor-Green"
        in NS_AB_EXTERNAL_PREMISES
        and len(NS_AB_EXTERNAL_PREMISES) == 5
    )
    return {
        "name": "g4_honesty_leftover_58",
        "passed": ok,
        "detail": "parent flags false; leftover #58 f0_not_a_corollary; A/B premises untouched",
    }


def _run_g5() -> dict[str, Any]:
    import omnibias.pinn.certified.machine  # noqa: F401

    entry = catalog_entry("force_is_essential")
    ok = (
        entry is not None
        and entry.parent == "Navier-Stokes forced blowup (Clay C/D)"
        and entry.parent_status == "already_true"
    )
    return {
        "name": "g5_catalog_cd_already_true",
        "passed": ok,
        "detail": "catalog kind force_is_essential, parent C/D, already_true",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "force_is_essential.json" if full else "force_is_essential_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    report = unforced_limit_of_forced_flat()
    payload = {
        **provenance(
            schema="omnibias.benchmarks.force_is_essential.v1",
            config={
                "family": "force_is_essential",
                "full": full,
                "honesty": report["honesty"],
                "leftover_id": report["leftover_id"],
                "f0_plant": report["f0_plant"],
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "force_is_essential"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
