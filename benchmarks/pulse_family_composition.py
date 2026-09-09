# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: locked pulse composed into the 07-13 field (theory 07-17)."""

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
    NS_SCALE_EXTERNAL_PREMISES,
)
from omnibias.core.pulse_envelope import locked_pulse_grid_matches_tower
from omnibias.pinn.certified.forced_flat import (
    compose_locked_pulse_family,
    honesty_payload,
)


def _run_g1() -> dict[str, Any]:
    ok = locked_pulse_grid_matches_tower()
    return {
        "name": "g1_tower_grid",
        "passed": ok,
        "detail": "P' matches the sigmoid tower on the locked occupancy grid",
    }


def _run_g2() -> dict[str, Any]:
    report = compose_locked_pulse_family()
    ok = report["composed_axis_zero"] is True and report["product_rule_ok"] is True
    return {
        "name": "g2_composed_zero_product_rule",
        "passed": ok,
        "detail": "corrected P T_0 = (0,0); growth/decay P' T = (±1/2) P T",
    }


def _run_g3() -> dict[str, Any]:
    report = compose_locked_pulse_family()
    ok = report["enclosure_ok"] is True
    return {
        "name": "g3_enclosure",
        "passed": ok,
        "detail": "P T_rtheta enclosure contains the locked grid and a random sample",
    }


def _run_g4() -> dict[str, Any]:
    report = compose_locked_pulse_family()
    flags = report["honesty"]
    ok = (
        report["identity_holds"] is True
        and report["leftover_id"] is None
        and flags["navier_stokes_proof_claim"] is False
        and flags["forced_blowup_reproof_claim"] is False
        and flags["pulses_leftover"] is False
        and flags["joining_leftover"] is True
        and flags["uniqueness_leftover"] is True
        and flags["c_infinity_through_t1_leftover"] is True
        and honesty_payload()["pulses_leftover"] is True
    )
    return {
        "name": "g4_honesty",
        "passed": ok,
        "detail": "parent flags false; 07-17 pulses leftover flipped; cycle leftover named",
    }


def _run_g5() -> dict[str, Any]:
    ok = (
        "construction of each pulse family and the cutoff summation"
        in NS_SCALE_EXTERNAL_PREMISES
    )
    return {
        "name": "g5_scale_premises_untouched",
        "passed": ok,
        "detail": "NS_SCALE_EXTERNAL_PREMISES still names pulse family + cutoff summation",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = (
        "pulse_family_composition.json" if full else "pulse_family_composition_smoke.json"
    )
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    report = compose_locked_pulse_family()
    payload = {
        **provenance(
            schema="omnibias.benchmarks.pulse_family_composition.v1",
            config={
                "family": "pulse_family_composition",
                "full": full,
                "honesty": report["honesty"],
                "leftover_id": report["leftover_id"],
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "pulse_family_composition"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
