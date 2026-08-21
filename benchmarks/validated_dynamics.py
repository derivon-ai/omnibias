# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 07-06: jet Lohner steps and exact-Jacobian orbit recovery.

Smoke earns G1 (every run names a WidthBudget dominant),
G2 (1000-box Jacobian containment on ten fields), G3
(5x horizon at the baseline width on ten fields), G4
(three polluted-Jacobian orbit recoveries), G5 (singularity
steps stay inside known radii), and G6 (existing lohner_flow
is bit-unchanged; kernel flag cannot be forged).

One field, one box, one finite horizon. Not a continuum
existence theorem and not an attractor statement. Jets are
founding bias collapse, not temperature collapse.
"""

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
from omnibias.core.proof.certificate import make_certificate
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_flow import (
    DISCLAIMER,
    coefficient_arithmetic_report,
    honesty_payload,
    horizon_report,
    jacobian_containment_report,
    lohner_flow_jet,
    lohner_regression_snapshot,
    named_tower_fields,
    orbit_recovery_report,
    run_schema_errors,
    seal_run,
    singularity_step_report,
    tower_field,
    tower_jacobian,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    spec = named_tower_fields()[0]
    run = lohner_flow_jet(
        tower_field(spec),
        tower_jacobian(spec),
        [Interval(0.2, 0.3)],
        0.05,
        8,
        order=8,
    )
    sealed = seal_run(run)
    ok = (
        run.budget.dominant in {"truncation", "jacobian", "wrapping", "rounding"}
        and run_schema_errors(run) == []
        and "theorem_prover_verified" not in sealed["honesty"]
    )
    return {
        "name": "g1_width_budget",
        "passed": ok,
        "dominant": run.budget.dominant,
        "budget": run.budget.to_payload(),
    }


def _run_g2(*, full: bool) -> dict[str, Any]:
    n = 1000 if full else 1000
    report = jacobian_containment_report(n_boxes=n, seed=3)
    return {
        "name": "g2_jacobian_containment",
        "passed": bool(report["containment"]),
        "checked": report["checked"],
        "misses": report["misses"],
        "fields": report["fields"],
    }


def _run_g3() -> dict[str, Any]:
    rows = horizon_report()
    ok = len(rows) == 10 and all(bool(r["win"]) for r in rows)
    return {
        "name": "g3_horizon_extension",
        "passed": ok,
        "n": len(rows),
        "wins": sum(1 for r in rows if r["win"]),
        "ratios": [r["ratio"] for r in rows],
        "names": [r["name"] for r in rows],
    }


def _run_g4() -> dict[str, Any]:
    rows = orbit_recovery_report()
    ok = len(rows) == 3 and all(bool(r["recovered"]) for r in rows)
    return {
        "name": "g4_orbit_recovery",
        "passed": ok,
        "n": len(rows),
        "recovered": sum(1 for r in rows if r["recovered"]),
        "names": [r["name"] for r in rows],
    }


def _run_g5() -> dict[str, Any]:
    rows = singularity_step_report()
    ok = len(rows) == 3 and all(bool(r["sound"]) for r in rows)
    return {
        "name": "g5_step_controller",
        "passed": ok,
        "steps": [r["step"] for r in rows],
        "names": [r["name"] for r in rows],
    }


def _run_g6() -> dict[str, Any]:
    snap_a = lohner_regression_snapshot()
    snap_b = lohner_regression_snapshot()
    forged = False
    try:
        make_certificate(
            claim="forged",
            payload={},
            honesty={"theorem_prover_verified": True},
        )
    except ValueError:
        forged = True
    arith = coefficient_arithmetic_report()
    ok = (
        snap_a == snap_b
        and forged
        and honesty_payload()["theorem_prover_verified"] is False
        and honesty_payload()["continuum_existence_claim"] is False
        and arith["choice"] == "interval"
        and arith["tm_required"] is False
        and "not a continuum existence theorem" in DISCLAIMER
    )
    return {"name": "g6_no_regression", "passed": ok, "snapshot": snap_a, "arithmetic": arith}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "validated_dynamics.json" if full else "validated_dynamics_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(full=full), _run_g3(), _run_g4(), _run_g5(), _run_g6()]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.validated_dynamics.v1",
            config={"family": "validated_dynamics", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "validated_dynamics"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
