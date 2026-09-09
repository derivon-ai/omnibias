# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: Boussinesq remainder CAP leftover (theory 07-16)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.pinn.certified.boussinesq import (
    BOUSSINESQ_REMAINDER_LEFTOVER,
    enclose_boussinesq_grid_residual,
)
from omnibias.pinn.jax.discovery.boussinesq import (
    BoussinesqDiscoveryConfig,
    run_boussinesq_discovery,
)
from omnibias.pinn.jax.discovery.pipeline import BoussinesqAdapter


def _run_g1() -> dict[str, Any]:
    raised = False
    try:
        BoussinesqAdapter(n=6, steps=1).discover(seed=0, optimizer="adam")
    except ValueError as exc:
        raised = "Adam" in str(exc)
    out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
    return {
        "name": "g1_cubic_gn",
        "passed": raised and out["optimizer"] == "cubic_gn",
        "detail": "CubicGN; Adam forbidden",
    }


def _run_g2() -> dict[str, Any]:
    out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
    report = enclose_boussinesq_grid_residual(out)
    return {
        "name": "g2_enclosure",
        "passed": bool(report["contains_truth_sample"]),
        "detail": "interval hull contains a truth sample",
    }


def _run_g3() -> dict[str, Any]:
    out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
    report = enclose_boussinesq_grid_residual(out)
    return {
        "name": "g3_full_unproved",
        "passed": report["full_boussinesq_proved"] is False,
        "detail": "full_boussinesq_proved stays false",
    }


def _run_g4() -> dict[str, Any]:
    leftover = BOUSSINESQ_REMAINDER_LEFTOVER
    out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
    ok = (
        leftover["leftover_id"] == 54
        and leftover["leftover_recorded"] is True
        and out["lambda_n_hypothesis"]["status"] == "empirical_hypothesis_not_theorem"
    )
    return {
        "name": "g4_leftover_lambda",
        "passed": ok,
        "detail": "leftover #54; lambda_n stays a hypothesis",
        "leftover_recorded": True,
    }


def _run_g5() -> dict[str, Any]:
    out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
    ok = out["honesty"]["navier_stokes_proof_claim"] is False
    return {"name": "g5_honesty", "passed": ok, "detail": "not Navier-Stokes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = (
        "boussinesq_remainder_cap.json" if full else "boussinesq_remainder_cap_smoke.json"
    )
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for entry in entries:
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.boussinesq_remainder_cap.v1",
            config={
                "family": "boussinesq",
                "full": full,
                "leftover": BOUSSINESQ_REMAINDER_LEFTOVER,
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    from pathlib import Path

    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "boussinesq_remainder_cap"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
