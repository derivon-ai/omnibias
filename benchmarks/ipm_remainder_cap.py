# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: IPM remainder CAP leftover (theory 07-15)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.pinn.certified.ipm import (
    IPM_REMAINDER_LEFTOVER,
    enclose_ipm_grid_residual,
    ipm_banded_toy_radii,
)
from omnibias.pinn.jax.discovery.ipm import IPMDiscoveryConfig, run_ipm_discovery
from omnibias.pinn.jax.discovery.pipeline import IPMAdapter


def _run_g1() -> dict[str, Any]:
    raised = False
    try:
        IPMAdapter(n=6, steps=1).discover(seed=0, optimizer="adam")
    except ValueError as exc:
        raised = "Adam" in str(exc)
    out = run_ipm_discovery(IPMDiscoveryConfig(n=6, steps=2))
    ok = raised and out["optimizer"] == "cubic_gn"
    return {"name": "g1_cubic_gn", "passed": ok, "detail": "CubicGN; Adam forbidden"}


def _run_g2() -> dict[str, Any]:
    out = run_ipm_discovery(IPMDiscoveryConfig(n=6, steps=2))
    report = enclose_ipm_grid_residual(out)
    return {
        "name": "g2_enclosure",
        "passed": bool(report["contains_truth_sample"]),
        "detail": "interval hull contains a truth sample",
    }


def _run_g3() -> dict[str, Any]:
    out = run_ipm_discovery(IPMDiscoveryConfig(n=6, steps=2))
    report = enclose_ipm_grid_residual(out)
    return {
        "name": "g3_full_unproved",
        "passed": report["full_ipm_proved"] is False,
        "detail": "full_ipm_proved stays false",
    }


def _run_g4() -> dict[str, Any]:
    leftover = IPM_REMAINDER_LEFTOVER
    toy = ipm_banded_toy_radii()
    ok = (
        leftover["leftover_recorded"] is True
        and leftover["leftover_id"] == 53
        and toy["full_ipm_proved"] is False
    )
    return {
        "name": "g4_leftover_toy",
        "passed": ok,
        "detail": "leftover #53: only ipm_banded_toy_radii closes",
        "leftover_recorded": True,
    }


def _run_g5() -> dict[str, Any]:
    out = run_ipm_discovery(IPMDiscoveryConfig(n=6, steps=2))
    ok = out["honesty"]["navier_stokes_proof_claim"] is False
    return {"name": "g5_honesty", "passed": ok, "detail": "not Navier-Stokes"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "ipm_remainder_cap.json" if full else "ipm_remainder_cap_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for entry in entries:
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.ipm_remainder_cap.v1",
            config={"family": "ipm", "full": full, "leftover": IPM_REMAINDER_LEFTOVER},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    from pathlib import Path

    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "ipm_remainder_cap"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
