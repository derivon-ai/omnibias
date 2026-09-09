# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: NS-core axis-regular profile search (theory 07-14)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.pinn.certified.anisotropic import (
    NS_CORE_OPPOSITE,
    NS_CORE_ORIGIN,
    check_ns_core_candidate,
    honesty_payload,
    locked_axis_regular_profile,
    profile_similarity_residual,
)
from omnibias.pinn.jax.discovery.ns_core import run_ns_core_discovery, run_ns_core_search
from omnibias.pinn.jax.discovery.pipeline import NSCoreAdapter, run_singularity_pipeline


def _run_g1() -> dict[str, Any]:
    origin = check_ns_core_candidate(NS_CORE_ORIGIN)
    ok = (
        origin is not None
        and origin.ok
        and profile_similarity_residual(locked_axis_regular_profile()) == 0
    )
    return {"name": "g1_locked_origin", "passed": ok, "detail": "origin still discharges"}


def _run_g2() -> dict[str, Any]:
    opposite = check_ns_core_candidate(NS_CORE_OPPOSITE)
    ok = (
        opposite is not None
        and opposite.ok is False
        and opposite.payload["cone_reason"] == "opposite_cone"
    )
    return {
        "name": "g2_cone_opposite",
        "passed": ok,
        "detail": "opposite-cone coefficient is named BLOCKED",
    }


def _run_g3() -> dict[str, Any]:
    result = run_ns_core_discovery(budget=0)
    ok = result.search_incomplete and result.status == "BLOCKED"
    return {
        "name": "g3_budget_zero",
        "passed": ok,
        "detail": "budget==0 is search_incomplete",
    }


def _run_g4() -> dict[str, Any]:
    payload = run_ns_core_search(budget=27)
    ok = payload["origin_only"] is False and bool(payload["non_origin_witnesses"])
    return {
        "name": "g4_second_witness",
        "passed": ok,
        "detail": "non-origin witness in the finite box",
        "non_origin_witnesses": payload["non_origin_witnesses"],
    }


def _run_g5() -> dict[str, Any]:
    flags = honesty_payload()
    out = run_singularity_pipeline(NSCoreAdapter(budget=8))
    ok = (
        flags["navier_stokes_proof_claim"] is False
        and flags["forced_blowup_reproof_claim"] is False
        and out.certificate["honesty"]["navier_stokes_proof_claim"] is False
        and out.discovery["status"] == "PROVED"
    )
    return {
        "name": "g5_pipeline_honesty",
        "passed": ok,
        "detail": "adapter + parent flags stay false",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "ns_core_search.json" if full else "ns_core_search_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for entry in entries:
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.ns_core_search.v1",
            config={"family": "ns_core_profile_search", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = __import__("pathlib").Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "ns_core_search"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
