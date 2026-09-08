# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: admissible stress cone (theory 07-10)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof import generate_obligation
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.obligations.convergence_ledger import (
    check_ledger,
    navier_stokes_scale_ledger,
)
from omnibias.core.proof.obligations.stress_cone import (
    check_cone,
    honesty_payload,
    locked_interior_cone,
    opposite_cone_query,
    parallel_generators_cone,
    replay_cone_certificate,
    seal_cone_certificate,
)


def _run_g1() -> dict[str, Any]:
    report = check_cone(locked_interior_cone())
    ok = report.holds and report.lambda1 == Fraction(1) and report.lambda2 == Fraction(1)
    return {"name": "g1_locked_cone", "passed": ok, "detail": "exact lambda_i = 1"}


def _run_g2() -> dict[str, Any]:
    parallel = check_cone(parallel_generators_cone())
    opposite = check_cone(opposite_cone_query())
    ok = (
        parallel.reason == "degenerate_generators"
        and opposite.reason == "opposite_cone"
    )
    return {"name": "g2_blocked_named", "passed": ok, "detail": "parallel / opposite named"}


def _run_g3() -> dict[str, Any]:
    report = check_ledger(navier_stokes_scale_ledger())
    tipping = report.binding_threshold
    ok = (
        report.holds
        and tipping is not None
        and tipping[0] == "h"
        and tipping[1] == "lt"
        and tipping[2] == Fraction(1, 100)
    )
    return {
        "name": "g3_scale_ledger",
        "passed": ok,
        "detail": "scale ledger discharges; binding h < 1/100",
    }


def _run_g4() -> dict[str, Any]:
    sealed = seal_cone_certificate(locked_interior_cone(), run_lean=False)
    tampered = dict(sealed.certificate)
    payload = dict(tampered["payload"])
    payload["holds"] = False
    tampered["payload"] = payload
    ok = (
        replay_cone_certificate(sealed.certificate) is True
        and verify_certificate_digest(sealed.certificate)
        and verify_certificate_digest(tampered) is False
        and sealed.certificate["honesty"]["navier_stokes_proof_claim"] is False
        and sealed.mathlib_verified is False
    )
    return {"name": "g4_seal_replay", "passed": ok, "detail": "digest / replay / no parent flag"}


def _run_g5() -> dict[str, Any]:
    src = generate_obligation(
        seal_cone_certificate(locked_interior_cone(), run_lean=False).certificate
    )
    ok = src is not None and "allRatLt" in src
    return {"name": "g5_kernel_emission", "passed": ok, "detail": "allRatLt on a discharged cone"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "stress_cone.json" if full else "stress_cone_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.stress_cone.v1",
            config={"family": "stress_cone", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "stress_cone"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
