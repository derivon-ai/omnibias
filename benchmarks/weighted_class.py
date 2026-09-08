# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: fixed-order weighted class (theory 07-11)."""

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
from omnibias.core.verified.sequence_space import ValidatedSeries
from omnibias.core.verified.weighted_class import (
    check_pointwise_bound,
    gevrey_majorant,
    honesty_payload,
    locked_bound,
    locked_samples,
    violating_sample,
)


def _run_g1() -> dict[str, Any]:
    report = check_pointwise_bound(locked_samples(), locked_bound())
    return {"name": "g1_locked_samples", "passed": report.holds, "detail": "locked samples enclosed"}


def _run_g2() -> dict[str, Any]:
    report = check_pointwise_bound((violating_sample(),), locked_bound())
    ok = report.holds is False and "too_large" in report.failing
    return {"name": "g2_named_violation", "passed": ok, "detail": "violating sample named"}


def _run_g3() -> dict[str, Any]:
    ok = gevrey_majorant(
        (Fraction(1), Fraction(1, 2), Fraction(1, 4)),
        s=0,
        rho=Fraction(1, 2),
        k_max=2,
    ).holds
    refused = False
    try:
        gevrey_majorant(
            (Fraction(1), Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)),
            s=0,
            rho=Fraction(1, 2),
            k_max=2,
        )
    except ValueError as exc:
        refused = "out-of-fragment" in str(exc)
    return {"name": "g3_gevrey_majorant", "passed": ok and refused, "detail": "k_max then refuse"}


def _run_g4() -> dict[str, Any]:
    series = ValidatedSeries.from_coeffs((1.0, 0.5), nu=0.5, tail=0.0)
    return {"name": "g4_validated_series", "passed": series.norm().lo >= 0.0, "detail": "geometric path"}


def _run_g5() -> dict[str, Any]:
    flags = honesty_payload()
    ok = flags["gevrey_class_claim"] is False and flags["navier_stokes_proof_claim"] is False
    return {"name": "g5_honesty", "passed": ok, "detail": "no Gevrey / NS class claim"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "weighted_class.json" if full else "weighted_class_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.weighted_class.v1",
            config={"family": "weighted_class", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "weighted_class"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
