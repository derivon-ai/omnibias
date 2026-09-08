# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: swirl-heat identity plus pulse envelope (theory 07-12)."""

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
from omnibias.core.pulse_envelope import (
    GROWTH_RATE,
    locked_growth_envelope,
    mollifier_tail_contains_truth,
    riccati_sigma_prime,
    tower_sigma_prime,
)
from omnibias.core.pulse_envelope import (
    honesty_payload as pulse_honesty,
)
from omnibias.core.verified.swirl_heat import (
    honesty_payload as heat_honesty,
)
from omnibias.core.verified.swirl_heat import (
    swirl_heat_residual,
    swirl_heat_residual_interval,
)
from omnibias.holonomic.swirl_heat import prove_swirl_heat_identity


def _run_g1() -> dict[str, Any]:
    box = swirl_heat_residual_interval()
    ok = swirl_heat_residual() == 0 and box.contains(0.0)
    return {"name": "g1_heat_residual", "passed": ok, "detail": "residual {0} at locked points"}


def _run_g2() -> dict[str, Any]:
    result = prove_swirl_heat_identity()
    return {"name": "g2_identity", "passed": result.proved, "detail": "prove(identity)"}


def _run_g3() -> dict[str, Any]:
    result = prove_swirl_heat_identity(lean_check=True)
    return {
        "name": "g3_optional_lean",
        "passed": result.proved,
        "detail": "no-toolchain degrades; identity still proves",
    }


def _run_g4() -> dict[str, Any]:
    env = locked_growth_envelope()
    ok = (
        env.derivative() == env.tower_derivative()
        and env.derivative() == GROWTH_RATE * env.value()
        and tower_sigma_prime(Fraction(1, 2)) == riccati_sigma_prime(Fraction(1, 2))
    )
    return {"name": "g4_pulse_tower", "passed": ok, "detail": "P' matches the tower, not FD"}


def _run_g5() -> dict[str, Any]:
    ok = mollifier_tail_contains_truth(half_width=3.0)
    return {"name": "g5_mollifier_tail", "passed": ok, "detail": "tail contains true_outside_mass"}


def _run_g6() -> dict[str, Any]:
    flags = {**heat_honesty(), **pulse_honesty()}
    ok = (
        flags["navier_stokes_proof_claim"] is False
        and flags["forced_blowup_reproof_claim"] is False
    )
    return {"name": "g6_honesty", "passed": ok, "detail": "parent flags stay false"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "swirl_heat_pulse.json" if full else "swirl_heat_pulse_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5(), _run_g6()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.swirl_heat_pulse.v1",
            config={
                "family": "swirl_heat_pulse",
                "full": full,
                "honesty": {**heat_honesty(), **pulse_honesty()},
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "swirl_heat_pulse"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
