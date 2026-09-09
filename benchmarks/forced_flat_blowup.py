# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: jet-flat forced concentrating field (theory 07-13)."""

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
from omnibias.core.pulse_envelope import mollifier_tail_contains_truth
from omnibias.pinn.certified.forced_flat import (
    LOCKED_H,
    LOCKED_TAU_HI,
    LOCKED_TAU_LO,
    axis_T0,
    core_energy_scale,
    core_linfty_scale,
    correct_axis_stress,
    energy_box_prefactor,
    from_rest_ramp,
    honesty_payload,
    on_window_ramp,
    uncorrected_jet_flat_profile,
)


def _run_g1() -> dict[str, Any]:
    profile = uncorrected_jet_flat_profile()
    s1 = core_linfty_scale(LOCKED_TAU_HI, profile)
    s2 = core_linfty_scale(LOCKED_TAU_LO, profile)
    ok = s1.prefactor == s2.prefactor != 0 and s1.exponent == s2.exponent == -profile.scales.A
    return {"name": "g1_linfty_ratio", "passed": ok, "detail": "prefactor shared; exponent -A"}


def _run_g2() -> dict[str, Any]:
    profile = uncorrected_jet_flat_profile()
    energy = core_energy_scale(LOCKED_TAU_HI, profile)
    ok = (
        energy.prefactor == energy_box_prefactor(profile)
        and energy.prefactor > 0
        and energy.exponent == Fraction(1, 2) - 3 * LOCKED_H
        and energy.exponent > 0
        and LOCKED_TAU_LO < LOCKED_TAU_HI
    )
    return {
        "name": "g2_energy_scale",
        "passed": ok,
        "detail": "exact prefactor; tau^{1/2-3h} decreases as tau -> 0",
    }


def _run_g3() -> dict[str, Any]:
    jet = axis_T0(uncorrected_jet_flat_profile())
    ok = jet != (0, 0)
    return {"name": "g3_uncorrected_jet", "passed": ok, "detail": "negative control nonzero"}


def _run_g4() -> dict[str, Any]:
    corrected, _a = correct_axis_stress(uncorrected_jet_flat_profile())
    ok = axis_T0(corrected) == (0, 0)
    return {"name": "g4_corrected_jet", "passed": ok, "detail": "axis jet (0,0) over Q"}


def _run_g5() -> dict[str, Any]:
    ok = (
        mollifier_tail_contains_truth(half_width=3.0)
        and from_rest_ramp().value() == 0
        and on_window_ramp().value() == 1
    )
    return {
        "name": "g5_cutoff_ramp",
        "passed": ok,
        "detail": "mollifier tail contains truth; ramp vanishes at t=0",
    }


def _run_g6() -> dict[str, Any]:
    flags = honesty_payload()
    ok = (
        flags["navier_stokes_proof_claim"] is False
        and flags["forced_blowup_reproof_claim"] is False
        and flags["c_infinity_through_t1_leftover"] is True
        and flags["pulses_leftover"] is True
        and flags["joining_leftover"] is True
        and flags["uniqueness_leftover"] is True
    )
    return {"name": "g6_honesty", "passed": ok, "detail": "parent flags false; leftover named"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "forced_flat_blowup.json" if full else "forced_flat_blowup_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5(), _run_g6()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.forced_flat_blowup.v1",
            config={
                "family": "forced_flat_blowup",
                "full": full,
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "forced_flat_blowup"
        dest.mkdir(parents=True, exist_ok=True)
        dest.joinpath(artifact).write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
