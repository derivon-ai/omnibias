# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: anisotropic similarity profile (theory 07-09)."""

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
from omnibias.pinn.certified.anisotropic import (
    apply_T_b,
    apply_Z_b,
    axis_germ,
    coefficient_source_jet,
    honesty_payload,
    locked_axis_regular_profile,
    profile_jet_coeffs,
    residual_plus_div,
)


def _run_g1() -> dict[str, Any]:
    profile = locked_axis_regular_profile()
    jet = profile.jet_at(Fraction(1), Fraction(0))
    ok = (
        apply_T_b(profile.h, Fraction(0), Fraction(1), Fraction(0), jet) == 1
        and apply_Z_b(profile.h, Fraction(0), Fraction(1), Fraction(0), jet) == 0
    )
    return {"name": "g1_lemma_41", "passed": ok, "detail": "T_b, Z_b match Lemma 4.1"}


def _run_g2() -> dict[str, Any]:
    ok = residual_plus_div(Fraction(1)) == 0
    return {"name": "g2_minus_div_T", "passed": ok, "detail": "R + div T = 0"}


def _run_g3() -> dict[str, Any]:
    germ = axis_germ(locked_axis_regular_profile(), order=2)
    ok = germ.remainder.lo == 0.0 and germ.remainder.hi == 0.0
    return {"name": "g3_axis_germ", "passed": ok, "detail": "TM remainder is {0}"}


def _run_g4() -> dict[str, Any]:
    report = coefficient_source_jet(profile_jet_coeffs(order=1), order=1)
    return {
        "name": "g4_source_jet",
        "passed": bool(report["matches"]),
        "detail": "jet_multiply matches Cauchy product",
    }


def _run_g5() -> dict[str, Any]:
    flags = honesty_payload()
    ok = (
        flags["navier_stokes_proof_claim"] is False
        and flags["forced_blowup_reproof_claim"] is False
    )
    return {"name": "g5_honesty", "passed": ok, "detail": "parent flags stay false"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "anisotropic_profile.json" if full else "anisotropic_profile_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5()]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.anisotropic_profile.v1",
            config={"family": "anisotropic_profile", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    if full:
        dest = scratch / "training" / "anisotropic_profile"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        write_json(artifact, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
