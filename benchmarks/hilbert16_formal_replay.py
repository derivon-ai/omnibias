#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay finite Hilbert saddle margin signs through omnibias's Lean bridge.

The source intervals are computed from the declared rectangle in the saddle
benchmark. Lean checks their rational sign obligations. This does not verify
the interval evaluator, identify the rectangle with the actual canonical
family, or formalize the analytic passage and cycle-count arguments.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from omnibias.core.proof.certificate import (
    interval_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lean_check import check_certificate

from benchmarks.hilbert16_root_saddle import gate_bounds, gate_passes, require


def replay() -> dict[str, object]:
    bounds = gate_bounds()
    require(gate_passes(bounds), "declared saddle rectangle passes")
    margins = {
        "strict_discriminant": bounds["leading_discriminant"],
        "positive_pre_entry_denominator": bounds["B_pre_entry"],
        "inward_left_wall": bounds["left_wall_velocity_factor"],
        "inward_right_wall": -bounds["right_wall_velocity_factor"],
        "stable_contraction_above_two": -bounds["stable_divergence_factor"] - 2,
        "positive_height_coupling": bounds["height_coupling_derivative"],
    }
    repository = Path(__file__).resolve().parents[1]
    source = repository / "formal" / "omnibias-verified-kernel"
    require(shutil.which("lake") is not None, "Lean lake toolchain is available")
    require(source.is_dir(), "existing omnibias verified kernel is available")
    reports: dict[str, object] = {}

    # The normal bridge writes and restores a generated obligation under its
    # project lock. A private snapshot also avoids touching a concurrent edit
    # or a manual build of the shared project's generated file.
    with tempfile.TemporaryDirectory(prefix="hilbert16-lean-") as work:
        workspace = Path(work)
        target = workspace / "formal" / "omnibias-verified-kernel"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(".lake", ".git"))
        for name, margin in margins.items():
            require(margin.lo > 0, f"positive computed margin: {name}")
            certificate = interval_certificate(
                f"positive rational endpoint margin for declared saddle rectangle: {name}",
                margin,
                honesty={"actual_canonical_family_envelope_established": False},
                meta={"scope": "sign of a supplied interval, conditional on enclosure membership"},
            )
            require(verify_certificate_digest(certificate), f"valid seal: {name}")
            result = check_certificate(certificate, timeout=180, start=workspace)
            require(result.available and result.verified, f"Lean margin replay: {name}: {result.detail}")
            stale = dict(certificate)
            stale["claim"] = "altered after sealing"
            rejected = check_certificate(stale, timeout=180, start=workspace)
            require(not rejected.verified and "digest mismatch" in rejected.detail, "tamper rejection")
            reports[name] = {
                "certificate": certificate,
                "lean_result": asdict(result),
                "stale_seal_rejected": True,
            }

    return {
        "schema": "hilbert16-finite-lean-margin-replay-v1",
        "source": "benchmarks/hilbert16_root_saddle.py:gate_bounds",
        "margin_count": len(reports),
        "margins": reports,
        "scope": {
            "finite_rational_interval_signs_lean_verified": True,
            "sealed_certificate_tamper_checks_passed": True,
            "interval_evaluator_formally_verified_by_this_run": False,
            "actual_canonical_family_envelope_established": False,
            "analytic_passage_formally_verified": False,
            "actual_cycle_count_formally_verified": False,
            "full_hilbert16_solved": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(replay(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
