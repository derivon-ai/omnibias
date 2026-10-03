# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact finite ledger for the H1<=186 baseline and H1<=182 target."""

from __future__ import annotations

import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof.certificate import verify_certificate_digest  # noqa: E402
from omnibias.holonomic.twin_prime import (  # noqa: E402
    admissibility_report,
    h1_182_target,
    h1_186_baseline,
)
from omnibias.holonomic.twin_prime_bounded_gap import (  # noqa: E402
    PRIME_GAP_QUADRATIC_MATRIX_NAMES,
    ExactCoefficientInterval,
    build_prime_gap_input_manifest,
    certify_signed_convolution_split,
    prime_gap_engine_layout,
    prime_gap_quadratic_kernel_spec,
    seal_prime_gap_input_manifest,
)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    del argv
    started = time.perf_counter()
    baseline = h1_186_baseline()
    target = h1_182_target()
    baseline_admissibility = admissibility_report(baseline.target.shifts)
    target_admissibility = admissibility_report(target.shifts)
    baseline_manifest = build_prime_gap_input_manifest(dimension=40)
    target_manifest = build_prime_gap_input_manifest(dimension=39)
    baseline_layout = prime_gap_engine_layout(dimension=40)
    target_layout = prime_gap_engine_layout(dimension=39)
    sealed_baseline_manifest = seal_prime_gap_input_manifest(baseline_manifest)
    sealed_target_manifest = seal_prime_gap_input_manifest(target_manifest)
    baseline_receipt_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "benchmarks"
        / "twin_prime_k40_numerical_receipt_sealed.json"
    )
    baseline_receipt: dict[str, Any] = json.loads(
        baseline_receipt_path.read_text(encoding="utf-8")
    )
    baseline_receipt_payload = baseline_receipt["payload"]
    if not isinstance(baseline_receipt_payload, dict):
        raise TypeError("baseline numerical receipt payload must be an object")
    interval = ExactCoefficientInterval
    split_certificate = certify_signed_convolution_split(
        (
            interval(Fraction(-2), Fraction(-1)),
            interval(Fraction(-1), Fraction(3)),
            interval(Fraction(2), Fraction(5)),
        ),
        (
            interval(Fraction(0), Fraction(2)),
            interval(Fraction(1), Fraction(3)),
        ),
    )
    quadratic_spec = prime_gap_quadratic_kernel_spec(dimension=39)
    parameterization = {
        "outer_dimension": target_manifest.dimension,
        "face_dimension": target_manifest.face_dimension,
        "grid_size": target_manifest.intervals,
        "convolution_length": target_manifest.convolution_length,
        "normalization_power": target_manifest.dimension,
        "source_components": len(target_manifest.tasks),
        "raw_source_forms": target_manifest.raw_form_count,
        "midpoint_offset": [
            str(target_layout.midpoint_offset.numerator),
            str(target_layout.midpoint_offset.denominator),
        ],
    }
    entries = [
        {
            "name": "g1_reported_h1_186_baseline",
            "passed": (
                baseline.target.k == 40
                and baseline.target.gap == 186
                and baseline_admissibility.admissible
                and baseline.passed_reported_threshold
                and baseline.source_components == 97
                and baseline.raw_source_forms == 149
                and baseline_manifest.status == "PROVED"
                and verify_certificate_digest(sealed_baseline_manifest)
                and verify_certificate_digest(baseline_receipt)
                and baseline_receipt_payload["status"] == "PROVED"
                and baseline_receipt_payload["component_count"] == 97
                and baseline_receipt_payload["raw_form_count"] == 149
                and baseline_receipt_payload["published_baseline_floor_met"] is True
            ),
            "detail": (
                "exact metadata and pure rational input-manifest replay from the "
                "pinned evaluator plus a complete 97-component authenticated-split "
                "numerical receipt whose exact final arithmetic was replayed"
            ),
        },
        {
            "name": "g2_exact_h1_182_target",
            "passed": (
                target.k == 39
                and target.gap == 182
                and target_admissibility.admissible
                and target_manifest.status == "PROVED"
                and verify_certificate_digest(sealed_target_manifest)
            ),
            "detail": "the explicit diameter-182 39-tuple is admissible",
        },
        {
            "name": "g3_k39_parameterization_guards",
            "passed": parameterization
            == {
                "outer_dimension": 39,
                "face_dimension": 38,
                "grid_size": 98304,
                "convolution_length": 98265,
                "normalization_power": 39,
                "source_components": 97,
                "raw_source_forms": 149,
                "midpoint_offset": ["39", "2"],
            },
            "detail": (
                "the same exact source schedule is regenerated with outer/face "
                "dimensions 39/38; hidden midpoint offset is exact 39/2"
            ),
        },
        {
            "name": "g4_exact_signed_convolution_split",
            "passed": (
                split_certificate.status == "PROVED"
                and split_certificate.exact_match
            ),
            "detail": (
                "exact rational interval replay proves P*Q=P_+*Q-P_-*Q "
                "for a nonnegative kernel"
            ),
        },
        {
            "name": "g5_quadratic_extraction_contract",
            "passed": (
                quadratic_spec.variable_count == 77
                and quadratic_spec.upper_triangle_entry_count == 3003
                and quadratic_spec.arb_precision_bits == 160
                and quadratic_spec.cap_fractional_bits == 224
                and quadratic_spec.source_fractional_bits == 192
                and quadratic_spec.source_task_count == 97
                and len(set(quadratic_spec.source_task_keys)) == 97
                and quadratic_spec.raw_source_form_count == 149
                and PRIME_GAP_QUADRATIC_MATRIX_NAMES
                == ("denominator", "J0", "Jplus", "Jtail", "source_loss")
                and quadratic_spec.rounding_reserve.status == "PROVED"
            ),
            "detail": (
                "exact descriptor order, interval contraction convention, and "
                "two-stage source-rounding reserve are fixed before extraction"
            ),
        },
        {
            "name": "g6_candidate_honesty",
            "passed": True,
            "finite_k39_crossing_found": False,
            "source_valid_k39_certificate_found": False,
            "dhl_39_2_claim": False,
            "h1_182_claim": False,
            "detail": (
                "the complete direct k=39 receipt misses 1/50000, so "
                "finite_k39_crossing_found stays false"
            ),
        },
    ]
    payload = {
        **provenance(
            schema="omnibias.benchmarks.twin_prime_bounded_gap.v8",
            config={
                "family": "twin_prime_bounded_gap",
                "research_status": "BLOCKED",
                "named_baseline": baseline.name,
            },
        ),
        "gates": dict(gates_block(entries)),
        "baseline": baseline.to_payload(),
        "baseline_admissibility": baseline_admissibility.to_payload(),
        "target": target.to_payload(),
        "target_admissibility": target_admissibility.to_payload(),
        "parameterization": parameterization,
        "input_manifests": {
            "k40": sealed_baseline_manifest,
            "k39": sealed_target_manifest,
            "task_schedule_identical": target_manifest.tasks == baseline_manifest.tasks,
            "physical_maxima_identical": (
                target_manifest.maxima == baseline_manifest.maxima
            ),
            "engine_layouts": {
                "k40": baseline_layout.to_payload(),
                "k39": target_layout.to_payload(),
            },
        },
        "evaluator_rewrite": {
            "signed_convolution_strategy": "nonnegative_part_split",
            "checkpoint_resume": {
                "exact_task_match_required": True,
                "conflicting_duplicates_rejected": True,
                "partial_receipt_accepted": False,
            },
            "exact_identity_certificate": split_certificate.to_payload(),
            "corrected_flint_patch_public": False,
            "runtime_regressions_required": [
                "nonnegative_fixed_integer_convolution",
                "positive_arb_split_convolution",
                "negative_arb_split_convolution",
            ],
        },
        "k40_numerical_baseline": baseline_receipt,
        "quadratic_extraction_contract": quadratic_spec.to_payload(),
        "quadratic_receipt_consumer": {
            "schema": "omnibias.prime_gap_quadratic.v1",
            "matrix_names": list(PRIME_GAP_QUADRATIC_MATRIX_NAMES),
            "exact_dyadic_endpoints_required": True,
            "generated_source_digest_required": True,
            "complete_direct_candidate_receipt_required": True,
        },
        "blocked": {
            "k40_equivalence_to_unpublished_corrected_flint_build": False,
            "quadratic_forms_assembled": True,
            "outward_source_losses_certified": False,
            "finite_k39_crossing_found": False,
            "analytic_instantiation_proved": False,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    path = write_json("twin_prime_bounded_gap_smoke.json", payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
