# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact tests for the parameterized PrimeGaps186 input manifest."""

import json
from fractions import Fraction
from pathlib import Path

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.holonomic.twin_prime_bounded_gap import (
    PRIME_GAP_QUADRATIC_MATRIX_NAMES,
    PRIME_GAPS_186_COMMIT,
    PRIME_GAPS_186_SOURCE_SHA256,
    ExactCoefficientInterval,
    ExactSymmetricIntervalMatrix,
    build_prime_gap_input_manifest,
    certify_prime_gap_numerical_receipt,
    certify_prime_gap_quadratic_candidate,
    certify_prime_gap_quadratic_receipt,
    certify_prime_gap_rounding_reserve,
    certify_signed_convolution_split,
    parameterize_prime_gap_evaluator_source,
    prime_gap_coefficient_descriptors,
    prime_gap_engine_layout,
    prime_gap_quadratic_kernel_spec,
    seal_prime_gap_input_manifest,
    seal_prime_gap_numerical_receipt,
    seal_prime_gap_quadratic_receipt,
)
from omnibias.holonomic.twin_prime_bounded_gap import (
    _prime_gap_evaluator_rewrites as evaluator_rewrites,
)


def _synthetic_numerical_receipt(dimension: int) -> dict[str, object]:
    manifest = build_prime_gap_input_manifest(dimension=dimension)
    components: list[dict[str, object]] = []
    for task in manifest.tasks:
        if task.kind == "low":
            parameters = {
                "low": str(task.lower),
                "high": str(task.upper),
                "slope": str(task.slope),
            }
        elif task.kind == "rank_two":
            parameters = {
                "q_low": str(task.lower),
                "q_high": str(task.upper),
            }
        else:
            parameters = {}
        task_payload: dict[str, object] = {
            "group": task.group,
            "kind": task.kind,
            "index": task.index,
            "parameters": parameters,
        }
        if task.young_q is not None:
            task_payload.update(
                {
                    "young_q": task.young_q,
                    "young_denominator": 10**6,
                    "young": str(Fraction(task.young_q, 10**6)),
                }
            )
        else:
            task_payload["restoration_coefficient"] = str(task.restoration)
        raw_names = (
            ("root_square", "outer_face_square")
            if task.group.startswith("outer_")
            else ("inner_face",)
        )
        components.append(
            {
                "task": task_payload,
                "raw_forms": {
                    name: {"lower": "0", "upper": "0"} for name in raw_names
                },
                "raw_relative_units": {name: 0 for name in raw_names},
                "component_relative_units": 0,
                "normalized_loss_upper": "0",
            }
        )
    margin = Fraction(2624989, 10**7) * 4 - 1
    return {
        "settings": {
            "intervals": 98304,
            "convolution_length": 98304 - dimension,
            "cap_decimal_scale": 100,
            "raw_relative_decimal_scale": 100,
            "component_relative_decimal_scale": 100,
            "young_denominator": 10**6,
        },
        "normalization": f"physical form / (h*sum(g_j^2))^{dimension}",
        "cap": {
            "normalized_forms": {
                "denominator": {"lower": "1", "upper": "1"},
            },
            "hybrid_numerator": {"lower": "4", "upper": "4"},
            "rounded_units": {"I_lower": 100, "I_upper": 100, "J_lower": 400},
        },
        "source_normalization_denominator": "1",
        "components": components,
        "component_count": 97,
        "raw_form_count": 149,
        "source_total_relative_units": 0,
        "normalized_source_loss_upper": "0",
        "final_quotient_lower": str(1 + margin),
        "final_margin_lower": str(margin),
        "required_margin": "1/50000",
        "passed": True,
        "status": "PASS_FRESH_NUMERICAL_CERTIFICATE",
    }


def _synthetic_quadratic_receipt() -> dict[str, object]:
    spec = prime_gap_quadratic_kernel_spec(dimension=39)
    entry_count = spec.upper_triangle_entry_count

    def matrix(first: int = 0) -> dict[str, object]:
        entries = [["0", "0"] for _ in range(entry_count)]
        entries[0] = [str(first), str(first)]
        return {"size": spec.variable_count, "upper_triangle": entries}

    matrices = {name: matrix() for name in PRIME_GAP_QUADRATIC_MATRIX_NAMES}
    matrices["denominator"] = matrix(1)
    matrices["J0"] = matrix(10)
    return {
        "schema": "omnibias.prime_gap_quadratic.v1",
        "algorithm": "coefficient_independent_interval_gram_v1",
        "source_sha256": spec.source_sha256,
        "transformed_source_sha256": "0" * 64,
        "dimension": spec.dimension,
        "intervals": spec.intervals,
        "arb_precision_bits": spec.arb_precision_bits,
        "cap_fractional_bits": spec.cap_fractional_bits,
        "source_fractional_bits": spec.source_fractional_bits,
        "signed_convolution_strategy": spec.signed_convolution_strategy,
        "descriptors": [
            descriptor.to_payload() for descriptor in spec.descriptors
        ],
        "source_task_keys": [list(key) for key in spec.source_task_keys],
        "matrices": matrices,
    }


def test_k40_manifest_replays_the_pinned_source_inventory() -> None:
    report = build_prime_gap_input_manifest(dimension=40)
    assert report.status == "PROVED"
    assert report.upstream_commit == PRIME_GAPS_186_COMMIT
    assert report.face_dimension == 39
    assert report.convolution_length == 98264
    assert report.basis_size == 77
    assert (report.old_ladder_length, report.new_ladder_length) == (29, 43)
    assert (report.selected_old_length, report.selected_new_length) == (28, 39)
    assert len(report.groups) == 6
    assert len(report.tasks) == 97
    assert (report.outer_task_count, report.inner_task_count) == (52, 45)
    assert report.raw_form_count == 149
    assert dict(report.maxima) == {
        "base": Fraction(245671040311, 258046918656),
        "enlarged": Fraction(246740809141, 258046918656),
        "full": Fraction(134821045547, 129023459328),
        "outer": Fraction(269644834091, 258046918656),
    }
    assert verify_certificate_digest(seal_prime_gap_input_manifest(report))


def test_k39_parameterization_changes_dimensions_not_source_schedule() -> None:
    baseline = build_prime_gap_input_manifest(dimension=40)
    target = build_prime_gap_input_manifest(dimension=39)
    assert target.status == "PROVED"
    assert target.face_dimension == 38
    assert target.convolution_length == 98265
    assert [group.dimension for group in target.groups] == [39, 39, 38, 38, 38, 38]
    assert target.tasks == baseline.tasks
    assert target.maxima == baseline.maxima
    assert (len(target.tasks), target.raw_form_count) == (97, 149)

    assert {
        group.identifier: tuple(
            (cell.first_index, cell.last_index) for cell in group.cells
        )
        for group in target.groups
    } == {
        "outer_h2": ((89198, 95599), (95600, 95638)),
        "outer_h25": ((95600, 98264),),
        "old_inner_h2": ((84932, 87195), (87196, 87233)),
        "old_inner_h25": ((87196, 89525),),
        "new_inner_h2": ((85163, 87250), (87251, 87288)),
        "new_inner_h25": ((87251, 89915),),
    }


def test_manifest_honesty_does_not_upgrade_exact_inputs_to_numerics() -> None:
    report = build_prime_gap_input_manifest(dimension=39)
    assert report.honesty["exact_input_manifest_replay"] is True
    assert report.honesty["arb_flint_integrals_recomputed"] is False
    assert report.honesty["source_loss_enclosures_recomputed"] is False
    assert report.honesty["finite_numerical_crossing_proved"] is False
    assert report.honesty["bounded_gap_analytic_instantiation_proved"] is False
    assert report.honesty["h1_182_claim"] is False


def test_k39_engine_layout_exposes_hidden_half_integer_midpoint() -> None:
    layout = prime_gap_engine_layout(dimension=39)
    assert layout.midpoint_offset == Fraction(39, 2)
    assert layout.outer_mask_count == 39
    assert layout.inner_mask_count == 38
    assert layout.denominator_moment_count == 39
    assert layout.face_moment_count == 38
    assert layout.normalization_multiplier == 39
    assert layout.normalization_power == 39
    assert layout.first_radial_point == Fraction(
        -773962461163,
        860156395520,
    )
    assert layout.last_radial_point == Fraction(
        373491188591,
        2580469186560,
    )


def test_evaluator_source_rewrite_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="pinned evaluator drift"):
        parameterize_prime_gap_evaluator_source("", dimension=39)

    source = "\n".join(
        old for old, _new in evaluator_rewrites(dimension=39)
    )
    transformed = parameterize_prime_gap_evaluator_source(source, dimension=39)
    assert '"convolution_length": 98265,' in transformed
    assert "k, N = 39, 98304" in transformed
    assert "F(self.k, 2)" in transformed
    assert "self.moment_interval(cap, self.k - 1, eta)" in transformed
    assert 'dimension=int(TRIAL["dimension"])' in transformed
    assert "normalization=f" in transformed
    assert "def signed_arb_poly_times_nonnegative" in transformed
    assert "correlation = signed_arb_poly_times_nonnegative" in transformed
    assert "def _driver_load_resume" in transformed
    assert '"--resume-log"' in transformed
    assert "os.fsync(output.fileno())" in transformed


def test_signed_convolution_split_matches_direct_exact_intervals() -> None:
    interval = ExactCoefficientInterval
    signed = (
        interval(Fraction(-2), Fraction(-1)),
        interval(Fraction(-1), Fraction(3)),
        interval(Fraction(2), Fraction(5)),
    )
    nonnegative = (
        interval(Fraction(0), Fraction(2)),
        interval(Fraction(1), Fraction(3)),
    )
    certificate = certify_signed_convolution_split(signed, nonnegative)
    assert certificate.status == "PROVED"
    assert certificate.exact_match
    assert certificate.direct == certificate.split
    assert certificate.direct == (
        interval(Fraction(-4), Fraction(0)),
        interval(Fraction(-8), Fraction(5)),
        interval(Fraction(-3), Fraction(19)),
        interval(Fraction(2), Fraction(15)),
    )
    assert certificate.honesty["finite_exact_interval_identity_proved"]
    assert not certificate.honesty["arb_implementation_equivalence_proved"]


def test_signed_convolution_split_rejects_signed_kernel() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        certify_signed_convolution_split(
            (ExactCoefficientInterval(Fraction(-1), Fraction(1)),),
            (ExactCoefficientInterval(Fraction(-1), Fraction(2)),),
        )


def test_quadratic_descriptor_order_and_interval_contraction_are_exact() -> None:
    descriptors = prime_gap_coefficient_descriptors()
    assert len(descriptors) == 77
    assert descriptors[0].to_payload() == {
        "index": 0,
        "signature": [],
        "radial_degree": 0,
    }
    assert descriptors[7].signature == (2,)
    assert descriptors[-1].signature == (2, 2, 2)
    assert descriptors[-1].radial_degree == 6

    interval = ExactCoefficientInterval
    matrix = ExactSymmetricIntervalMatrix(
        size=2,
        upper_triangle=(
            interval(Fraction(1), Fraction(2)),
            interval(Fraction(-1), Fraction(3)),
            interval(Fraction(4), Fraction(5)),
        ),
    )
    assert matrix.entry(1, 0) == interval(Fraction(-1), Fraction(3))
    assert matrix.contract((Fraction(2), Fraction(-1))) == interval(
        Fraction(-4),
        Fraction(17),
    )
    with pytest.raises(ValueError, match="expected 3"):
        ExactSymmetricIntervalMatrix(
            size=2,
            upper_triangle=(interval(Fraction(0), Fraction(0)),),
        )


def test_k39_quadratic_kernel_spec_and_rounding_reserve_are_complete() -> None:
    reserve = certify_prime_gap_rounding_reserve(dimension=39)
    assert reserve.status == "PROVED"
    assert (reserve.outer_task_count, reserve.inner_task_count) == (52, 45)
    assert abs(float(reserve.relative_reserve) - 9.70094937136989e-11) < 1e-24
    assert reserve.honesty["finite_rounding_reserve_proved"]
    assert not reserve.honesty["quadratic_kernels_extracted"]

    spec = prime_gap_quadratic_kernel_spec(dimension=39)
    assert spec.intervals == 98304
    assert spec.source_sha256 == PRIME_GAPS_186_SOURCE_SHA256
    assert (
        spec.arb_precision_bits,
        spec.cap_fractional_bits,
        spec.source_fractional_bits,
    ) == (160, 224, 192)
    assert spec.signed_convolution_strategy == "nonnegative_part_split"
    assert spec.variable_count == 77
    assert spec.upper_triangle_entry_count == 3003
    assert spec.cap_forms == ("denominator", "J0", "Jplus", "Jtail")
    assert (spec.source_task_count, spec.raw_source_form_count) == (97, 149)
    assert len(spec.source_task_keys) == len(set(spec.source_task_keys)) == 97
    assert spec.rounding_reserve == reserve


def test_completed_k39_receipts_are_sealed() -> None:
    """Lock the extracted k=39 kernels and the direct margin miss."""

    benchmarks = Path(__file__).resolve().parents[3] / "docs" / "benchmarks"
    numerical = json.loads(
        (benchmarks / "twin_prime_k39_numerical_receipt_sealed.json").read_text(
            encoding="utf-8"
        )
    )
    quadratic = json.loads(
        (benchmarks / "twin_prime_k39_quadratic_receipt_sealed.json").read_text(
            encoding="utf-8"
        )
    )
    assert verify_certificate_digest(numerical)
    assert verify_certificate_digest(quadratic)

    numerical_payload = numerical["payload"]
    assert numerical_payload["dimension"] == 39
    assert numerical_payload["status"] == "DISPROVED"
    assert (numerical_payload["component_count"], numerical_payload["raw_form_count"]) == (
        97,
        149,
    )
    margin = Fraction(numerical_payload["final_margin"])
    assert margin == Fraction(
        "-5040770402079525193269995987/860722528790000000000000000000"
    )
    assert margin < Fraction(numerical_payload["required_margin"])
    assert numerical_payload["published_baseline_floor_met"] is None
    assert not numerical["honesty"]["finite_k39_crossing_found"]
    assert not numerical["honesty"]["source_valid_k39_crossing_proved"]
    assert not numerical["honesty"]["h1_182_claim"]
    assert not numerical["honesty"]["dhl_claim"]
    assert not numerical["honesty"]["twin_prime_claim"]

    quadratic_payload = quadratic["payload"]
    assert quadratic_payload["dimension"] == 39
    assert quadratic_payload["status"] == "PROVED"
    assert quadratic_payload["matrix_size"] == 77
    assert quadratic_payload["matrix_entry_count"] == 5 * 3003
    assert quadratic_payload["matrix_names"] == sorted(PRIME_GAP_QUADRATIC_MATRIX_NAMES)
    assert quadratic_payload["transformed_source_sha256"] == (
        "8cf99362e0770654abcd8e3a2b3875baac2277fedc1cfb6226355a67b07b286f"
    )
    assert quadratic["honesty"]["quadratic_kernels_extracted"]
    assert quadratic["honesty"]["exact_dyadic_entries_replayed"]
    assert quadratic["honesty"]["source_task_inventory_replayed"]
    assert not quadratic["honesty"]["matrix_values_independently_recomputed"]
    assert not quadratic["honesty"]["finite_k39_crossing_found"]
    assert not quadratic["honesty"]["source_valid_k39_crossing_proved"]
    assert not quadratic["honesty"]["h1_182_claim"]
    assert not quadratic["honesty"]["dhl_claim"]
    assert not quadratic["honesty"]["twin_prime_claim"]


def test_completed_k40_numerical_receipt_summary_is_sealed() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "benchmarks"
        / "twin_prime_k40_numerical_receipt_sealed.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert verify_certificate_digest(receipt)
    assert receipt["payload"]["status"] == "PROVED"
    assert receipt["payload"]["component_count"] == 97
    assert receipt["payload"]["raw_form_count"] == 149
    assert receipt["payload"]["published_baseline_floor_met"]
    assert Fraction(receipt["payload"]["final_margin"]) > Fraction(1, 50000)
    assert not receipt["honesty"]["arb_interval_algorithms_formally_verified"]
    assert not receipt["honesty"]["dhl_claim"]
    assert not receipt["honesty"]["twin_prime_claim"]


def test_quadratic_receipt_and_candidate_screen_are_exact_but_conditional() -> None:
    receipt = certify_prime_gap_quadratic_receipt(
        _synthetic_quadratic_receipt(),
        dimension=39,
    )
    assert receipt.status == "PROVED"
    assert set(receipt.matrices) == set(PRIME_GAP_QUADRATIC_MATRIX_NAMES)
    assert receipt.to_payload()["matrix_entry_count"] == 5 * 3003
    assert receipt.honesty["quadratic_kernels_extracted"]
    assert not receipt.honesty["finite_k39_crossing_found"]
    assert not receipt.honesty["matrix_values_independently_recomputed"]
    assert not receipt.honesty["source_valid_k39_crossing_proved"]
    assert not receipt.honesty["h1_182_claim"]
    assert verify_certificate_digest(seal_prime_gap_quadratic_receipt(receipt))

    coefficients = (Fraction(1),) + (Fraction(0),) * 76
    candidate = certify_prime_gap_quadratic_candidate(receipt, coefficients)
    assert candidate.status == "PROVED"
    assert candidate.denominator == ExactCoefficientInterval(
        Fraction(1),
        Fraction(1),
    )
    assert candidate.cap_numerator == ExactCoefficientInterval(
        Fraction(10),
        Fraction(10),
    )
    assert candidate.residual_lower > 0
    assert candidate.honesty["quadratic_candidate_is_screen"]
    assert not candidate.honesty["finite_k39_crossing_found"]
    assert not candidate.honesty["source_valid_k39_crossing_proved"]
    assert not candidate.honesty["h1_182_claim"]
    assert not candidate.honesty["complete_direct_receipt_replayed"]


def test_quadratic_receipt_rejects_nondyadic_or_reordered_data() -> None:
    nondyadic = _synthetic_quadratic_receipt()
    nondyadic_matrices = nondyadic["matrices"]
    assert isinstance(nondyadic_matrices, dict)
    denominator = nondyadic_matrices["denominator"]
    assert isinstance(denominator, dict)
    upper_triangle = denominator["upper_triangle"]
    assert isinstance(upper_triangle, list)
    upper_triangle[0] = ["1/3", "1/3"]
    with pytest.raises(ArithmeticError, match="not exact dyadics"):
        certify_prime_gap_quadratic_receipt(nondyadic, dimension=39)

    reordered = _synthetic_quadratic_receipt()
    descriptors = reordered["descriptors"]
    assert isinstance(descriptors, list)
    descriptors[0], descriptors[1] = descriptors[1], descriptors[0]
    with pytest.raises(ArithmeticError, match="coefficient order"):
        certify_prime_gap_quadratic_receipt(reordered, dimension=39)


def test_numerical_receipt_checker_replays_inventory_and_arithmetic() -> None:
    receipt = _synthetic_numerical_receipt(39)
    certificate = certify_prime_gap_numerical_receipt(receipt, dimension=39)
    assert certificate.status == "PROVED"
    assert (certificate.component_count, certificate.raw_form_count) == (97, 149)
    assert certificate.final_margin > certificate.required_margin
    assert certificate.published_baseline_floor_met is None
    assert certificate.honesty["finite_receipt_arithmetic_replayed"]
    assert certificate.honesty["finite_k39_crossing_found"]
    assert not certificate.honesty["source_valid_k39_crossing_proved"]
    assert not certificate.honesty["h1_182_claim"]
    assert not certificate.honesty["analytic_distribution_inputs_proved"]
    sealed = seal_prime_gap_numerical_receipt(certificate)
    assert verify_certificate_digest(sealed)
    assert (
        sealed["meta"]["source_receipt_sha256"]  # type: ignore[index]
        == certificate.receipt_sha256
    )


def test_k39_crossing_flag_requires_the_margin_bar() -> None:
    passing = certify_prime_gap_numerical_receipt(
        _synthetic_numerical_receipt(40),
        dimension=40,
    )
    assert passing.final_margin > passing.required_margin
    assert not passing.honesty["finite_k39_crossing_found"]
    assert not passing.honesty["h1_182_claim"]

    failing = _synthetic_numerical_receipt(39)
    failing["cap"] = {
        "normalized_forms": {"denominator": {"lower": "1", "upper": "1"}},
        "hybrid_numerator": {"lower": "1", "upper": "1"},
        "rounded_units": {"I_lower": 100, "I_upper": 100, "J_lower": 100},
    }
    failing["final_quotient_lower"] = "2624989/10000000"
    failing["final_margin_lower"] = "-7375011/10000000"
    failing["passed"] = False
    failing["status"] = "FAIL_NUMERICAL_MARGIN"
    certificate = certify_prime_gap_numerical_receipt(failing, dimension=39)
    assert certificate.status == "DISPROVED"
    assert certificate.final_margin < certificate.required_margin
    assert not certificate.honesty["finite_k39_crossing_found"]
    assert not certificate.honesty["source_valid_k39_crossing_proved"]
    assert not certificate.honesty["h1_182_claim"]


def test_numerical_receipt_checker_rejects_task_tampering() -> None:
    receipt = _synthetic_numerical_receipt(40)
    components = receipt["components"]
    assert isinstance(components, list)
    first = components[0]
    assert isinstance(first, dict)
    task = first["task"]
    assert isinstance(task, dict)
    parameters = task["parameters"]
    assert isinstance(parameters, dict)
    parameters["q_low"] = "999"
    with pytest.raises(ArithmeticError, match="parameters"):
        certify_prime_gap_numerical_receipt(receipt, dimension=40)
