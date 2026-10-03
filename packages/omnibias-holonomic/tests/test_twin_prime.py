# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact finite tests for matrix-valued close-pair sieve certificates."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest
from omnibias.core.proof import Conjecture, generate_obligation
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.holonomic import build_holonomic_machine
from omnibias.holonomic.twin_prime import (
    ASYMPTOTIC_SIEVE_KIND,
    FIAtlasParameters,
    KnownEstimate,
    RationalCell,
    admissibility_report,
    asymptotic_sieve_local_factor,
    canonical_fi_term_ledger,
    certify_asymptotic_sieve_local_product,
    certify_bounded_gap_quadratic_witness,
    certify_fi_combinatorial_replay,
    certify_fi_terminal_atlas,
    certify_fixed_shift_determinant_replay,
    certify_fixed_shift_kernel_cell,
    certify_matrix_close_pair,
    certify_matrix_variational_witness,
    certify_mobius_cell_reduction,
    certify_rho_insertion,
    fi_atlas_obligation_certificates,
    fi_combinatorial_obligation_certificates,
    fi_rational_terminal_family,
    fi_vaughan_coefficient_case,
    fixed_shift_determinant_case,
    gamma_truncation,
    h1_182_target,
    h1_186_baseline,
    local_factor_obligation_certificates,
    maximal_far_sets,
    oriented_lambda_log_vector,
    psd_report,
    seal_asymptotic_sieve_local_product,
    seal_bounded_gap_quadratic_witness,
    seal_fi_combinatorial_replay,
    seal_fi_terminal_atlas,
    seal_fixed_shift_determinant_replay,
    seal_matrix_variational_certificate,
    seal_mobius_cell_reduction,
    select_fi_dyadic_s,
    unsigned_oriented_lambda_log_vector,
    validate_fi_term_ledger,
    verify_dyadic_membership,
    verify_gamma_mobius_identity,
    von_mangoldt_log_vector,
    wright_fixed_shift_range,
)


def _projector(x: int, y: int) -> tuple[tuple[Fraction, ...], ...]:
    norm = x * x + y * y
    return (
        (Fraction(x * x, norm), Fraction(x * y, norm)),
        (Fraction(x * y, norm), Fraction(y * y, norm)),
    )


def _scale(
    matrix: tuple[tuple[Fraction, ...], ...],
    factor: Fraction,
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(factor * value for value in row) for row in matrix)


def _path_projectors() -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    return (
        _projector(1, 0),
        _projector(0, 1),
        _projector(1, 3),
        _projector(3, 4),
        _projector(4, -3),
    )


def _path_certificate(scale: Fraction = Fraction(5, 9)):
    projectors = _path_projectors()
    matrices = tuple(_scale(matrix, scale) for matrix in projectors)
    return certify_matrix_close_pair(
        shifts=(0, 30, 60, 90, 120),
        max_gap=30,
        scoring_matrices=matrices,
    )


def test_admissibility_is_a_finite_exact_check() -> None:
    report = admissibility_report((0, 30, 60, 90, 120))
    assert report.admissible
    assert dict(report.missing_residues)[2] == (1,)
    assert dict(report.missing_residues)[3] == (1, 2)
    assert dict(report.missing_residues)[5] == (1, 2, 3, 4)

    blocked = admissibility_report((0, 1))
    assert not blocked.admissible
    assert blocked.obstruction_prime == 2


def test_h1_182_target_and_h1_186_baseline_are_exact() -> None:
    target = h1_182_target()
    assert target.k == 39
    assert target.gap == 182
    assert target.shifts[-1] - target.shifts[0] == 182
    assert admissibility_report(target.shifts).admissible

    baseline = h1_186_baseline()
    assert baseline.target.k == 40
    assert baseline.target.gap == 186
    assert baseline.basis_size == 77
    assert baseline.source_components == 97
    assert baseline.raw_source_forms == 149
    assert baseline.passed_reported_threshold
    assert baseline.reported_margin == Fraction(230382667, 10**13)


def test_bounded_gap_quadratic_witness_is_exact_but_conditional() -> None:
    target = h1_182_target()
    report = certify_bounded_gap_quadratic_witness(
        target=target,
        numerator_matrix=((4, 0), (0, 4)),
        denominator_matrix=((1, 0), (0, 1)),
        coefficients=(1, 2),
    )
    assert report.status == "PROVED"
    assert report.numerator == 20
    assert report.denominator == 5
    assert report.quotient_minus_one == 4 * target.rho_star - 1
    assert report.threshold_surplus > 0
    assert report.honesty["exact_rational_quadratic_check"] is True
    assert report.honesty["source_forms_outward_certified"] is False
    assert report.honesty["bounded_gap_analytic_instantiation_proved"] is False
    assert report.honesty["h1_182_claim"] is False
    sealed = seal_bounded_gap_quadratic_witness(report)
    assert verify_certificate_digest(sealed)

    below = certify_bounded_gap_quadratic_witness(
        target=target,
        numerator_matrix=((3, 0), (0, 3)),
        denominator_matrix=((1, 0), (0, 1)),
        coefficients=(1, 2),
    )
    assert below.status == "BLOCKED"
    indefinite = certify_bounded_gap_quadratic_witness(
        target=target,
        numerator_matrix=((4, 0), (0, 4)),
        denominator_matrix=((1, 0), (0, -1)),
        coefficients=(1, 0),
    )
    assert indefinite.status == "BLOCKED"


def test_asymptotic_sieve_local_factors_recover_twin_factors_exactly() -> None:
    at_two = asymptotic_sieve_local_factor(2)
    assert at_two.verified
    assert at_two.combined_factor == 2

    at_five = asymptotic_sieve_local_factor(5)
    assert at_five.divisor_density == Fraction(4, 19)
    assert at_five.combined_factor == Fraction(15, 16)
    assert at_five.combined_factor == at_five.twin_factor

    product = certify_asymptotic_sieve_local_product(5)
    assert product.proved
    assert product.combined_product == Fraction(45, 32)
    assert product.combined_product == product.twin_product
    sealed = seal_asymptotic_sieve_local_product(product)
    assert verify_certificate_digest(sealed)
    assert sealed["honesty"]["finite_local_factor_check"] is True
    assert sealed["honesty"]["twin_prime_conjecture_proof_claim"] is False
    obligations = local_factor_obligation_certificates(product)
    assert len(obligations) == len(product.factors) + 1
    assert all(verify_certificate_digest(item) for item in obligations)
    assert all(generate_obligation(item) is not None for item in obligations)

    with pytest.raises(ValueError, match="not prime"):
        asymptotic_sieve_local_factor(9)


def test_proofmachine_replays_only_the_finite_local_factor_prefix() -> None:
    verdict = build_holonomic_machine().evaluate(
        Conjecture(
            name="shifted-prime local factors through 97",
            kind=ASYMPTOTIC_SIEVE_KIND,
            data={"prime_limit": 97},
        )
    )
    assert verdict.status == "PROVED"
    assert verdict.replay_ok is True
    assert verdict.certificate is not None
    assert verdict.certificate["honesty"]["finite_local_factor_check"] is True
    assert (
        verdict.certificate["honesty"]["twin_prime_conjecture_proof_claim"]
        is False
    )


def test_mobius_cell_atlas_certifies_only_a_finite_strict_reduction() -> None:
    classical = (RationalCell(0, 1, 0, 1),)
    discharged = (RationalCell(0, Fraction(1, 2), 0, 1),)
    required = (RationalCell(Fraction(1, 2), 1, 0, 1),)
    report = certify_mobius_cell_reduction(
        nu=Fraction(3, 4),
        classical_region=classical,
        discharged_region=discharged,
        required_region=required,
        discharged_by=(KnownEstimate("model Type II estimate", "type_ii"),),
    )
    assert report.status == "PROVED"
    assert report.exact_partition
    assert report.strict_reduction
    assert report.classical_area == 1
    assert report.required_area == Fraction(1, 2)
    assert report.external_premises
    assert report.honesty["finite_mobius_cell_partition_check"] is True
    assert report.honesty["friedlander_iwaniec_reduction_replayed"] is False
    assert report.honesty["mobius_bilinear_estimate_proved"] is False
    assert report.honesty["twin_prime_conjecture_proof_claim"] is False

    sealed = seal_mobius_cell_reduction(report)
    assert verify_certificate_digest(sealed)


def test_mobius_cell_atlas_rejects_gaps_and_parity_blind_levels() -> None:
    classical = (RationalCell(0, 1, 0, 1),)
    discharged = (RationalCell(0, Fraction(1, 3), 0, 1),)
    required = (RationalCell(Fraction(2, 3), 1, 0, 1),)
    blocked = certify_mobius_cell_reduction(
        nu=Fraction(3, 4),
        classical_region=classical,
        discharged_region=discharged,
        required_region=required,
        discharged_by=(KnownEstimate("model Type I estimate", "type_i"),),
    )
    assert blocked.status == "BLOCKED"
    assert not blocked.exact_partition

    with pytest.raises(ValueError, match="exceed 2/3"):
        certify_mobius_cell_reduction(
            nu=Fraction(2, 3),
            classical_region=classical,
            discharged_region=discharged,
            required_region=required,
            discharged_by=(KnownEstimate("model Type I estimate", "type_i"),),
        )


def test_fi_terminal_atlas_has_exact_one_over_38_unresolved_volume() -> None:
    report = certify_fi_terminal_atlas()
    assert report.status == "PROVED"
    assert report.classical_volume == Fraction(57, 2048)
    assert report.required_volume == Fraction(3, 4096)
    assert report.discharged_volume == Fraction(111, 4096)
    assert report.unresolved_fraction == Fraction(1, 38)
    assert report.far_log_saving == 3
    assert report.honesty["finite_log_polyhedral_partition_check"] is True
    assert report.honesty["friedlander_iwaniec_reduction_replayed"] is False
    assert report.honesty["remainder_estimate_proved"] is False
    assert report.honesty["mobius_bilinear_estimate_proved"] is False
    assert report.honesty["twin_prime_conjecture_proof_claim"] is False

    sealed = seal_fi_terminal_atlas(report)
    assert verify_certificate_digest(sealed)
    obligations = fi_atlas_obligation_certificates(report)
    assert len(obligations) == 3
    assert all(verify_certificate_digest(item) for item in obligations)
    assert all(generate_obligation(item) is not None for item in obligations)


@pytest.mark.parametrize(
    ("y", "z", "s"),
    [
        (1, 1, 2),
        (2, 3, 2),
        (Fraction(5, 2), Fraction(7, 3), 4),
        (4, 4, 8),
    ],
)
def test_fi_vaughan_identity_is_universal_coefficientwise(
    y: int | Fraction,
    z: int | Fraction,
    s: int,
) -> None:
    report = certify_fi_combinatorial_replay(n_max=96, y=y, z=z, s=s)
    assert report.status == "PROVED"
    assert report.identity_proved
    assert report.terminal_partition_proved
    assert report.dyadic_partition_proved
    assert all(case.left == case.right for case in report.cases)
    assert report.honesty["friedlander_iwaniec_combinatorial_replay"] is True
    assert report.honesty["friedlander_iwaniec_reduction_replayed"] is False
    assert report.honesty["mobius_bilinear_estimate_proved"] is False


def test_fi_vaughan_endpoint_assignment_is_disjoint() -> None:
    case = fi_vaughan_coefficient_case(120, y=2, z=3, s=2)
    assignments = {
        (b, c): region for region, b, c in case.assigned_pairs
    }
    assert assignments[(4, 5)] == "f2"
    assert assignments[(5, 6)] == "f3"
    assert assignments[(5, 8)] == "f1"
    assert len(assignments) == len(case.terminal_pairs)
    assert case.terminal_partition_proved


def test_fi_combinatorial_replay_seals_only_finite_algebra() -> None:
    report = certify_fi_combinatorial_replay(n_max=64, y=2, z=3, s=4)
    sealed = seal_fi_combinatorial_replay(report)
    assert verify_certificate_digest(sealed)
    assert sealed["honesty"]["friedlander_iwaniec_combinatorial_replay"] is True
    assert sealed["honesty"]["friedlander_iwaniec_reduction_replayed"] is False
    assert sealed["honesty"]["twin_prime_conjecture_proof_claim"] is False
    obligations = fi_combinatorial_obligation_certificates(report)
    assert len(obligations) == 3
    assert all(verify_certificate_digest(item) for item in obligations)
    assert all(generate_obligation(item) is not None for item in obligations)


def test_fi_combinatorial_replay_rejects_non_dyadic_dilation() -> None:
    with pytest.raises(ValueError, match="power of two"):
        certify_fi_combinatorial_replay(n_max=8, y=2, z=3, s=3)


def test_fi_dyadic_selection_and_pointwise_membership_are_exact() -> None:
    for a_squared, expected in (
        (Fraction(1), 2),
        (Fraction(9, 4), 2),
        (Fraction(4), 4),
        (Fraction(81, 4), 8),
    ):
        selection = select_fi_dyadic_s(a_squared)
        assert selection.status == "PROVED"
        assert selection.s == expected
    for value in (
        Fraction(2),
        Fraction(3),
        Fraction(4),
        Fraction(6),
        Fraction(8),
        Fraction(9),
    ):
        assert verify_dyadic_membership(value, y=2, s=4)


def test_fi_rho_insertion_is_finite_and_support_scoped() -> None:
    report = certify_rho_insertion(
        n_max=64,
        delta=2,
        z=3,
        lambdas={1: 1, 2: -1},
    )
    assert report.status == "PROVED"
    assert report.finite_nonnegative
    assert report.identity_proved
    assert all(rho == 1 for _prime, rho in report.support_values)

    blocked = certify_rho_insertion(
        n_max=16,
        delta=2,
        z=3,
        lambdas={1: 0, 2: -1},
    )
    assert blocked.status == "BLOCKED"


def test_fi_term_ledger_is_complete_single_owner_and_noncircular() -> None:
    canonical = canonical_fi_term_ledger()
    report = validate_fi_term_ledger(canonical)
    assert report.status == "PROVED"
    assert report.complete
    assert report.noncircular
    assert not report.missing
    assert not report.duplicates

    assert validate_fi_term_ledger(canonical[:-1]).status == "BLOCKED"
    assert validate_fi_term_ledger((*canonical, canonical[0])).duplicates

    s2_index = next(i for i, use in enumerate(canonical) if use.term_id == "S2")
    wrong = list(canonical)
    wrong[s2_index] = replace(wrong[s2_index], mechanism="r_prime")
    assert validate_fi_term_ledger(wrong).wrong_routes == ("S2",)

    averaged = list(canonical)
    averaged[s2_index] = replace(averaged[s2_index], shift_mode="averaged")
    assert validate_fi_term_ledger(averaged).shift_mismatches == ("S2",)

    circular = list(canonical)
    circular[s2_index] = replace(
        circular[s2_index],
        q_one_implies_twin_target=True,
    )
    assert validate_fi_term_ledger(circular).circular == ("S2",)


def test_fixed_shift_determinant_transform_is_exact_and_signed() -> None:
    assert all(
        oriented_lambda_log_vector(n) == von_mangoldt_log_vector(n)
        for n in range(1, 257)
    )
    assert oriented_lambda_log_vector(6) == ()
    assert unsigned_oriented_lambda_log_vector(6) != ()

    case = fixed_shift_determinant_case(r=5, s=3, cutoff=2)
    assert case.shifted_value == 13
    assert case.proved
    assert case.q_one == ((13, 1),)
    assert case.k_one_zero

    exception = fixed_shift_determinant_case(r=3, s=2, cutoff=2)
    assert exception.shifted_value == 4
    assert exception.power_of_two_exception
    assert exception.proved


def test_fixed_shift_determinant_replay_seals_only_finite_algebra() -> None:
    report = certify_fixed_shift_determinant_replay(
        r_max=24,
        s_max=24,
        cutoff=6,
    )
    assert report.status == "PROVED"
    assert report.cases
    assert report.no_shift_average
    assert report.no_termwise_absolute_value
    assert report.honesty["finite_fixed_shift_determinant_replay"] is True
    assert report.honesty["fixed_shift_2_completion_lemma_proved"] is False
    assert report.honesty["mobius_bilinear_estimate_proved"] is False
    sealed = seal_fixed_shift_determinant_replay(report)
    assert verify_certificate_digest(sealed)


def test_fixed_shift_kernel_cell_has_exact_unspent_loss_budget() -> None:
    cell = certify_fixed_shift_kernel_cell()
    assert cell.status == "PROVED"
    assert cell.kernel_saving_floor == Fraction(41, 640)
    assert cell.target_saving == Fraction(1, 1000)
    assert cell.completion_loss_budget == Fraction(1009, 16000)
    assert cell.honesty["fixed_shift_2_completion_lemma_proved"] is False

    general = wright_fixed_shift_range(
        short_exponent=Fraction(1, 10),
        modulus_exponent=Fraction(2, 5),
        epsilon=Fraction(1, 1000),
    )
    assert general.general_range
    fixed = wright_fixed_shift_range(
        short_exponent=Fraction(1, 7),
        modulus_exponent=Fraction(1, 2),
        epsilon=Fraction(1, 1000),
    )
    assert fixed.fixed_small_shift_range
    assert fixed.fixed_shift_size_condition_external
    uncovered = wright_fixed_shift_range(
        short_exponent=Fraction(1, 3),
        modulus_exponent=Fraction(1, 2),
        epsilon=Fraction(1, 1000),
    )
    assert not uncovered.exponent_range_covered


def test_fi_polyhedral_cells_assign_terminal_boundary_once() -> None:
    report = certify_fi_terminal_atlas()
    assert report.classical.contains(
        u=Fraction(1, 2),
        v=Fraction(3, 8),
        w=Fraction(1, 8),
    )
    assert report.discharged.contains(
        u=Fraction(1, 2),
        v=Fraction(3, 8),
        w=Fraction(1, 8),
    )
    assert not report.required.contains(
        u=Fraction(1, 2),
        v=Fraction(3, 8),
        w=Fraction(1, 8),
    )

    boundary_u = Fraction(63, 64) - Fraction(3, 8)
    assert report.discharged.contains(u=boundary_u, v=Fraction(3, 8), w=0)
    assert not report.required.contains(u=boundary_u, v=Fraction(3, 8), w=0)
    assert report.required.contains(u=Fraction(5, 8), v=Fraction(3, 8), w=0)

    assert report.moving_required.contains(
        u=Fraction(21, 40),
        v=Fraction(3, 8),
        w=0,
        ell=Fraction(1, 1000),
    )


def test_fi_rational_terminal_family_has_vanishing_log_volume() -> None:
    reports = fi_rational_terminal_family(
        (Fraction(1, 64), Fraction(1, 128), Fraction(1, 256))
    )
    assert [report.unresolved_fraction for report in reports] == [
        Fraction(1, 38),
        Fraction(1, 76),
        Fraction(1, 152),
    ]
    assert all(report.status == "PROVED" for report in reports)
    assert all(
        report.honesty["mobius_bilinear_estimate_proved"] is False
        for report in reports
    )
    with pytest.raises(ValueError, match="strictly between"):
        fi_rational_terminal_family((0,))


def test_gamma_mobius_identity_is_exact_and_squarefree_scoped() -> None:
    assert gamma_truncation(30, 5) == -2
    for value in (1, 2, 6, 30, 210):
        for cutoff in (1, 2, 5, 11):
            assert verify_gamma_mobius_identity(value, cutoff)
    with pytest.raises(ValueError, match="squarefree"):
        verify_gamma_mobius_identity(12, 5)


def test_fi_atlas_rejects_inconsistent_parameters() -> None:
    with pytest.raises(ValueError, match="lower N exponent"):
        FIAtlasParameters(lower_n_exponent=Fraction(1, 3))
    with pytest.raises(ValueError, match="positive logarithmic saving"):
        FIAtlasParameters(moving_log_power=128)


def test_exact_psd_elimination_handles_singular_and_indefinite() -> None:
    singular = psd_report(((1, 1), (1, 1)))
    assert singular.psd
    assert singular.rank == 1
    assert singular.positive_pivots == (Fraction(1),)

    zero = psd_report(((0, 0), (0, 0)))
    assert zero.psd
    assert zero.rank == 0

    assert not psd_report(((0, 1), (1, 0))).psd
    assert not psd_report(((1, 2), (2, 1))).psd


def test_path_matrix_certificate_checks_every_maximal_far_set() -> None:
    report = _path_certificate()
    assert report.proved
    assert report.maximal_far_sets == (
        (0, 2, 4),
        (0, 3),
        (1, 3),
        (1, 4),
    )
    assert all(item.psd for item in report.matrix_reports)
    assert all(item.psd for item in report.slack_reports)


def test_larger_path_scaling_is_exactly_rejected() -> None:
    report = _path_certificate(Fraction(3, 5))
    assert not report.proved
    assert "exceeding the identity" in report.detail
    assert any(not item.psd for item in report.slack_reports)


def test_matrix_certificate_rejects_shift_matrix_misalignment() -> None:
    matrices = tuple(
        _scale(matrix, Fraction(5, 9))
        for matrix in _path_projectors()
    )
    with pytest.raises(ValueError, match="matrix alignment"):
        certify_matrix_close_pair(
            shifts=(30, 0, 60, 90, 120),
            max_gap=30,
            scoring_matrices=matrices,
        )


def test_matrix_variational_crossing_is_finite_and_parent_blocked() -> None:
    close_pair = _path_certificate()
    report = certify_matrix_variational_witness(
        close_pair,
        face_grams=_path_projectors(),
        mass=Fraction(5, 2),
    )
    assert report.finite_crossing
    assert report.score == Fraction(25, 9)
    assert report.surplus == Fraction(5, 18)
    assert report.score / report.mass == Fraction(10, 9)
    assert report.external_premises
    assert report.honesty["finite_admissibility_check"] is True
    assert report.honesty["finite_matrix_inequality_check"] is True
    assert report.honesty["exact_rational_variational_check"] is True
    assert report.honesty["twin_prime_conjecture_proof_claim"] is False
    assert report.honesty["analytic_sieve_asymptotics_proved"] is False

    sealed = seal_matrix_variational_certificate(report)
    assert verify_certificate_digest(sealed)
    assert sealed["honesty"]["twin_prime_conjecture_proof_claim"] is False


def test_nonpositive_surplus_stays_blocked() -> None:
    report = certify_matrix_variational_witness(
        _path_certificate(),
        face_grams=_path_projectors(),
        mass=3,
    )
    assert report.status == "BLOCKED"
    assert report.surplus == Fraction(-2, 9)
    assert report.honesty["exact_rational_variational_check"] is False


def test_failed_matrix_constraint_cannot_claim_a_finite_matrix_check() -> None:
    report = certify_matrix_variational_witness(
        _path_certificate(Fraction(3, 5)),
        face_grams=_path_projectors(),
        mass=2,
    )
    assert report.status == "BLOCKED"
    assert report.honesty["finite_admissibility_check"] is True
    assert report.honesty["finite_matrix_inequality_check"] is False
    assert report.honesty["exact_rational_variational_check"] is False


def test_far_set_search_has_an_explicit_budget() -> None:
    with pytest.raises(ValueError, match="search_budget"):
        maximal_far_sets(tuple(range(20)), max_gap=0, search_budget=10)

