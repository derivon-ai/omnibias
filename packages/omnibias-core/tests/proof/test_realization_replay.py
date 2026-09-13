# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.core.proof.realization_replay import (
    generate_replay_obligation,
    interval_error_budget_certificate,
    interval_ldlt_replay_certificate,
    krawczyk_replay_certificate,
    polynomial_evaluation_certificate,
    polynomial_inequality_certificate,
    rank_replay_certificate,
    source_digest,
    strict_box_inclusion_certificate,
    verify_replay_certificate,
)
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.realization.rank import certify_matrix_rank


def _examples():
    x, y = SparsePolynomial.variable(2, 0), SparsePolynomial.variable(2, 1)
    return [
        polynomial_evaluation_certificate((x * x + 2 * y,), (Q(1, 2), 3), (Q(25, 4),)),
        polynomial_inequality_certificate((x * y, x * x - y), (2, 3), ("gt", "ge")),
        rank_replay_certificate(certify_matrix_rank([[1, 2, 3], [0, 1, 2]])),
        rank_replay_certificate(certify_matrix_rank([[0, 0], [0, 0]])),
        interval_ldlt_replay_certificate(
            [[(2, 3), (Q(1, 10), Q(1, 5))], [(Q(1, 10), Q(1, 5)), (2, 3)]]
        ),
        strict_box_inclusion_certificate([(Q(1, 4), Q(3, 4))], [(0, 1)]),
        interval_error_budget_certificate((-Q(1, 5), Q(1, 4)), Q(1, 3)),
        krawczyk_replay_certificate(
            (Q(3, 2),),
            ((Q(1, 4), Q(1, 4)),),
            (((Q(5, 2), Q(7, 2)),),),
            ((Q(1, 3),),),
            ((Q(5, 4), Q(7, 4)),),
        ),
    ]


@pytest.mark.parametrize("certificate", _examples())
def test_finite_operands_round_trip_and_bridges(certificate):
    from omnibias.core.proof.lean_check import generate_obligation

    assert verify_replay_certificate(certificate)
    assert not verify_replay_certificate(certificate, expected_source_digest="other source")
    source = generate_replay_obligation(certificate)
    assert source == generate_obligation(certificate)
    assert "by decide" in source and "RealizationReplay" in source
    assert "OmnibiasAnalytic.RealizationReplay" in generate_replay_obligation(
        certificate, mathlib=True
    )
    tamper = deepcopy(certificate)
    tamper["payload"]["source_digest"] = "wrong"
    assert not verify_replay_certificate(seal_certificate(tamper))


def test_resealed_false_target_is_emitted_for_independent_lean_rejection():
    certificate = _examples()[0]
    false = deepcopy(certificate)
    false["payload"]["expected"][0] = [100, 1]
    false = seal_certificate(false)
    assert not verify_replay_certificate(false)
    assert "100 : Rat" in generate_replay_obligation(false)


def test_resealed_rank_factor_substitution_cannot_borrow_original_minor():
    certificate = _examples()[2]
    false = deepcopy(certificate)
    false["payload"]["left"][0][0] = [100, 1]
    false = seal_certificate(false)
    assert not verify_replay_certificate(false)
    assert generate_replay_obligation(false) is not None


def test_interval_ldlt_is_bound_to_original_matrix_even_when_resealed():
    certificate = _examples()[4]
    false = deepcopy(certificate)
    false["payload"]["source"][0][0] = [[-3, 1], [-2, 1]]
    false["payload"]["source_digest"] = source_digest(false["payload"]["source"])
    false = seal_certificate(false)
    assert not verify_replay_certificate(false)
    assert not verify_replay_certificate(
        false, expected_source_digest=certificate["payload"]["source_digest"]
    )
    assert "positivePivots" in generate_replay_obligation(false)
    with pytest.raises(ValueError):
        interval_ldlt_replay_certificate([[(-3, -2), (0, 0)], [(0, 0), (2, 3)]])


def test_reject_malformed_or_naked_pivot_inputs():
    false = deepcopy(_examples()[0])
    false["payload"]["point"][0] = [True, 1]
    false = seal_certificate(false)
    assert not verify_replay_certificate(false)
    assert generate_replay_obligation(false) is None
    with pytest.raises(ValueError):
        strict_box_inclusion_certificate([(0, 1)], [(0, 2)])
    with pytest.raises(ValueError):
        interval_error_budget_certificate((-2, 1), 1)


def test_krawczyk_recomputes_from_gradient_hessian_preconditioner_operands():
    certificate = _examples()[-1]
    assert verify_replay_certificate(certificate)
    for operand, replacement in [
        ("gradient", [[[100, 1], [100, 1]]]),
        ("preconditioner", [[[0, 1]]]),
        ("hessian", [[[[0, 1], [0, 1]]]]),
    ]:
        false = deepcopy(certificate)
        false["payload"]["source"][operand] = replacement
        false["payload"]["source_digest"] = source_digest(false["payload"]["source"])
        false = seal_certificate(false)
        assert not verify_replay_certificate(false)
        assert generate_replay_obligation(false) is not None


def test_claimed_stationary_box_must_contain_the_recomputed_image():
    certificate = krawczyk_replay_certificate(
        (Q(3, 2),),
        ((Q(1, 4), Q(1, 4)),),
        (((Q(5, 2), Q(7, 2)),),),
        ((Q(1, 3),),),
        ((Q(5, 4), Q(7, 4)),),
        stationary_box=((Q(11, 8), Q(35, 24)),),
    )
    assert verify_replay_certificate(certificate)
    false = deepcopy(certificate)
    false["payload"]["source"]["stationary_box"] = [[[3, 2], [3, 2]]]
    false["payload"]["source_digest"] = source_digest(false["payload"]["source"])
    assert not verify_replay_certificate(seal_certificate(false))
    with pytest.raises(ValueError):
        krawczyk_replay_certificate(
            (Q(3, 2),),
            ((Q(1, 4), Q(1, 4)),),
            (((Q(5, 2), Q(7, 2)),),),
            ((Q(1, 3),),),
            ((Q(5, 4), Q(7, 4)),),
            stationary_box=((Q(3, 2), Q(3, 2)),),
        )


@pytest.mark.parametrize("certificate", _examples())
def test_actual_kernel_build_when_toolchain_available(certificate):
    from omnibias.core.proof.lean_check import check_certificate, lean_check_available

    if not lean_check_available():
        pytest.skip("Lean toolchain unavailable")
    assert check_certificate(certificate).verified
