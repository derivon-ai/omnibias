# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.core.proof.lean_check import lean_check_available
from omnibias.core.proof.realization_replay import verify_replay_certificate
from omnibias.core.verified.interval import Interval
from omnibias.geometry.neuromanifold import affine_quotient
from omnibias.verify._core.param_loss import HyperDual
from omnibias.verify.neuromanifold import (
    IntervalObjective,
    certify_quotient_minimum,
    certify_slice_minimum,
)
from omnibias.verify.neuromanifold.formal import formalize_minimum, minimum_replay_certificates


def _minimum(morse_bott=False):
    chart = affine_quotient([[1, 2, -1]])

    def objective(q):
        residual = q[0] - HyperDual.constant(Q(1, 3))
        return residual * residual

    expression = IntervalObjective(objective, "squared_reduced_coordinate", ("q",))
    return certify_quotient_minimum(chart, expression, (Interval(0, 1),), morse_bott=morse_bott)


@pytest.mark.parametrize("morse_bott", [False, True])
def test_affine_quotient_minimum_binds_original_rank_and_reduced_operands(morse_bott):
    minimum = _minimum(morse_bott)
    assert minimum.status == "proved"
    assert minimum.certificate["meta"]["minimum_scope"] == "preimage of the certified reduced search box under projection"
    assert minimum.stationary_box[0].contains(1 / 3)
    certificates = minimum_replay_certificates(minimum)
    assert [c["payload"]["kind"] for c in certificates] == [
        "matrix_rank",
        "krawczyk_replay",
        "interval_ldlt",
    ]
    assert all(verify_replay_certificate(c) for c in certificates)
    assert certificates[1]["payload"]["source"]["stationary_box"] is not None


def test_actual_quotient_minimum_formalizes_in_both_projects():
    if not lean_check_available():
        pytest.skip("Lean toolchain unavailable")
    from omnibias.formal.mathlib_check import mathlib_check_available

    result = formalize_minimum(_minimum(True), mathlib=mathlib_check_available())
    assert result.theorem_prover_verified
    if mathlib_check_available():
        assert result.mathlib_verified
    assert any("intended objective" in dependency for dependency in result.dependencies)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", [["0", "0", "0"]]),
        ("projection", [["1", "100", "-1"]]),
        ("kernel", [["0", "1", "0"], ["0", "0", "1"]]),
    ],
)
def test_resealed_spoofed_source_rank_or_quotient_relations_are_refused(field, value):
    minimum = _minimum()
    forged = deepcopy(minimum.certificate)
    forged["meta"][field] = value
    minimum = replace(minimum, certificate=seal_certificate(forged))
    with pytest.raises(ValueError, match="quotient factorization"):
        formalize_minimum(minimum)


def test_resealed_claimed_stationary_point_box_cannot_borrow_search_box_proof():
    minimum = _minimum()
    forged = deepcopy(minimum.certificate)
    reduced = forged["meta"]["reduced_certificate"]
    reduced["meta"]["stationary_box"] = [[0.5, 0.5]]
    forged["meta"]["reduced_certificate"] = seal_certificate(reduced)
    minimum = replace(
        minimum, stationary_box=(Interval.point(0.5),), certificate=seal_certificate(forged)
    )
    with pytest.raises(ValueError, match="stationary box"):
        formalize_minimum(minimum)


def test_valid_but_different_rank_chart_cannot_change_reduced_coordinate_dimension():
    minimum = _minimum()
    forged = deepcopy(minimum.certificate)
    chart = affine_quotient([[1, 0, 0], [0, 1, 0]])
    forged["meta"].update(
        source=[[str(x) for x in row] for row in chart.source],
        projection=[[str(x) for x in row] for row in chart.projection],
        kernel=[[str(x) for x in row] for row in chart.kernel],
        independent_columns=list(chart.independent_columns),
    )
    with pytest.raises(ValueError, match="quotient factorization"):
        minimum_replay_certificates(replace(minimum, certificate=seal_certificate(forged)))


def test_declared_claim_and_metadata_scope_must_agree():
    minimum = _minimum()
    with pytest.raises(ValueError, match="claim or status"):
        minimum_replay_certificates(replace(minimum, claim="slice_minimum"))


def test_unavailable_toolchain_never_earns_either_flag(monkeypatch):
    import omnibias.core.proof.lean_check as kernel
    import omnibias.formal.mathlib_check as mathlib

    monkeypatch.setattr(kernel.shutil, "which", lambda _command: None)
    monkeypatch.setattr(mathlib.shutil, "which", lambda _command: None)
    result = formalize_minimum(_minimum(), mathlib=True)
    assert not result.theorem_prover_verified and not result.mathlib_verified
    assert all(
        not proof.available and not proof.verified for proof in (*result.kernel, *result.mathlib)
    )


def test_saddle_or_center_stationarity_does_not_become_a_minimum():
    def saddle(q):
        return q[0] * q[0] - q[1] * q[1]

    objective = IntervalObjective(saddle, "indefinite_quadratic", ("a", "b"))
    minimum = certify_slice_minimum(objective, (Interval(-1, 1), Interval(-1, 1)))
    assert minimum.status == "inconclusive"
    with pytest.raises(ValueError, match="successful minimum"):
        formalize_minimum(minimum)
