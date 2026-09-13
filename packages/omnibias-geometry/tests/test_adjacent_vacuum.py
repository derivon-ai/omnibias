# SPDX-License-Identifier: Apache-2.0
"""Exact actual-vacuum gates, source-specific tails, and full-parent replay."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.adjacent_vacuum import (
    replay_su2_adjacent_preconditioned_vacuum_certificate as replay,
)
from omnibias.geometry.gauge.transfer.adjacent_vacuum import (
    su2_adjacent_preconditioned_vacuum as certify,
)


def _good() -> dict[str, Any]:
    return certify(7, correction_radius=Q(1, 10))


def _poly(rows: list[dict[str, Any]]) -> dict[tuple[int, ...], Q]:
    return {tuple(row["state"]): Q(row["coefficient"]) for row in rows}


def _norm(poly: dict[tuple[int, ...], Q]) -> tuple[Q, Q]:
    anchors = []
    for axis in range(3):
        value = Q(0)
        for state, coefficient in poly.items():
            a, b, s = state
            energy = Q(3 * a * (a + 2) + 3 * b * (b + 2) + s * (s + 2), 4)
            value += Q(state[axis], 2) * energy * (a + 1)**2 * (b + 1)**2 * abs(coefficient)
        anchors.append(value)
    return max(anchors), sum(anchors, Q(0))


@pytest.mark.parametrize("kappa,radius,gap", [
    (7, Q(1, 10), Q(73, 140)), (8, Q(1, 16), Q(1)),
])
def test_actual_seven_edge_vacuum_and_exact_gap(kappa: int, radius: Q, gap: Q) -> None:
    result = certify(kappa, correction_radius=radius)
    a = result["witness"]["arithmetic"]
    g = Q(4, kappa**2)
    inverse = Q(a["original_N_inverse_upper"])
    epsilon = Q(a["preconditioned_residual_N_upper"])
    beta = inverse * Q(4, 3)
    curvature = Q(1, 2) - Q(8, 3) * g - Q(4, 3) * radius
    assert Q(a["preconditioned_quadratic_upper"]) == beta
    assert Q(a["self_map_slack"]) == radius - epsilon - beta * radius**2 > 0
    assert Q(a["contraction_upper"]) == 2 * beta * radius < 1
    assert Q(a["seed_hessian_row_upper"]) == 4 * g / 3
    assert Q(a["correction_hessian_row_upper"]) == 2 * radius / 3
    assert Q(a["curvature_lower"]) == curvature > 0
    assert Q(result["physical_gap_lower"]) == kappa * curvature / 2 == gap
    assert result["actual_vacuum_verified"] is True
    assert result["all_spin_actual_vacuum_verified"] is True
    assert result["finite_graph_physical_gap_verified"] is True
    assert result["status"] == "PASS"
    assert replay(result["certificate"])


def test_simple_rational_kappa7_witness_is_implied_by_exact_fields() -> None:
    a = _good()["witness"]["arithmetic"]
    assert Q(a["original_N_inverse_upper"]) < Q(13, 5)
    assert Q(a["preconditioned_residual_N_upper"]) < Q(8, 125)
    beta, radius = Q(13, 5) * Q(4, 3), Q(1, 10)
    assert radius - Q(8, 125) - beta * radius**2 == Q(1, 750)
    assert 2 * beta * radius == Q(52, 75)
    assert Q(a["self_map_slack"]) > Q(1, 750)
    assert Q(a["contraction_upper"]) < Q(52, 75)


def test_exact_residual_has_all_four_original_theta_modes() -> None:
    result = _good()
    witness, g = result["witness"], Q(4, 49)
    residual = _poly(witness["residual_coefficients"])
    assert residual == {
        (2, 0, 2): -g**2 / 24, (0, 2, 2): -g**2 / 24,
        (1, 1, 0): g**2 / 27, (1, 1, 2): -g**2 / 39,
    }
    assert _norm(residual) == (Q(26, 3) * g**2, 20 * g**2)
    assert Q(witness["arithmetic"]["raw_reference_residual_N"]) == Q(26, 3) * g**2
    assert witness["arithmetic"]["residual_identity_verified"] is True
    assert witness["arithmetic"]["residual_fully_retained"] is True


def test_residual_error_uses_half_norm_conversion_and_complete_outgoing_shell() -> None:
    witness = _good()["witness"]
    a = witness["arithmetic"]
    head = _poly(witness["retained_residual_solution"])
    outgoing = _poly(witness["complete_outgoing_residual"])
    assert outgoing
    assert all(max(state) <= 3 for state in head)
    assert all(max(state) == 4 for state in outgoing)
    head_n, _ = _norm(head)
    out_n, out_m = _norm(outgoing)
    assert 2 * out_n <= out_m <= 3 * out_n
    z = Q(a["parent_all_spin_defect_upper"])
    error = out_m / (2 * (1 - z))
    assert Q(a["finite_preconditioned_residual_N"]) == head_n
    assert Q(a["outgoing_residual_M"]) == out_m
    assert Q(a["complete_residual_error_N"]) == error
    assert Q(a["preconditioned_residual_N_upper"]) == head_n + error
    assert Q(a["preconditioned_residual_N_lower"]) == max(Q(0), head_n - error)
    assert witness["arithmetic"]["finite_residual_solve_verified"] is True
    assert "N(error)<=M(h)/(2*(1-z))" in witness["residual_error_identity"]


def test_parent_is_replayed_and_the_inherited_inverse_is_not_a_user_number() -> None:
    witness = _good()["witness"]
    parent = witness["source_inverse_certificate"]
    parent_a = parent["payload"]["witness"]["arithmetic"]
    assert witness["arithmetic"]["parent_inverse_replayed"] is True
    assert witness["arithmetic"]["original_N_inverse_upper"] == parent_a["original_N_inverse_upper"]
    assert witness["arithmetic"]["parent_all_spin_defect_upper"] == parent_a["all_spin_defect_upper"]
    assert parent["honesty"]["full_spin_fourier_reference_inverse_verified"] is True
    assert parent["honesty"]["actual_vacuum_verified"] is False


@pytest.mark.parametrize("kappa", [7, 8, 9])
def test_automatic_radius_is_a_constructive_rational_witness(kappa: int) -> None:
    result = certify(kappa)
    a = result["witness"]["arithmetic"]
    epsilon, beta = Q(a["preconditioned_residual_N_upper"]), Q(a["preconditioned_quadratic_upper"])
    discriminant = 1 - 4 * beta * epsilon
    assert Q(a["radius_feasibility_discriminant"]) == discriminant > 0
    assert Q(a["selected_correction_radius"]) == 2 * epsilon
    assert Q(a["self_map_slack"]) == epsilon * discriminant > 0
    assert Q(a["contraction_upper"]) == 1 - discriminant < 1
    assert result["actual_vacuum_verified"] is True
    assert replay(result["certificate"])


@pytest.mark.parametrize("kappa", [5, 6])
def test_inverse_success_does_not_assert_nonlinear_success(kappa: int) -> None:
    result = certify(kappa)
    a = result["witness"]["arithmetic"]
    assert result["replayed_reference_inverse_verified"] is True
    assert result["preconditioned_residual_verified"] is True
    assert Q(a["radius_feasibility_discriminant"]) < 0
    assert a["selected_correction_radius"] is None
    assert a["source_radius_exists_for_criterion"] is False
    assert result["actual_vacuum_verified"] is False
    assert result["physical_gap_lower"] is None
    assert result["status"] == "INCONCLUSIVE"
    assert replay(result["certificate"])


def test_failed_requested_radius_cannot_borrow_other_radius_success() -> None:
    result = certify(7, correction_radius=Q(1, 100))
    a = result["witness"]["arithmetic"]
    assert a["source_radius_exists_for_criterion"] is True
    assert Q(a["self_map_slack"]) < 0
    assert result["actual_vacuum_verified"] is False
    assert result["status"] == "INCONCLUSIVE"
    assert replay(result["certificate"])


def test_failed_parent_stays_unearned() -> None:
    result = certify(1, cutoff=2)
    assert result["replayed_reference_inverse_verified"] is False
    assert result["preconditioned_residual_verified"] is False
    assert result["actual_vacuum_verified"] is False
    assert result["witness"]["arithmetic"]["preconditioned_residual_N_upper"] is None
    assert result["physical_gap_lower"] is None
    assert replay(result["certificate"])


def test_cutoff_two_keeps_the_entire_residual_and_replays() -> None:
    result = certify(20, cutoff=2)
    assert result["witness"]["arithmetic"]["residual_fully_retained"] is True
    assert result["actual_vacuum_verified"] is True
    assert replay(result["certificate"])


@pytest.mark.parametrize("field", [
    "original_N_inverse_upper", "parent_all_spin_defect_upper",
    "raw_reference_residual_N", "finite_preconditioned_residual_N",
    "outgoing_residual_M", "complete_residual_error_N",
    "preconditioned_residual_N_upper", "quadratic_constant",
    "preconditioned_quadratic_upper", "radius_feasibility_discriminant",
    "selected_correction_radius", "self_map_slack", "contraction_upper",
    "curvature_lower", "physical_gap_lower",
])
def test_rehashed_arithmetic_tampering_is_rejected(field: str) -> None:
    certificate = deepcopy(_good()["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = "123"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("field", [
    "residual_coefficients", "retained_residual_solution", "complete_outgoing_residual",
])
def test_rehashed_coefficient_or_outgoing_tail_change_is_rejected(field: str) -> None:
    certificate = deepcopy(_good()["certificate"])
    certificate["payload"]["witness"][field][0]["coefficient"] = "0"
    assert not replay(seal_certificate(certificate))


def test_rehashed_nested_parent_forgery_is_rejected() -> None:
    certificate = deepcopy(_good()["certificate"])
    parent = certificate["payload"]["witness"]["source_inverse_certificate"]
    parent["payload"]["witness"]["arithmetic"]["all_spin_defect_upper"] = "0"
    certificate["payload"]["witness"]["source_inverse_certificate"] = seal_certificate(parent)
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay(certificate)


@pytest.mark.parametrize("field", [
    "graph", "normalization", "reference", "linearization", "residual_identity",
    "residual_error_identity", "norm_conversion", "quadratic_premise",
    "curvature_proof", "reconstruction",
])
def test_rehashed_graph_geometry_or_proof_scope_rewrites_rejected(field: str) -> None:
    certificate = deepcopy(_good()["certificate"])
    certificate["payload"]["witness"][field] = "unearned replacement"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("field", [
    "uniform_in_volume_claim", "infinite_volume_claim", "uniform_in_a_claim",
    "all_scale_refinement_claim", "continuum_claim", "yang_mills_claim",
    "yang_mills_mass_gap_claim",
])
def test_finite_graph_source_does_not_promote_family_or_parent(field: str) -> None:
    result = _good()
    assert result[field] is False
    certificate = deepcopy(result["certificate"])
    certificate["honesty"][field] = True
    assert not replay(seal_certificate(certificate))


def test_actual_gap_uses_the_original_seven_edge_metric() -> None:
    graph = _good()["witness"]["graph"]
    assert graph["n_vertices"] == 6
    assert graph["path_lengths"] == [3, 3, 1]
    assert graph["original_electric_weights"] == [1] * 7
    assert len(graph["oriented_edges"]) == 7
    assert graph["plaquettes_signed_one_based"] == [[1, 6, -3, -5], [2, 7, -4, -6]]
    assert "all six vertices" in graph["gauge_constraint"]


@pytest.mark.parametrize("field,value", [
    ("kappa", 0), ("kappa", -1), ("kappa", True), ("kappa", 7.0),
    ("cutoff", 0), ("cutoff", 1), ("cutoff", True), ("cutoff", 3.0),
    ("correction_radius", 0), ("correction_radius", Q(-1, 10)),
    ("correction_radius", True), ("correction_radius", 0.1),
])
def test_input_guards_and_incomplete_source_cutoff(field: str, value: Any) -> None:
    inputs: dict[str, Any] = {"kappa": 7, "correction_radius": Q(1, 10)}
    inputs[field] = value
    with pytest.raises((TypeError, ValueError)):
        certify(**inputs)


@pytest.mark.parametrize("field,value", [
    ("kappa", "8"), ("cutoff", 2), ("correction_radius", None),
])
def test_rehashed_changed_inputs_are_recomputed(field: str, value: Any) -> None:
    certificate = deepcopy(_good()["certificate"])
    certificate["payload"]["witness"]["inputs"][field] = value
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("value", [None, False, [], {}, {"payload": {}}, {"digest": "bad"}])
def test_malformed_replay_refuses(value: Any) -> None:
    assert replay(value) is False


def test_formal_tiers_are_not_inferred_from_rational_replay() -> None:
    result = _good()
    assert result["theorem_prover_verified"] is False
    assert result["mathlib_verified"] is False
