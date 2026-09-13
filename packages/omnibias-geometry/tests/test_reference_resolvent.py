# SPDX-License-Identifier: Apache-2.0
"""Exact reference-inverse premises, projection scope, and replay falsification."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.reference_resolvent import (
    replay_su2_reference_linearized_inverse_certificate as replay,
)
from omnibias.geometry.gauge.transfer.reference_resolvent import (
    su2_reference_linearized_inverse as certify,
)


@pytest.mark.parametrize("kappa,base,gap,kernel", [
    (Q(7), Q(9, 8), Q(19, 294), Q(588, 5)),
    (Q(8), Q(5, 4), Q(1, 6), Q(64, 5)),
])
def test_cubic_original_link_reference_constants(
    kappa: Q, base: Q, gap: Q, kernel: Q,
) -> None:
    result = certify(kappa, family="cubic", decay_base=base)
    a = result["witness"]["arithmetic"]
    g = 4 / kappa**2
    assert Q(a["curvature_lower"]) == Q(1, 2) - Q(16, 3) * g == gap
    assert Q(a["weighted_hessian_row_upper"]) == 4 * g * (1 + base)**2 / 6
    assert Q(a["weighted_covariance_margin_lower"]) == 1 / kernel > 0
    assert Q(a["weighted_covariance_kernel_row_upper"]) == kernel
    assert Q(a["reference_poincare_lower"]) == gap
    assert Q(a["reference_inverse_l2_upper"]) == 1 / gap
    # Energy is <h,A^-1 h>, so its bound has ONE inverse-gap power.
    assert Q(a["reference_dirichlet_over_forcing_l2_squared_upper"]) == 1 / gap
    assert Q(a["previous_fourier_linear_upper"]) > 1
    assert result["beyond_previous_fourier_linear_gate_verified"] is True
    assert result["spatial_reference_covariance_verified"] is True
    assert result["status"] == "PASS"
    assert replay(result["certificate"])


def test_strip_product_reference_beyond_negative_curvature() -> None:
    result = certify(2, family="strip", exponent_steps=4)
    a = result["witness"]["arithmetic"]
    assert a["g"] == "1"
    assert Q(a["curvature_lower"]) < 0
    assert a["strip_one_link_log_density_oscillation_upper"] == "8/3"
    assert Q(a["strip_product_poincare_lower"]) == Q(3, 4) * Q(1, 3)**4 == Q(1, 108)
    assert a["reference_inverse_l2_upper"] == "108"
    assert a["reference_dirichlet_over_forcing_l2_squared_upper"] == "108"
    assert a["weighted_covariance_kernel_row_upper"] is None
    assert result["reference_inverse_verified"] is True
    assert result["spatial_reference_covariance_verified"] is False
    assert result["target_hamiltonian_gap_verified"] is False
    assert result["actual_vacuum_verified"] is False
    assert replay(result["certificate"])


def test_exact_exponential_boundary_refuses_zero_floor_but_more_steps_pass() -> None:
    # g=9/4 gives Omega=6 exactly. N=6 produces zero, never a positive gap.
    failed = certify(Q(4, 3), family="strip", exponent_steps=6)
    assert failed["witness"]["arithmetic"]["strip_one_link_log_density_oscillation_upper"] == "6"
    assert failed["witness"]["arithmetic"]["strip_exponential_domain_verified"] is False
    assert failed["status"] == "INCONCLUSIVE"
    assert failed["reference_inverse_verified"] is False
    assert failed["witness"]["arithmetic"]["reference_inverse_l2_upper"] is None
    assert replay(failed["certificate"])
    passed = certify(Q(4, 3), family="strip", exponent_steps=7)
    assert passed["status"] == "PASS"
    assert Q(passed["witness"]["arithmetic"]["reference_poincare_lower"]) == Q(3, 4 * 7**7)
    assert replay(passed["certificate"])


def test_rational_exponential_refinement_and_maximum_of_earned_routes() -> None:
    floors = [
        Q(certify(2, family="strip", exponent_steps=n)["witness"]["arithmetic"]
          ["reference_poincare_lower"])
        for n in (3, 4, 8, 16)
    ]
    assert all(x < y for x, y in zip(floors, floors[1:], strict=False))
    result = certify(20, family="strip")
    a = result["witness"]["arithmetic"]
    assert Q(a["reference_poincare_lower"]) == max(
        Q(a["curvature_lower"]), Q(a["strip_product_poincare_lower"]),
    )


def test_no_strip_product_measure_is_inferred_for_cubic_family() -> None:
    result = certify(2, family="cubic", exponent_steps=64)
    a = result["witness"]["arithmetic"]
    assert a["strip_exponential_domain_verified"] is True
    assert a["strip_product_poincare_lower"] is None
    assert result["status"] == "INCONCLUSIVE"
    assert result["reference_inverse_verified"] is False
    assert a["reference_inverse_l2_upper"] is None
    assert replay(result["certificate"])


def test_weighted_covariance_failure_does_not_erase_unweighted_inverse() -> None:
    result = certify(8, family="cubic", decay_base=2)
    a = result["witness"]["arithmetic"]
    assert a["reference_inverse_l2_upper"] == "6"
    assert Q(a["weighted_covariance_margin_lower"]) < 0
    assert a["weighted_covariance_kernel_row_upper"] is None
    assert result["reference_inverse_verified"] is True
    assert result["spatial_reference_covariance_verified"] is False
    assert replay(result["certificate"])


def test_centering_and_derivative_loss_are_explicit() -> None:
    witness = certify(7)["witness"]
    assert "nu-centered" in witness["domain"]
    assert "gauge-invariant" in witness["domain"]
    assert "not the Haar projection" in witness["input_projection"]
    assert witness["haar_inverse_identity"].endswith("on Haar-centered smooth inputs")
    assert "Pi_H*A_nu^(-1)*Pi_nu" in witness["haar_inverse_identity"]
    assert witness["preconditioned_identity"].endswith("Pi_H*A_nu^(-1)*Pi_nu*C")
    assert "C on the input" in witness["derivative_loss"]
    assert "does not bound the Fourier-norm inverse" in witness["derivative_loss"]
    assert "unscaled diffusion A" in witness["operator_units"]


@pytest.mark.parametrize("key", [
    "reference_poincare_lower", "reference_inverse_l2_upper",
    "reference_dirichlet_over_forcing_l2_squared_upper",
    "weighted_covariance_margin_lower", "weighted_covariance_kernel_row_upper",
    "strip_product_poincare_lower", "previous_fourier_linear_upper",
])
def test_rehashed_arithmetic_forgery_rejected(key: str) -> None:
    certificate = deepcopy(certify(7, decay_base=Q(9, 8))["certificate"])
    certificate["payload"]["witness"]["arithmetic"][key] = "999"
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay(certificate)


@pytest.mark.parametrize("key", [
    "reference", "operator", "operator_units", "domain", "input_projection",
    "haar_inverse_identity", "preconditioned_identity", "derivative_loss",
    "strip_product_route", "curvature_route", "spatial_route",
])
def test_rehashed_scope_or_projection_rewrite_rejected(key: str) -> None:
    certificate = deepcopy(certify(7)["certificate"])
    certificate["payload"]["witness"][key] = "unearned replacement"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("key", [
    "fourier_nuclear_inverse_verified", "actual_vacuum_verified",
    "target_hamiltonian_gap_verified", "nonlinear_correction_verified",
    "infinite_volume_claim", "uniform_in_a_claim", "continuum_claim",
    "yang_mills_claim", "yang_mills_mass_gap_claim",
])
def test_reference_does_not_promote_target_or_parent_claims(key: str) -> None:
    result = certify(7)
    assert result[key] is False
    certificate = deepcopy(result["certificate"])
    certificate["honesty"][key] = True
    assert not replay(seal_certificate(certificate))


def test_failed_reference_bound_cannot_be_resealed_as_verified() -> None:
    result = certify(1, family="cubic")
    assert result["finite_gate_verified"] is False
    certificate = deepcopy(result["certificate"])
    certificate["honesty"]["reference_inverse_verified"] = True
    certificate["honesty"]["volume_uniform_reference_inverse_verified"] = True
    assert not replay(seal_certificate(certificate))
    assert result["theorem_prover_verified"] is False
    assert result["mathlib_verified"] is False


@pytest.mark.parametrize("key,value", [
    ("family", "periodic"), ("family", None), ("family", ["strip"]),
    ("kappa", 0), ("kappa", -1), ("kappa", True), ("kappa", 7.0), ("kappa", "7"),
    ("decay_base", Q(1, 2)), ("decay_base", True), ("decay_base", 1.0),
    ("exponent_steps", 0), ("exponent_steps", -1), ("exponent_steps", True),
    ("exponent_steps", 4.0), ("exponent_steps", Q(4)),
])
def test_input_guards(key: str, value: Any) -> None:
    kwargs: dict[str, Any] = {"kappa": 7}
    kwargs[key] = value
    with pytest.raises((TypeError, ValueError)):
        certify(**kwargs)


@pytest.mark.parametrize("value", [None, False, [], {}, {"payload": {}}, {"digest": "bad"}])
def test_malformed_replay_refuses(value: Any) -> None:
    assert replay(value) is False


@pytest.mark.parametrize("key,value", [
    ("kappa", "8"), ("family", "cubic"), ("decay_base", "2"), ("exponent_steps", 8),
])
def test_rehashed_changed_input_recomputed(key: str, value: Any) -> None:
    certificate = deepcopy(certify(2, family="strip")["certificate"])
    certificate["payload"]["witness"]["inputs"][key] = value
    assert not replay(seal_certificate(certificate))
