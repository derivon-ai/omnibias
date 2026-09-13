# SPDX-License-Identifier: Apache-2.0
"""Sharper full-space inverse and nonlinear source gates are distinct."""

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer import (
    replay_su2_wilson_linear_vacuum_certificate as replay,
)
from omnibias.geometry.gauge.transfer import su2_wilson_linear_vacuum as certify
from omnibias.geometry.gauge.transfer.wilson_residual_source import (
    replay_su2_wilson_residual_vacuum_certificate,
    su2_wilson_residual_vacuum,
)


def test_cubic_fourier_inverse_crosses_the_previous_linear_boundary() -> None:
    row = certify(13)
    a = row["witness"]["arithmetic"]
    assert a["linear_upper"] == "448/507"
    assert a["generic_linear_upper"] == "512/507"
    assert a["fourier_linear_inverse_upper"] == "507/59"
    assert row["fourier_linear_inverse_verified"] is True
    assert row["beyond_previous_linear_gate_verified"] is True
    assert row["actual_vacuum_verified"] is False
    assert row["physical_gap_lower"] is None
    assert row["status"] == "INCONCLUSIVE"
    assert replay(row["certificate"])


def test_new_actual_vacuum_crosses_the_previous_radius_boundary() -> None:
    row = certify(16, correction_radius=Q(1, 8))
    a = row["witness"]["arithmetic"]
    assert a["linear_upper"] == "7/12"
    assert a["residual_norm_upper"] == "31/1024"
    assert a["radius_feasibility_discriminant"] == "7/576"
    assert a["self_map_slack"] == "1/1024"
    assert a["contraction_upper"] == "11/12"
    assert a["previous_source_radius_exists_for_criterion"] is False
    assert row["actual_vacuum_verified"] is True
    assert row["beyond_previous_source_criterion_verified"] is True
    assert row["physical_gap_lower"] == "2"
    assert row["status"] == "PASS"
    assert replay(row["certificate"])
    previous = su2_wilson_residual_vacuum(16, family="cubic", correction_radius=Q(1, 8))
    assert previous["status"] == "INCONCLUSIVE"
    assert replay_su2_wilson_residual_vacuum_certificate(previous["certificate"])
    assert not replay(previous["certificate"])
    assert not replay_su2_wilson_residual_vacuum_certificate(row["certificate"])


def test_weighted_witness_preserves_spatial_conditional_hierarchy() -> None:
    row = certify(17, decay_base=Q(17, 16), correction_radius=Q(1, 10))
    assert row["status"] == "PASS"
    assert row["actual_conditional_hierarchy_verified"] is True
    assert row["spatial_exponential_covariance_bound_verified"] is True
    assert row["beyond_previous_source_criterion_verified"] is True
    assert row["physical_gap_lower"] == "2539/1020"
    assert row["witness"]["arithmetic"]["self_map_slack"] == "1/346800"


@pytest.mark.parametrize("kappa,family", [(12, "cubic"), (8, "strip"), (1, "cubic")])
def test_linear_refusal_never_supplies_a_fourier_inverse(kappa: int, family: str) -> None:
    row = certify(kappa, family=family)
    assert row["fourier_linear_inverse_verified"] is False
    assert row["witness"]["arithmetic"]["fourier_linear_inverse_upper"] is None
    assert row["actual_vacuum_verified"] is False
    assert row["status"] == "INCONCLUSIVE"
    assert replay(row["certificate"])


@pytest.mark.parametrize("kappa,radius", [(16, Q(1, 100)), (16, Q(1)), (13, Q(1, 8))])
def test_bad_nonlinear_radius_cannot_destroy_or_promote_the_linear_result(kappa: int, radius: Q) -> None:
    row = certify(kappa, correction_radius=radius)
    assert row["fourier_linear_inverse_verified"] is True
    assert row["actual_vacuum_verified"] is False
    assert row["physical_gap_lower"] is None


@pytest.mark.parametrize("kwargs", [
    {"kappa": True}, {"kappa": 16.0}, {"kappa": 0}, {"family": "su3"},
    {"correction_radius": 0.125}, {"decay_base": Q(1, 2)}, {"exponent_steps": True},
    {"fundamental_linear": False}, {"linear_upper": 0}, {"residual_norm_upper": 0},
])
def test_no_caller_supplied_proof_constants_or_inexact_inputs(kwargs: dict[str, Any]) -> None:
    args: dict[str, Any] = {"kappa": 16}
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        certify(**args)


@pytest.mark.parametrize("field", ["linear_upper", "generic_linear_upper", "fourier_linear_inverse_upper",
                                 "previous_source_radius_exists_for_criterion", "self_map_slack"])
def test_resealed_arithmetic_tampering_is_rejected(field: str) -> None:
    certificate = deepcopy(certify(16, correction_radius=Q(1, 8))["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = "unearned"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("flag", ["actual_vacuum_verified", "continuum_claim", "yang_mills_mass_gap_claim"])
def test_inverse_only_certificate_cannot_be_resealed_as_a_nonlinear_or_parent_proof(flag: str) -> None:
    certificate = deepcopy(certify(13)["certificate"])
    certificate["honesty"][flag] = True
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("value", [None, [], {}, 1, "certificate"])
def test_malformed_replay(value: Any) -> None:
    assert not replay(value)


@pytest.mark.parametrize("kappa,family,radius,base,digest", [
    (11, "strip", Q(1, 12), Q(1), "485ff477a59496492028b844f38fa8ba1170fb756149f10584aa10c9283c2380"),
    (12, "strip", Q(1, 20), Q(17, 16), "2a5bd0e675269b4132c235560f08e27f76279a9276719a94aae54ff7909b6db4"),
    (17, "cubic", Q(1, 12), Q(1), "65e8a9d528a371ba03c7dec2ae8e60330d1552c11e8365c1fede6496ca64c44b"),
    (18, "cubic", Q(1, 12), Q(17, 16), "4e4b87a32f2a241c79510523edb2b74e012a243b8ab7bdb92a0ef23e2e5a9cb5"),
    (10, "strip", None, Q(1), "aad2ef39fc4a9cd457ad2ae1dd83b46b14355498a506ae52164211d301ece793"),
    (16, "cubic", None, Q(1), "0ac2c2a097eb5980d4488e3bf32ecb8724f4cba1e079322346c712d60269cf5b"),
])
def test_six_preexisting_v1_certificate_digests_are_unchanged(
    kappa: int, family: str, radius: Q | None, base: Q, digest: str,
) -> None:
    row = su2_wilson_residual_vacuum(kappa, family=family, correction_radius=radius, decay_base=base)
    assert row["certificate"]["digest"] == "sha256:" + digest
    assert replay_su2_wilson_residual_vacuum_certificate(row["certificate"])
