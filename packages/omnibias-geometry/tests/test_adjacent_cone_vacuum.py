# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.adjacent_cone_vacuum import (
    replay_su2_adjacent_cone_vacuum_certificate as replay,
)
from omnibias.geometry.gauge.transfer.adjacent_cone_vacuum import (
    su2_adjacent_cone_vacuum as certify,
)


@pytest.fixture(scope="module")
def row() -> dict[str, Any]:
    return certify(6, correction_radius=Q(6, 25))


def test_actual_vacuum_passes_while_curvature_test_fails(row: dict[str, Any]) -> None:
    assert row["status"] == "PASS" and row["actual_vacuum_verified"]
    arithmetic = row["witness"]["arithmetic"]
    assert Q(arithmetic["original_N_inverse_upper"]) < Q(57, 25)
    assert Q(arithmetic["preconditioned_residual_N_upper"]) < Q(1231, 10000)
    assert Q(arithmetic["self_map_slack"]) > Q(41, 250000)
    assert Q(arithmetic["contraction_upper"]) < Q(608, 625)
    assert arithmetic["curvature_lower"] == "-157/1350"
    assert row["curvature_gap_verified"] is False
    assert row["physical_gap_lower"] is None
    assert replay(row["certificate"])


def test_simple_rational_caps_independently_pass_the_nonlinear_gate() -> None:
    j, epsilon, b, radius = Q(57, 25), Q(1231, 10000), Q(8, 9), Q(6, 25)
    assert radius - epsilon - j * b * radius**2 == Q(41, 250000)
    assert 2 * j * b * radius == Q(608, 625)


def test_old_source_certificate_is_still_inconclusive(row: dict[str, Any]) -> None:
    old = row["witness"]["residual_source_certificate"]
    assert old["honesty"]["actual_vacuum_verified"] is False
    assert old["payload"]["witness"]["arithmetic"]["quadratic_constant"] == "4/3"
    assert row["witness"]["arithmetic"]["quadratic_constant"] == "8/9"


def test_automatic_radius_is_a_true_actual_source() -> None:
    row = certify(6)
    assert row["status"] == "PASS"
    arithmetic = row["witness"]["arithmetic"]
    assert Q(arithmetic["selected_correction_radius"]) == 2 * Q(arithmetic["preconditioned_residual_N_upper"])
    assert replay(row["certificate"])


def test_radius_feasibility_does_not_license_a_noncontracting_radius() -> None:
    row = certify(6, correction_radius=Q(1, 4))
    assert Q(row["witness"]["arithmetic"]["radius_feasibility_discriminant"]) > 0
    assert Q(row["witness"]["arithmetic"]["contraction_upper"]) > 1
    assert row["status"] == "INCONCLUSIVE" and not row["actual_vacuum_verified"]


def test_kappa5_remains_an_honest_refusal() -> None:
    row = certify(5)
    assert row["status"] == "INCONCLUSIVE"
    assert row["witness"]["arithmetic"]["cone_inverse_replayed"]
    assert not row["actual_vacuum_verified"]
    assert replay(row["certificate"])


@pytest.mark.parametrize("field,value", [("quadratic_constant", "1/100"),
                                       ("fixed_point_verified", False),
                                       ("curvature_lower", "1"),
                                       ("self_map_slack", "1")])
def test_resealed_mathematical_edits_fail(row: dict[str, Any], field: str, value: Any) -> None:
    certificate = deepcopy(row["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = value
    assert not replay(seal_certificate(certificate))


def test_twice_resealed_inverse_forgery_fails(row: dict[str, Any]) -> None:
    certificate = deepcopy(row["certificate"])
    nested = certificate["payload"]["witness"]["cone_inverse_certificate"]
    nested["payload"]["witness"]["arithmetic"]["original_N_inverse_upper"] = "0"
    certificate["payload"]["witness"]["cone_inverse_certificate"] = seal_certificate(nested)
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("flag", ["continuum_claim", "yang_mills_mass_gap_claim",
                                 "uniform_in_volume_claim", "all_scale_refinement_claim"])
def test_parent_flags_are_not_earned(row: dict[str, Any], flag: str) -> None:
    assert row[flag] is False
    certificate = deepcopy(row["certificate"])
    certificate["honesty"][flag] = True
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("kwargs", [{"kappa": True}, {"kappa": 6.0}, {"kappa": 0},
                                  {"kappa": 6, "cutoff": 1}, {"kappa": 6, "cutoff": 3.0},
                                  {"kappa": 6, "correction_radius": 0.24},
                                  {"kappa": 6, "correction_radius": -1}])
def test_input_guards(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(**kwargs)
