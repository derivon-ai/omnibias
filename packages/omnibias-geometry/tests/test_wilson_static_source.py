# SPDX-License-Identifier: Apache-2.0
"""Only an actual original-edge cubic vacuum licenses the charged cut theorem."""

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.reference_resolvent import su2_reference_linearized_inverse
from omnibias.geometry.gauge.transfer.wilson_residual_source import su2_wilson_linear_vacuum
from omnibias.geometry.gauge.transfer.wilson_static_source import (
    replay_su2_wilson_static_family_certificate as replay,
)
from omnibias.geometry.gauge.transfer.wilson_static_source import (
    su2_wilson_static_family as certify,
)


def _source() -> dict[str, Any]:
    return su2_wilson_linear_vacuum(16, correction_radius=Q(1, 8))["certificate"]


def test_actual_cubic_source_gives_two_to_six_per_separating_cut() -> None:
    row = certify(_source())
    assert row["status"] == "PASS"
    assert row["witness"]["conditional_curvature_lower"] == "1/4"
    assert row["witness"]["linear_static_energy_coefficients"] == ["2", "6"]
    assert row["finite_graph_family_confinement"] is True
    assert row["actual_vacuum_cut_poincare_verified"] is True
    assert row["witness"]["source_bare_rest_energy"] == "0"
    assert replay(row["certificate"])


def test_inverse_only_is_a_replayable_refusal_without_static_energy_coefficients() -> None:
    source = su2_wilson_linear_vacuum(13)
    assert source["fourier_linear_inverse_verified"] is True
    row = certify(source["certificate"])
    assert row["status"] == "INCONCLUSIVE"
    assert row["witness"]["linear_static_energy_coefficients"] is None
    assert row["actual_vacuum_cut_poincare_verified"] is False
    assert row["finite_graph_family_confinement"] is False
    assert replay(row["certificate"])


def test_strip_vertical_chart_cannot_supply_original_edge_cut_curvature() -> None:
    source = su2_wilson_linear_vacuum(12, family="strip")
    assert source["actual_vacuum_verified"] is True
    with pytest.raises(ValueError, match="original-edge cubic"):
        certify(source["certificate"])


def test_reference_measure_inverse_cannot_replace_actual_vacuum() -> None:
    reference = su2_reference_linearized_inverse(7)
    assert reference["reference_inverse_verified"] is True
    with pytest.raises(ValueError, match="canonical Wilson"):
        certify(reference["certificate"])


@pytest.mark.parametrize("kappa,radius", [(16, Q(1, 8)), (17, Q(1, 10)), (24, Q(1, 8))])
def test_vacuum_and_charged_energy_units_are_consistent(kappa: int, radius: Q) -> None:
    source = su2_wilson_linear_vacuum(kappa, correction_radius=radius)
    row = certify(source["certificate"])
    rho = Q(source["witness"]["arithmetic"]["curvature_lower"])
    lower, upper = (Q(v) for v in row["witness"]["linear_static_energy_coefficients"])
    assert lower == Q(kappa, 2) * rho
    assert upper == Q(3 * kappa, 8)
    assert 0 < lower <= upper
    assert row["witness"]["all_spins_included"] is True


@pytest.mark.parametrize("field,value", [
    ("conditional_curvature_lower", "1"), ("linear_static_energy_coefficients", ["9", "9"]),
    ("vacuum_subtraction", "variational upper energy"), ("source_bare_rest_energy", "10"),
    ("distance", "continuum distance"), ("family", "all continuum theories"),
])
def test_resealed_physical_scope_or_arithmetic_tampering_is_rejected(field: str, value: Any) -> None:
    certificate = deepcopy(certify(_source())["certificate"])
    certificate["payload"]["witness"][field] = value
    assert not replay(seal_certificate(certificate))


def test_nested_source_forgery_is_rejected_even_with_both_digests_resealed() -> None:
    certificate = deepcopy(certify(_source())["certificate"])
    source = certificate["payload"]["witness"]["source_certificate"]
    source["payload"]["witness"]["arithmetic"]["curvature_lower"] = "1"
    certificate["payload"]["witness"]["source_certificate"] = seal_certificate(source)
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("flag", ["continuum_claim", "yang_mills_mass_gap_claim",
                                 "infinite_volume_claim", "asymptotic_string_tension_claim"])
def test_parent_flags_cannot_be_promoted_by_resealing(flag: str) -> None:
    certificate = deepcopy(certify(_source())["certificate"])
    assert certificate["honesty"][flag] is False
    certificate["honesty"][flag] = True
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("value", [None, [], {}, 1, "certificate"])
def test_malformed_replay_returns_false(value: Any) -> None:
    assert not replay(value)
