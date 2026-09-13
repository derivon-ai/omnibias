# SPDX-License-Identifier: Apache-2.0
"""Independent radii arithmetic and scope regressions for gauge-polar sources."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.wilson_polar_source import (
    replay_su2_wilson_polar_vacuum_certificate as replay,
)
from omnibias.geometry.gauge.transfer.wilson_polar_source import su2_wilson_polar_vacuum as certify
from omnibias.geometry.gauge.transfer.wilson_residual_source import (
    replay_su2_wilson_linear_vacuum_certificate,
    replay_su2_wilson_residual_vacuum_certificate,
    su2_wilson_linear_vacuum,
    su2_wilson_residual_vacuum,
)
from omnibias.geometry.gauge.transfer.wilson_static_source import (
    replay_su2_wilson_static_family_certificate,
    su2_wilson_static_family,
)


@pytest.mark.parametrize("family,kappa,base,radius", [
    ("cubic", Q(15), Q(1), Q(1, 10)),
    ("cubic", Q(15), Q(33, 32), Q(1, 8)),
    ("cubic", Q(141, 10), Q(1), Q(23, 100)),
    ("cubic", Q(29, 2), Q(1), Q(3, 20)),
    ("strip", Q(10), Q(1), Q(1, 16)),
])
def test_complete_family_source_and_physical_gap(
    family: str, kappa: Q, base: Q, radius: Q,
) -> None:
    result = certify(kappa, family=family, correction_radius=radius, decay_base=base)
    a = result["witness"]["arithmetic"]
    d, pairs = (2, 3) if family == "strip" else (4, 42)
    g, quadratic = 4 / kappa**2, Q(8, 9)
    residual = (3 * d * base**2 + Q(8, 3) * pairs * base**3) * g**2
    linear = 8 * quadratic * d * g * base**2
    assert Q(a["quadratic_constant"]) == quadratic
    assert Q(a["linear_upper"]) == linear
    assert Q(a["residual_norm_upper"]) == residual
    assert Q(a["self_map_slack"]) == radius - residual - linear * radius - quadratic * radius**2 >= 0
    assert Q(a["contraction_upper"]) == linear + 2 * quadratic * radius < 1
    assert Q(a["radius_feasibility_discriminant"]) == (1 - linear)**2 - 4 * quadratic * residual > 0
    assert result["status"] == "PASS"
    assert result["actual_vacuum_verified"]
    assert result["volume_uniform_actual_vacuum_family_verified"]
    assert result["beyond_previous_source_criterion_verified"]
    assert Q(result["physical_gap_lower"]) > 0
    assert replay(result["certificate"])


def test_cubic_fifteen_exact_arithmetic_and_static_energy() -> None:
    source = certify(15, correction_radius=Q(1, 10))
    a = source["witness"]["arithmetic"]
    assert a["linear_upper"] == "1024/2025"
    assert a["self_map_slack"] == "137/101250"
    assert a["contraction_upper"] == "1384/2025"
    assert a["curvature_physical_gap_lower"] == "367/180"
    assert Q(source["physical_gap_lower"]) >= Q(367, 180) > 2
    assert not su2_wilson_linear_vacuum(15)["actual_vacuum_verified"]
    result = su2_wilson_static_family(source["certificate"])
    assert result["status"] == "PASS"
    assert result["witness"]["linear_static_energy_coefficients"] == ["367/180", "45/8"]
    assert replay_su2_wilson_static_family_certificate(result["certificate"])


def test_weighted_source_supplies_spatial_covariance_rate() -> None:
    result = certify(15, correction_radius=Q(1, 8), decay_base=Q(33, 32))
    a = result["witness"]["arithmetic"]
    assert a["self_map_slack"] == "149/144000"
    assert a["contraction_upper"] == "19/25"
    assert a["curvature_physical_gap_lower"] == "161/90"
    assert Q(a["conditional_comparison_margin_lower"]) > 0
    assert result["spatial_exponential_covariance_bound_verified"]
    assert result["physical_compression_hierarchy_verified"]
    assert a["actual_weighted_influence_row_upper"] == "3467/3600"
    assert a["actual_weighted_dobrushin_margin_lower"] == "133/3600"
    assert result["actual_weighted_dobrushin_bound_verified"]


@pytest.mark.parametrize("kappa", [Q(14), Q(12), Q(1)])
def test_unsuccessful_nonlinear_criterion_is_inconclusive(kappa: Q) -> None:
    result = certify(kappa)
    assert result["status"] == "INCONCLUSIVE"
    assert not result["actual_vacuum_verified"]
    assert result["physical_gap_lower"] is None
    assert replay(result["certificate"])
    assert su2_wilson_static_family(result["certificate"])["status"] == "INCONCLUSIVE"


def test_reference_inverse_without_actual_vacuum_at_twelve() -> None:
    result = certify(12)
    a = result["witness"]["arithmetic"]
    assert result["fourier_linear_inverse_verified"]
    assert result["beyond_previous_linear_gate_verified"]
    assert a["fourier_linear_inverse_upper"] == "81/17"
    assert not result["actual_vacuum_verified"]


@pytest.mark.parametrize("radius", [Q(1, 1000), Q(1), Q(1, 4)])
def test_supplied_failing_radius_is_not_replaced(radius: Q) -> None:
    result = certify(Q(141, 10), correction_radius=radius)
    assert result["witness"]["arithmetic"]["source_radius_exists_for_criterion"]
    assert not result["actual_vacuum_verified"]
    assert result["status"] == "INCONCLUSIVE"
    assert replay(result["certificate"])


@pytest.mark.parametrize("family,kappa", [("strip", 10), ("cubic", 15)])
def test_automatic_radius_is_proved(family: str, kappa: int) -> None:
    result = certify(kappa, family=family)
    a = result["witness"]["arithmetic"]
    d, linear = Q(a["residual_norm_upper"]), Q(a["linear_upper"])
    assert Q(a["selected_correction_radius"]) == 2 * d / (1 - linear)
    assert result["actual_vacuum_verified"]
    assert replay(result["certificate"])


@pytest.mark.parametrize("bad", [True, False, 1.0, "15", None])
def test_strict_coupling_input(bad: Any) -> None:
    with pytest.raises(TypeError):
        certify(bad)


@pytest.mark.parametrize("kwargs", [
    {"family": "torus"}, {"family": "su3"}, {"decay_base": Q(9, 10)},
    {"correction_radius": Q(0)}, {"exponent_steps": 0},
])
def test_domain_guards(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        certify(15, **kwargs)


@pytest.mark.parametrize("field", ["quadratic_constant", "linear_upper", "residual_norm_upper",
                                  "self_map_slack", "curvature_physical_gap_lower"])
def test_rehashed_arithmetic_forgery_refused(field: str) -> None:
    certificate = deepcopy(certify(15, correction_radius=Q(1, 10))["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = "0"
    certificate = seal_certificate(certificate)
    assert not replay(certificate)
    with pytest.raises(ValueError, match="canonical Wilson"):
        su2_wilson_static_family(certificate)


@pytest.mark.parametrize("flag", ["actual_vacuum_verified", "neutral_physical_gap_verified",
                                 "continuum_claim", "yang_mills_mass_gap_claim"])
def test_rehashed_honesty_forgery_refused(flag: str) -> None:
    certificate = deepcopy(certify(14)["certificate"])
    certificate["honesty"][flag] = True
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("bad", [None, [], 0, {}, {"payload": {"type": "su2_wilson_polar_vacuum_v1"}}])
def test_malformed_certificate(bad: Any) -> None:
    assert not replay(bad)


def test_legacy_v1_theorems_and_static_consumer_remain_canonical() -> None:
    old_residual = su2_wilson_residual_vacuum(17)
    old_linear = su2_wilson_linear_vacuum(16, correction_radius=Q(1, 8))
    assert old_residual["witness"]["arithmetic"]["quadratic_constant"] == "4/3"
    assert old_linear["witness"]["arithmetic"]["quadratic_constant"] == "4/3"
    assert old_linear["witness"]["arithmetic"]["linear_upper"] == "7/12"
    assert old_linear["physical_gap_lower"] == "2"
    assert replay_su2_wilson_residual_vacuum_certificate(old_residual["certificate"])
    assert replay_su2_wilson_linear_vacuum_certificate(old_linear["certificate"])
    assert not replay(old_linear["certificate"])
    old_static = su2_wilson_static_family(old_linear["certificate"])
    assert old_static["witness"]["linear_static_energy_coefficients"] == ["2", "6"]
    assert replay_su2_wilson_static_family_certificate(old_static["certificate"])


def test_scope_remains_fixed_coupling_and_finite_volume() -> None:
    source = certify(15, correction_radius=Q(1, 10))
    for flag in ("infinite_volume_claim", "uniform_in_a_claim", "continuum_claim",
                 "yang_mills_mass_gap_claim", "theorem_prover_verified", "mathlib_verified"):
        assert source[flag] is False
    with pytest.raises(TypeError):
        certify(15, quadratic_constant=Q(0))  # type: ignore[call-arg]
