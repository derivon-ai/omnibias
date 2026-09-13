# SPDX-License-Identifier: Apache-2.0
"""Actual-source conditional-gap gates, directed diagnostics and canonical replay."""

from __future__ import annotations

import random
from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import certificate_mode, exp_iv
from omnibias.geometry.gauge.transfer import strip_conditional_hierarchy as module
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_family,
)
from omnibias.geometry.gauge.transfer.strip_conditional_hierarchy import (
    replay_su2_strip_conditional_hierarchy_certificate as replay,
)
from omnibias.geometry.gauge.transfer.strip_conditional_hierarchy import (
    su2_strip_conditional_hierarchy as certify,
)


def _good(steps: int = 4) -> dict[str, Any]:
    return certify(24, correction_radius=Q(1, 8), decay_base=Q(5, 4), exponent_steps=steps)


def test_actual_source_oscillation_mixed_distance_conversion_and_gap_are_exact() -> None:
    result = _good()
    arithmetic = result["witness"]["arithmetic"]
    source = result["witness"]["source_certificate"]["payload"]["witness"]["arithmetic"]
    g, b, radius, alpha = Q(1, 144), Q(5, 4), Q(1, 8), Q(12)
    omega = (16 * g + 8 * radius) / 3
    gamma = Q(3, 4) * (1 - omega / 4) ** 4
    mixed = g * b / 3 + 2 * radius / (3 * b)
    margin = gamma - 2 * mixed
    assert source["fixed_point_verified"] is True
    assert Q(source["self_map_slack"]) == Q(95, 15552)
    assert Q(source["contraction_upper"]) == Q(43, 54)
    assert Q(arithmetic["conditional_log_density_oscillation_upper"]) == omega == Q(10, 27)
    assert Q(arithmetic["conditional_poincare_lower"]) == gamma == Q(5764801, 11337408)
    assert Q(arithmetic["weighted_mixed_hessian_row_upper"]) == mixed == Q(601, 8640)
    assert Q(arithmetic["uniform_conditional_comparison_margin_lower"]) == margin
    assert Q(arithmetic["weighted_covariance_kernel_row_upper"]) == 1 / margin
    assert Q(result["all_compressed_physical_gap_lower"]) == alpha * margin == Q(20937683, 4723920)
    assert alpha * margin > Q(22, 5) > 4
    old_mixed = g * b / 3 + 2 * radius / 3
    assert alpha * margin - alpha * (gamma - 2 * old_mixed) == Q(2, 5)
    assert result["status"] == "PASS" and result["digest_verified"]
    assert replay(result["certificate"])


def test_directed_grid_and_seeded_random_check_of_the_rational_exponential_inequality() -> None:
    # These enclosure comparisons independently exercise the finite formula.
    # The proof is log(1-z)<=-z, not the grid or random sample.
    rng = random.Random(9713)
    with certificate_mode():
        for steps in (1, 2, 4, 8, 16):
            points = [Q(i * steps, 32) for i in range(1, 32)]
            points += [Q(rng.randint(1, 9999) * steps, 10000) for _ in range(20)]
            for omega in points:
                lower = (1 - omega / steps) ** steps
                enclosure = exp_iv(Interval.from_value(-omega))
                assert 0 < lower <= Q.from_float(enclosure.lo)
            assert exp_iv(Interval.point(0)).contains(1)


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 16])
def test_more_exponent_steps_tighten_the_same_actual_conditional_bound(steps: int) -> None:
    lower = _good(steps)
    tighter = _good(2 * steps)
    assert lower["status"] == tighter["status"] == "PASS"
    assert Q(tighter["all_compressed_physical_gap_lower"]) > Q(
        lower["all_compressed_physical_gap_lower"]
    )
    assert replay(lower["certificate"])


@pytest.mark.parametrize("kappa, base", [(19, Q(1)), (24, Q(5, 4)), (40, Q(2))])
def test_default_radius_uses_the_earned_source_forcing(kappa: int, base: Q) -> None:
    result = certify(kappa, decay_base=base)
    forcing = 64 * base**2 / kappa**2
    assert result["witness"]["inputs"]["correction_radius"] == str(forcing)
    assert result["witness"]["arithmetic"]["source_radius_exists_for_criterion"] is True
    assert result["status"] == "PASS"
    assert replay(result["certificate"])


@pytest.mark.parametrize("kappa, radius, base", [(18, None, Q(1)), (24, Q(1, 8), Q(2))])
def test_failed_source_cannot_earn_conditional_bounds(
    kappa: int, radius: Q | None, base: Q
) -> None:
    result = certify(kappa, correction_radius=radius, decay_base=base)
    a = result["witness"]["arithmetic"]
    assert result["status"] == "INCONCLUSIVE"
    assert "actual_vacuum_fixed_point" in result["witness"]["failed_constraints"]
    assert result["actual_conditional_poincare_verified"] is False
    assert result["physical_strip_compression_hierarchy_verified"] is False
    assert result["all_compressed_physical_gap_lower"] is None
    for field in (
        "conditional_poincare_lower",
        "weighted_mixed_hessian_row_upper",
        "uniform_conditional_comparison_margin_lower",
    ):
        assert a[field] is None
    assert replay(result["certificate"])  # A faithful refusal replay remains a refusal.


def test_exponential_domain_endpoint_is_refused_not_raised_to_an_even_power() -> None:
    # g=1/4,r=1 gives Omega=4 exactly. A negative/zero base is not a usable floor.
    for radius in (Q(1), Q(2)):
        result = certify(4, correction_radius=radius, exponent_steps=4)
        a = result["witness"]["arithmetic"]
        assert a["rational_exponential_domain_verified"] is False
        assert a["exp_negative_rational_candidate_lower"] is None
        assert "rational_exponential_domain" in result["witness"]["failed_constraints"]
        assert result["all_compressed_physical_gap_lower"] is None
        assert replay(result["certificate"])


def test_actual_single_site_gap_does_not_imply_positive_joint_margin() -> None:
    result = certify(100, correction_radius=Q(1, 3))
    a = result["witness"]["arithmetic"]
    assert a["fixed_point_verified"] is True
    assert result["actual_conditional_poincare_verified"] is True
    assert Q(a["conditional_poincare_lower"]) > 0
    assert Q(a["conditional_comparison_margin_candidate_lower"]) < 0
    assert result["status"] == "INCONCLUSIVE"
    assert result["witness"]["failed_constraints"] == [
        "strict_weighted_conditional_comparison_margin"
    ]
    assert result["physical_strip_compression_hierarchy_verified"] is False
    assert replay(result["certificate"])


def test_source_curvature_and_gap_fields_are_not_premises(monkeypatch: pytest.MonkeyPatch) -> None:
    original = invariant_vacuum_fourier_family

    def erase_unconsumed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        source = original(*args, **kwargs)
        for key in list(source["witness"]["arithmetic"]):
            if any(word in key for word in ("gap", "curvature", "factorization")):
                source["witness"]["arithmetic"].pop(key)
        source["status"] = "INCONCLUSIVE"
        return source

    monkeypatch.setattr(module, "invariant_vacuum_fourier_family", erase_unconsumed)
    result = _good()
    assert result["status"] == "PASS"
    assert result["all_compressed_physical_gap_lower"] == "20937683/4723920"


def test_unweighted_kernel_is_not_promoted_to_spatial_decay() -> None:
    result = certify(24, correction_radius=Q(1, 8))
    assert result["status"] == "PASS"
    assert result["spatial_exponential_covariance_bound_verified"] is False
    assert result["witness"]["arithmetic"]["tail_rate"] is None
    assert result["witness"]["arithmetic"]["covariance_tail_prefactor_upper"] is None


@pytest.mark.parametrize(
    "key",
    [
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "infinite_volume_claim",
        "uniform_in_a_claim",
        "all_scale_refinement_claim",
        "absolute_hessian_ball_preserved_claim",
    ],
)
def test_resealed_parent_or_different_theorem_promotion_is_rejected(key: str) -> None:
    result = _good()
    assert result[key] is False
    certificate = deepcopy(result["certificate"])
    certificate["honesty"][key] = True
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize(
    "field",
    [
        "conditional_poincare_lower",
        "weighted_mixed_hessian_row_upper",
        "uniform_conditional_comparison_margin_lower",
        "all_compressed_physical_gap_lower",
    ],
)
def test_resealed_derived_arithmetic_is_rejected(field: str) -> None:
    certificate = deepcopy(_good()["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = "999"
    assert not replay(seal_certificate(certificate))


def test_resealed_source_and_physical_units_are_rejected() -> None:
    certificate = deepcopy(_good()["certificate"])
    upstream = certificate["payload"]["witness"]["source_certificate"]
    upstream["payload"]["witness"]["arithmetic"]["fixed_point_verified"] = False
    certificate["payload"]["witness"]["source_certificate"] = seal_certificate(upstream)
    assert not replay(seal_certificate(certificate))
    certificate = deepcopy(_good()["certificate"])
    certificate["payload"]["witness"]["energy_units"] = "rescaled coarse units"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("value", [True, 24.0, "24", 0, -1])
def test_exact_positive_coupling_guard(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(value)


@pytest.mark.parametrize(
    "field, value",
    [
        ("correction_radius", True),
        ("correction_radius", 0.125),
        ("correction_radius", 0),
        ("decay_base", True),
        ("decay_base", 1.25),
        ("decay_base", Q(3, 4)),
        ("exponent_steps", True),
        ("exponent_steps", 4.0),
        ("exponent_steps", Q(4)),
        ("exponent_steps", 0),
        ("exponent_steps", -1),
    ],
)
def test_strict_exact_parameter_guards(field: str, value: Any) -> None:
    args: dict[str, Any] = {"correction_radius": Q(1, 8), "decay_base": Q(5, 4)}
    args[field] = value
    with pytest.raises((TypeError, ValueError)):
        certify(24, **args)


@pytest.mark.parametrize("certificate", [None, [], (), "certificate", True, {}, {"payload": None}])
def test_malformed_replay_is_false(certificate: Any) -> None:
    assert replay(certificate) is False
