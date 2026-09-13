# SPDX-License-Identifier: Apache-2.0
"""Exact regressions for an embedding defined by the actual theta vacuum."""

from __future__ import annotations

import copy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import theta_vacuum_refinement as module
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_bounds,
)
from omnibias.geometry.gauge.transfer.theta_vacuum_refinement import (
    _sqrt_upper,
    replay_su2_theta_vacuum_refinement_certificate,
    su2_theta_vacuum_refinement,
)


def _arithmetic(result: dict[str, Any]) -> dict[str, Any]:
    return result["witness"]["arithmetic"]  # type: ignore[no-any-return]


def _assert_exact_schur_floor(result: dict[str, Any]) -> None:
    arithmetic = _arithmetic(result)
    coarse = Q(arithmetic["compression_gap_lower"])
    fiber = Q(arithmetic["all_physical_fiber_modes_gap_lower"])
    beta2 = Q(arithmetic["relative_cross_form_beta_squared_upper"])
    gap = Q(arithmetic["physical_gap_lower"])
    discriminant = (coarse - fiber)**2 + 4 * coarse * beta2
    upper = Q(arithmetic["schur_sqrt_upper"])
    assert Q(arithmetic["schur_discriminant"]) == discriminant
    assert upper**2 >= discriminant
    assert gap == (coarse + fiber - upper) / 2
    assert 0 < gap <= min(coarse, fiber)
    assert beta2 < fiber
    # Independent exact positivity check for the 2x2 comparison form.
    assert (coarse - gap) * (fiber - gap) >= coarse * beta2


def test_compact_exact_kappa24_fixture() -> None:
    result = su2_theta_vacuum_refinement(24, correction_radius=Q(1, 36), exponent_steps=1)
    arithmetic = _arithmetic(result)
    assert result["status"] == "PASS"
    expected = {
        "g": "1/144", "electric_alpha": "12",
        "fixed_point_self_map_slack": "1/486",
        "marginal_log_oscillation_upper": "5/54",
        "fiber_log_oscillation_upper": "1/9",
        "marginal_exp_negative_lower": "49/54",
        "fiber_exp_negative_lower": "8/9",
        "conditional_loop_poincare_lower": "2/3",
        "coarse_kinetic_coefficient": "6",
        "physical_vertical_kinetic_coefficient": "5/2",
        "compression_gap_lower": "49",
        "all_physical_fiber_modes_gap_lower": "20",
        "conditional_drift_covariance_direct_upper": "25/5184",
        "conditional_drift_covariance_hessian_upper": "121/124416",
        "relative_cross_form_beta_squared_upper": "121/15552",
    }
    for key, value in expected.items():
        assert arithmetic[key] == value
    assert arithmetic["marginal_density_ratio_enclosure"] == ["49/54", "54/49"]
    assert arithmetic["marginal_bound_method"] == "outer_edge_oscillation"
    _assert_exact_schur_floor(result)
    assert Q(result["physical_gap_lower"]) > Q(199, 10)


def test_kappa19_actual_source_and_non_circular_refinement() -> None:
    result = su2_theta_vacuum_refinement(19, correction_radius=Q(1, 9))
    arithmetic = _arithmetic(result)
    assert result["status"] == "PASS"
    assert arithmetic["fixed_point_self_map_slack"] == "6791/31668003"
    assert arithmetic["marginal_log_oscillation_upper"] == "3176/9747"
    assert arithmetic["fiber_log_oscillation_upper"] == "3464/9747"
    assert Q(arithmetic["marginal_exp_negative_lower"]) == (1 - Q(3176, 9747 * 8))**8
    assert Q(arithmetic["fiber_exp_negative_lower"]) == (1 - Q(3464, 9747 * 8))**8
    assert Q(12236, 1000) < Q(result["physical_gap_lower"]) < Q(12237, 1000)
    source = result["witness"]["source_certificate"]
    assert verify_certificate_digest(source)
    assert source["payload"]["witness"]["actual_electric_graph_girth"] == 4
    assert source["payload"]["witness"]["weighted_incidence"] == ["1"] * 6 + ["2"]
    assert len(result["witness"]["graph"]["edges"]) == 7
    _assert_exact_schur_floor(result)


def test_source_gap_and_factorization_summaries_are_not_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress derived source gap summaries while retaining its actual certificate."""
    original_function = invariant_vacuum_fourier_bounds
    expected = su2_theta_vacuum_refinement(19, correction_radius=Q(1, 9))

    def source_without_gap(*args: Any, **kwargs: Any) -> dict[str, Any]:
        source: dict[str, Any] = original_function(*args, **kwargs)
        # Detach summaries, leaving the genuine nested certificate untouched.
        source["witness"] = copy.deepcopy(source["witness"])
        source["finite_gate_verified"] = False
        source["status"] = "INCONCLUSIVE"
        for values in (source["witness"]["arithmetic"],
                       source["witness"]["family"]["arithmetic"]):
            for key in tuple(values):
                if "gap" in key or "factorization" in key:
                    del values[key]
        return source

    monkeypatch.setattr(module, "invariant_vacuum_fourier_bounds", source_without_gap)
    actual = su2_theta_vacuum_refinement(19, correction_radius=Q(1, 9))
    assert actual == expected


def test_quadratic_haar_marginal_branch_is_used_when_tighter() -> None:
    result = su2_theta_vacuum_refinement(64, correction_radius=Q(1, 1000))
    arithmetic = _arithmetic(result)
    g, radius = Q(1, 1024), Q(1, 1000)
    assert result["status"] == "PASS"
    expected = 8 * g**2 / 9 + 14 * radius / 3
    assert expected < 8 * (g + radius) / 3
    assert Q(arithmetic["marginal_log_oscillation_upper"]) == expected
    assert arithmetic["marginal_bound_method"] == "quadratic_Haar_cancellation"
    _assert_exact_schur_floor(result)


@pytest.mark.parametrize("offset, passes", [(Q(-1, 10**30), False), (Q(0), True),
                                             (Q(1, 10**30), True)])
def test_exact_source_self_map_boundary(offset: Q, passes: bool) -> None:
    # At kappa=64/3 and r=3/64 the correction self-map inequality is equality.
    result = su2_theta_vacuum_refinement(Q(64, 3), correction_radius=Q(3, 64) + offset)
    assert result["actual_vacuum_embedding_verified"] is passes
    assert result["finite_gate_verified"] is passes
    if offset == 0:
        assert _arithmetic(result)["fixed_point_self_map_slack"] == "0"
        assert _arithmetic(result)["fixed_point_contraction_upper"] == "1/2"
    assert replay_su2_theta_vacuum_refinement_certificate(result["certificate"])


@pytest.mark.parametrize("kappa,radius", [(18, Q(1, 9)), (24, Q(19, 72)), (1, Q(1))])
def test_failed_source_never_earns_computed_conditional_bounds(kappa: int, radius: Q) -> None:
    result = su2_theta_vacuum_refinement(kappa, correction_radius=radius)
    assert result["status"] == "INCONCLUSIVE"
    assert result["physical_gap_lower"] is None
    assert not result["actual_vacuum_embedding_verified"]
    assert not result["actual_conditional_estimates_verified"]
    assert "actual_vacuum_fixed_point" in result["witness"]["failed_constraints"]
    for key in ("compression_gap_lower", "all_physical_fiber_modes_gap_lower",
                "relative_cross_form_beta_squared_upper", "schur_sqrt_upper",
                "physical_gap_lower"):
        assert _arithmetic(result)[key] is None
    assert replay_su2_theta_vacuum_refinement_certificate(result["certificate"])


@pytest.mark.parametrize("value", [Q(0), Q(1), Q(9, 16), Q(4, 9), Q(2), Q(1001, 317)])
@pytest.mark.parametrize("bits", [0, 1, 8, 64])
def test_sqrt_upper_is_the_smallest_exact_dyadic_upper(value: Q, bits: int) -> None:
    upper = _sqrt_upper(value, bits)
    step = Q(1, 1 << bits)
    assert upper**2 >= value
    assert (upper / step).denominator == 1
    if upper:
        assert (upper - step)**2 < value
    else:
        assert value == 0
    if value in (Q(0), Q(1)):
        assert upper == value
    if value == Q(9, 16) and bits >= 2:
        assert upper == Q(3, 4)


def test_finer_sqrt_precision_tightens_without_overstating_root() -> None:
    previous = Q(0)
    for bits in (0, 4, 16, 64):
        result = su2_theta_vacuum_refinement(
            24, correction_radius=Q(1, 36), exponent_steps=1, sqrt_bits=bits
        )
        _assert_exact_schur_floor(result)
        gap = Q(result["physical_gap_lower"])
        assert gap >= previous
        previous = gap


@pytest.mark.parametrize("kappa,kwargs,error", [
    (0, {"correction_radius": Q(1, 9)}, ValueError),
    (-1, {"correction_radius": Q(1, 9)}, ValueError),
    (19.0, {"correction_radius": Q(1, 9)}, TypeError),
    (True, {"correction_radius": Q(1, 9)}, TypeError),
    (19, {"correction_radius": 0}, ValueError),
    (19, {"correction_radius": -1}, ValueError),
    (19, {"correction_radius": 0.1}, TypeError),
    (19, {"correction_radius": True}, TypeError),
    (19, {"correction_radius": Q(1, 9), "exponent_steps": 0}, ValueError),
    (19, {"correction_radius": Q(1, 9), "exponent_steps": 2.0}, TypeError),
    (19, {"correction_radius": Q(1, 9), "exponent_steps": True}, TypeError),
    (19, {"correction_radius": Q(1, 9), "sqrt_bits": -1}, ValueError),
    (19, {"correction_radius": Q(1, 9), "sqrt_bits": 8.0}, TypeError),
    (19, {"correction_radius": Q(1, 9), "sqrt_bits": False}, TypeError),
])
def test_invalid_inputs(kappa: Any, kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        su2_theta_vacuum_refinement(kappa, **kwargs)


@pytest.mark.parametrize("kappa", [18, 19])
def test_scope_and_replay_for_both_outcomes(kappa: int) -> None:
    result = su2_theta_vacuum_refinement(kappa, correction_radius=Q(1, 9))
    for key in ("all_scale_refinement_claim", "coarse_wilson_family_closed",
                "infinite_volume_claim", "uniform_in_a_claim", "continuum_claim",
                "yang_mills_claim", "yang_mills_mass_gap_claim",
                "theorem_prover_verified", "mathlib_verified"):
        assert result[key] is False
    assert "gauge-invariant physical seven-edge" in result["witness"]["gap_scope"]
    assert result["digest_verified"]
    assert replay_su2_theta_vacuum_refinement_certificate(result["certificate"])


@pytest.mark.parametrize("key,value", [
    ("physical_vertical_kinetic_coefficient", "3"),
    ("coarse_kinetic_coefficient", "7"),
    ("physical_gap_lower", "100"),
    ("marginal_log_oscillation_upper", "0"),
    ("relative_cross_form_beta_squared_upper", "0"),
])
def test_rehashed_arithmetic_tampering_fails(key: str, value: str) -> None:
    certificate = copy.deepcopy(su2_theta_vacuum_refinement(19, correction_radius=Q(1, 9))["certificate"])
    certificate["payload"]["witness"]["arithmetic"][key] = value
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay_su2_theta_vacuum_refinement_certificate(certificate)


def test_rehashing_nested_source_and_outer_digest_does_not_forge_a_proof() -> None:
    certificate = copy.deepcopy(su2_theta_vacuum_refinement(19, correction_radius=Q(1, 9))["certificate"])
    source = certificate["payload"]["witness"]["source_certificate"]
    source["payload"]["witness"]["arithmetic"]["correction_radius"] = "0"
    certificate["payload"]["witness"]["source_certificate"] = seal_certificate(source)
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert verify_certificate_digest(certificate["payload"]["witness"]["source_certificate"])
    assert not replay_su2_theta_vacuum_refinement_certificate(certificate)


@pytest.mark.parametrize("bad", [None, [], 1, {}, {"digest": "sha256:bad"}])
def test_malformed_replay_is_false(bad: Any) -> None:
    assert not replay_su2_theta_vacuum_refinement_certificate(bad)
