# SPDX-License-Identifier: Apache-2.0
"""Exact comparison consequences of replayed actual adjacent vacua."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.adjacent_cone_vacuum import su2_adjacent_cone_vacuum
from omnibias.geometry.gauge.transfer.adjacent_gap_comparison import (
    replay_su2_adjacent_vacuum_gap_comparison_certificate as replay,
)
from omnibias.geometry.gauge.transfer.adjacent_gap_comparison import (
    su2_adjacent_vacuum_gap_comparison as compare,
)
from omnibias.geometry.gauge.transfer.adjacent_vacuum import su2_adjacent_preconditioned_vacuum

State = tuple[int, int, int]


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    return su2_adjacent_cone_vacuum(6, correction_radius=Q(6, 25))


@pytest.fixture(scope="module")
def good(source: dict[str, Any]) -> dict[str, Any]:
    return compare(source["certificate"])


def _weights(state: State) -> tuple[Q, Q, Q]:
    a, b, s = state
    energy = Q(3 * a * (a + 2) + 3 * b * (b + 2) + s * (s + 2), 4)
    common = energy * (a + 1)**2 * (b + 1)**2 / 2
    return common * a, common * b, common * s


def _states(cutoff: int) -> list[State]:
    return [
        (a, b, s)
        for a in range(cutoff + 1)
        for b in range(cutoff + 1)
        for s in range(abs(a - b), min(a + b, cutoff) + 1, 2)
        if (a, b, s) != (0, 0, 0)
    ]


def test_independent_dual_inequality_finite_algebra_regression() -> None:
    # The all-spin proof uses s=0/s>=1, not this finite regression.
    for state in _states(18):
        wa, wb, ws = _weights(state)
        assert (wa + wb + 11 * ws) / 72 >= 1


def test_coefficient_bound_has_an_exact_sharpness_witness() -> None:
    values = {(1, 0, 1): Q(1, 12), (0, 1, 1): Q(1, 12), (1, 1, 0): Q(1, 72)}
    anchors = [
        sum((_weights(state)[i] * abs(value) for state, value in values.items()), Q(0))
        for i in range(3)
    ]
    assert anchors == [Q(1)] * 3
    assert sum(values.values(), Q(0)) == Q(13, 72)


def test_signed_rational_polynomials_obey_the_same_dual() -> None:
    rng = Random(90721)
    states = _states(12)
    for _ in range(64):
        values = {state: Q(rng.randint(-7, 7), rng.randint(1, 17))
                  for state in rng.sample(states, 20)}
        anchors = [
            sum((_weights(state)[i] * abs(value) for state, value in values.items()), Q(0))
            for i in range(3)
        ]
        l1 = sum((abs(value) for value in values.values()), Q(0))
        assert l1 <= (anchors[0] + anchors[1] + 11 * anchors[2]) / 72
        assert l1 <= Q(13, 72) * max(anchors)


def test_new_source_is_actual_even_when_its_curvature_gate_fails(
    source: dict[str, Any], good: dict[str, Any],
) -> None:
    assert source["status"] == "PASS"
    assert source["actual_vacuum_verified"] is True
    assert source["curvature_gap_verified"] is False
    assert source["physical_gap_lower"] is None
    assert Q(source["witness"]["arithmetic"]["curvature_lower"]) == -Q(157, 1350)
    assert good["status"] == "PASS"
    assert good["beyond_nonpositive_curvature_verified"] is True
    assert good["positive_global_curvature_required"] is False
    assert good["witness"]["source_certificate"] == source["certificate"]
    assert replay(good["certificate"])


def test_kappa6_gap_uses_exact_oscillation_and_physical_units(good: dict[str, Any]) -> None:
    a = good["witness"]["arithmetic"]
    assert Q(a["g"]) == Q(1, 9)
    assert Q(a["selected_correction_radius"]) == Q(6, 25)
    assert Q(a["correction_log_density_oscillation_upper"]) == Q(13, 75)
    assert Q(a["neutral_combined_oscillation_upper"]) == Q(317, 675)
    assert Q(a["full_scalar_combined_oscillation_upper"]) == Q(517, 675)
    assert a["effective_exponent_steps"] == 4
    neutral = Q(9, 4) * Q(2383, 2700)**4
    scalar = Q(9, 4) * Q(2183, 2700)**4
    assert neutral == Q(32247508758721, 23619600000000) > Q(4, 3)
    assert scalar == Q(22709885409121, 23619600000000) > Q(24, 25)
    assert Q(good["physical_gap_lower"]) == neutral
    assert Q(good["full_scalar_gap_lower"]) == scalar < neutral


def test_simple_nonlinear_caps_are_actually_earned(source: dict[str, Any]) -> None:
    a = source["witness"]["arithmetic"]
    assert Q(a["original_N_inverse_upper"]) < Q(57, 25)
    assert Q(a["preconditioned_residual_N_upper"]) < Q(1231, 10000)
    beta, radius = Q(57, 25) * Q(8, 9), Q(6, 25)
    assert radius - Q(1231, 10000) - beta * radius**2 == Q(41, 250000)
    assert 2 * beta * radius == Q(608, 625)


@pytest.mark.parametrize("steps", [1, 2, 4, 8, 17])
def test_positive_rational_exponential_floors_and_replay(
    source: dict[str, Any], steps: int,
) -> None:
    result = compare(source["certificate"], exponent_steps=steps)
    a = result["witness"]["arithmetic"]
    assert a["effective_exponent_steps"] == steps
    for sector in ("neutral", "full_scalar"):
        omega = Q(a[sector + "_combined_oscillation_upper"])
        value = Q(a[sector + "_exp_negative_lower"])
        assert 0 < omega < steps
        assert value == (1 - omega / steps)**steps > 0
    assert replay(result["certificate"])


def test_more_exponential_steps_improve_floor_for_identical_source(
    source: dict[str, Any],
) -> None:
    gaps = [Q(compare(source["certificate"], exponent_steps=m)["physical_gap_lower"])
            for m in (1, 2, 4, 8)]
    assert gaps == sorted(gaps)
    assert len(set(gaps)) == 4


@pytest.mark.parametrize("kappa,radius,neutral,scalar", [
    (7, Q(1, 10), Q(1146601351654183441, 590180693114880000),
     Q(900268518173886481, 590180693114880000)),
    (8, Q(1, 16), Q(1416768858961, 587068342272),
     Q(1183415446801, 587068342272)),
])
def test_earlier_canonical_sources_remain_supported(
    kappa: int, radius: Q, neutral: Q, scalar: Q,
) -> None:
    source = su2_adjacent_preconditioned_vacuum(kappa, correction_radius=radius)
    result = compare(source["certificate"])
    assert result["witness"]["source_type"] == "su2_adjacent_preconditioned_vacuum_v1"
    assert result["status"] == "PASS"
    assert Q(result["physical_gap_lower"]) == neutral
    assert Q(result["full_scalar_gap_lower"]) == scalar
    assert result["beyond_nonpositive_curvature_verified"] is False
    assert replay(result["certificate"])


@pytest.mark.parametrize("kind", ["old", "cone"])
def test_failed_actual_source_is_inconclusive_not_gapless(kind: str) -> None:
    source = (su2_adjacent_preconditioned_vacuum(5) if kind == "old"
              else su2_adjacent_cone_vacuum(5))
    assert source["actual_vacuum_verified"] is False
    result = compare(source["certificate"])
    assert result["status"] == "INCONCLUSIVE"
    assert result["physical_gap_lower"] is None
    assert result["full_scalar_gap_lower"] is None
    assert result["actual_vacuum_verified"] is False
    assert result["neutral_physical_gap_verified"] is False
    assert result["full_scalar_gap_verified"] is False
    assert replay(result["certificate"])


def test_failed_supplied_radius_cannot_borrow_an_available_actual_vacuum() -> None:
    source = su2_adjacent_cone_vacuum(6, correction_radius=Q(1, 100))
    assert source["actual_vacuum_verified"] is False
    result = compare(source["certificate"])
    assert result["witness"]["arithmetic"]["selected_correction_radius"] == "1/100"
    assert result["status"] == "INCONCLUSIVE"
    assert result["physical_gap_lower"] is None
    assert replay(result["certificate"])


def test_reference_only_inverse_is_not_an_actual_source(source: dict[str, Any]) -> None:
    inverse = source["witness"]["cone_inverse_certificate"]
    assert inverse["honesty"]["actual_vacuum_verified"] is False
    with pytest.raises(ValueError, match="actual-source"):
        compare(inverse)


@pytest.mark.parametrize("field", [
    "selected_correction_radius", "source_nonlinear_fixed_point_verified",
    "inherited_global_curvature_lower", "correction_log_density_oscillation_upper",
    "neutral_combined_oscillation_upper", "full_scalar_combined_oscillation_upper",
    "effective_exponent_steps", "neutral_exp_negative_lower",
    "full_scalar_exp_negative_lower", "neutral_physical_gap_lower", "full_scalar_gap_lower",
])
def test_rehashed_arithmetic_tampering_is_rejected(
    good: dict[str, Any], field: str,
) -> None:
    certificate = deepcopy(good["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = "123"
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay(certificate)


@pytest.mark.parametrize("field", [
    "coefficient_dual", "neutral_sector", "full_scalar_sector", "neutral_reference_metric",
    "reference_poincare", "actual_density_comparison", "variance_comparison",
    "source_type", "normalization", "graph_scope", "exponential_rule", "vacuum_subtraction",
])
def test_rehashed_mathematical_scope_tampering_is_rejected(
    good: dict[str, Any], field: str,
) -> None:
    certificate = deepcopy(good["certificate"])
    certificate["payload"]["witness"][field] = "unearned"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("field", [
    "actual_vacuum_verified", "neutral_physical_gap_verified", "full_scalar_gap_verified",
    "beyond_nonpositive_curvature_verified", "positive_global_curvature_required",
    "uniform_in_volume_claim", "infinite_volume_claim", "uniform_in_a_claim",
    "all_scale_refinement_claim", "continuum_claim", "yang_mills_claim",
    "yang_mills_mass_gap_claim", "theorem_prover_verified", "mathlib_verified",
])
def test_rehashed_honesty_tampering_is_rejected(good: dict[str, Any], field: str) -> None:
    certificate = deepcopy(good["certificate"])
    certificate["honesty"][field] = not certificate["honesty"].get(field, False)
    assert not replay(seal_certificate(certificate))


def test_nested_source_must_replay_after_both_seals_are_repaired(good: dict[str, Any]) -> None:
    certificate = deepcopy(good["certificate"])
    witness = certificate["payload"]["witness"]
    nested = witness["source_certificate"]
    nested["payload"]["witness"]["arithmetic"]["selected_correction_radius"] = "1/1000000"
    witness["source_certificate"] = seal_certificate(nested)
    assert not replay(seal_certificate(certificate))


def test_nested_false_nonlinear_flag_cannot_be_forged_as_true() -> None:
    failed = su2_adjacent_cone_vacuum(5)["certificate"]
    failed["honesty"]["actual_vacuum_verified"] = True
    failed["payload"]["witness"]["arithmetic"]["fixed_point_verified"] = True
    with pytest.raises(ValueError, match="actual-source"):
        compare(seal_certificate(failed))


@pytest.mark.parametrize("value", [0, -1, True, False, Q(3, 2), 1.0, "4", None])
def test_exponent_steps_is_a_strict_positive_integer(
    source: dict[str, Any], value: Any,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        compare(source["certificate"], exponent_steps=value)


@pytest.mark.parametrize("value", [None, [], {}, {"payload": {}}, {"payload": "bad"}])
def test_malformed_source_and_replay_refuse(value: Any) -> None:
    with pytest.raises(ValueError, match="actual-source"):
        compare(value)
    assert not replay(value)


def test_only_finite_graph_scopes_are_earned(good: dict[str, Any]) -> None:
    for field in (
        "uniform_in_volume_claim", "infinite_volume_claim", "uniform_in_a_claim",
        "all_scale_refinement_claim", "continuum_claim", "yang_mills_claim",
        "yang_mills_mass_gap_claim", "theorem_prover_verified", "mathlib_verified",
    ):
        assert good[field] is False
    assert good["witness"]["coefficient_dual"]["oscillation_optimality_claim"] is False
    assert "not the actual" in good["witness"]["reference_measure"]

