# SPDX-License-Identifier: Apache-2.0
"""Actual cubic exterior conditionals, local budgets, and canonical replay."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.wilson_conditional_block import (
    _exp_floor,
)
from omnibias.geometry.gauge.transfer.wilson_conditional_block import (
    replay_su2_wilson_conditional_block_certificate as replay,
)
from omnibias.geometry.gauge.transfer.wilson_conditional_block import (
    su2_wilson_conditional_block as certify,
)
from omnibias.geometry.gauge.transfer.wilson_polar_source import su2_wilson_polar_vacuum
from omnibias.geometry.gauge.transfer.wilson_residual_source import (
    su2_wilson_linear_vacuum,
    su2_wilson_residual_vacuum,
)


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    return su2_wilson_polar_vacuum(15, family="cubic", correction_radius=Q(1, 10))


@pytest.fixture(scope="module")
def good(source: dict[str, Any]) -> dict[str, Any]:
    return certify(source["certificate"], block_size=7)


def test_new_source_and_every_exterior_quantifiers(
    source: dict[str, Any], good: dict[str, Any],
) -> None:
    assert source["actual_vacuum_verified"] is True
    assert good["witness"]["source_certificate"] == source["certificate"]
    assert good["status"] == "PASS"
    assert good["uniform_over_all_exteriors_verified"] is True
    assert good["volume_uniform_conditional_block_verified"] is True
    assert "every nonempty" in good["witness"]["block_quantifier"]
    assert "before integration or sampling" in good["witness"]["exterior_quantifier"]
    assert "boundary plaquettes" in good["witness"]["reference_conditional"]
    assert replay(good["certificate"])


def test_exact_block_local_budgets(good: dict[str, Any]) -> None:
    a = good["witness"]["arithmetic"]
    g, r, size = Q(4, 225), Q(1, 10), 7
    assert Q(a["g"]) == g
    assert Q(a["selected_correction_radius"]) == r
    assert a["touching_plaquettes_upper"] == 28
    assert Q(a["invariant_electric_floor"]) == 3
    assert Q(a["block_correction_log_oscillation_upper"]) == Q(8, 3) * size * r
    assert Q(a["block_seed_log_oscillation_upper"]) == Q(8, 3) * 28 * g
    assert Q(a["block_log_density_oscillation_upper"]) == Q(8, 3) * size * (4 * g + r)
    assert Q(a["single_edge_log_density_oscillation_upper"]) == Q(308, 675)
    assert Q(a["unweighted_mixed_hessian_row_upper"]) == Q(23, 225)


def test_cardinality_independent_curvature_and_schur_are_separate(good: dict[str, Any]) -> None:
    a = good["witness"]["arithmetic"]
    single = Q(31970155204, 69198046875)
    schur = single - Q(46, 225)
    assert Q(a["single_edge_poincare_lower"]) == single
    assert Q(a["conditional_schur_margin_lower"]) == schur > 0
    assert Q(a["conditional_schur_poincare_lower"]) == schur
    assert Q(a["conditional_curvature_poincare_lower"]) == Q(367, 1350) > schur
    assert Q(a["all_cardinalities_conditional_poincare_lower"]) == Q(367, 1350)
    assert Q(good["conditional_poincare_lower"]) == Q(367, 1350)
    assert Q(good["conditional_energy_units_lower"]) == Q(367, 180)
    assert good["all_cardinalities_conditional_gap_verified"] is True
    assert good["conditional_schur_gap_verified"] is True


def test_single_edge_floor_is_not_substituted_for_whole_volume(source: dict[str, Any]) -> None:
    result = certify(source["certificate"], block_size=1)
    a = result["witness"]["arithmetic"]
    assert Q(result["conditional_poincare_lower"]) == Q(31970155204, 69198046875)
    assert Q(a["all_cardinalities_conditional_poincare_lower"]) == Q(367, 1350)
    assert Q(result["conditional_poincare_lower"]) > Q(a["all_cardinalities_conditional_poincare_lower"])
    assert result["frozen_wilson_hamiltonian_claim"] is False
    assert result["witness"]["frozen_hamiltonian_identification"] is False


@pytest.mark.parametrize("size", [1, 2, 7, 64, 128])
def test_block_size_does_not_degrade_the_uniform_route(
    source: dict[str, Any], size: int,
) -> None:
    result = certify(source["certificate"], block_size=size)
    a = result["witness"]["arithmetic"]
    assert Q(a["all_cardinalities_conditional_poincare_lower"]) == Q(367, 1350)
    assert Q(result["conditional_poincare_lower"]) >= Q(367, 1350)
    steps = a["block_effective_exponent_steps"]
    omega = Q(a["block_log_density_oscillation_upper"])
    assert steps > omega >= 0
    assert Q(a["block_exp_negative_lower"]) == (1 - omega / steps)**steps > 0
    assert replay(result["certificate"])


def test_actual_vacuum_with_failed_global_gates_still_has_local_conditionals() -> None:
    source = su2_wilson_polar_vacuum(100, family="cubic", correction_radius=Q(1, 2))
    assert source["actual_vacuum_verified"] is True
    assert source["physical_gap_lower"] is None
    assert source["status"] == "INCONCLUSIVE"
    result = certify(source["certificate"], block_size=7, exponent_steps=1)
    a = result["witness"]["arithmetic"]
    assert result["status"] == "PASS"
    assert Q(a["original_coordinate_curvature_lower"]) < 0
    assert Q(a["conditional_schur_margin_lower"]) < 0
    assert a["all_cardinalities_conditional_poincare_lower"] is None
    assert result["all_cardinalities_conditional_gap_verified"] is False
    assert result["conditional_curvature_gap_verified"] is False
    assert result["conditional_schur_gap_verified"] is False
    assert a["block_effective_exponent_steps"] > Q(a["block_log_density_oscillation_upper"])
    assert Q(result["conditional_poincare_lower"]) == Q(a["direct_block_poincare_lower"]) > 0
    assert replay(result["certificate"])


def test_weighted_source_earns_tail_without_changing_original_metric() -> None:
    source = su2_wilson_polar_vacuum(
        15, family="cubic", correction_radius=Q(1, 8), decay_base=Q(33, 32),
    )
    result = certify(source["certificate"], block_size=7)
    assert result["status"] == "PASS"
    assert result["spatial_correction_tail_verified"] is True
    a = result["witness"]["arithmetic"]
    assert Q(a["decay_base"]) == Q(33, 32)
    assert Q(a["unweighted_mixed_hessian_row_upper"]) == Q(107, 900)
    assert Q(result["conditional_poincare_lower"]) == Q(161, 675)
    assert Q(result["conditional_energy_units_lower"]) == Q(161, 90)
    assert replay(result["certificate"])


def test_near_threshold_source_cannot_borrow_failed_schur_margin() -> None:
    source = su2_wilson_polar_vacuum(
        Q(141, 10), family="cubic", correction_radius=Q(23, 100),
    )
    result = certify(source["certificate"], block_size=7)
    assert source["actual_vacuum_verified"] is True
    assert result["conditional_schur_gap_verified"] is False
    assert Q(result["witness"]["arithmetic"]["conditional_schur_margin_lower"]) < 0
    assert result["conditional_curvature_gap_verified"] is True
    assert Q(result["conditional_energy_units_lower"]) == Q(256549, 423000)
    assert replay(result["certificate"])


@pytest.mark.parametrize("kind,kappa,radius", [
    ("residual", 17, Q(1, 12)),
    ("linear", 16, Q(1, 8)),
])
def test_old_sources_are_replayed_without_changing_their_schema(
    kind: str, kappa: int, radius: Q,
) -> None:
    make = su2_wilson_residual_vacuum if kind == "residual" else su2_wilson_linear_vacuum
    source = make(kappa, family="cubic", correction_radius=radius)
    result = certify(source["certificate"], block_size=7)
    assert result["witness"]["source_type"] == "su2_wilson_" + kind + "_vacuum_v1"
    assert result["actual_vacuum_verified"] is True
    assert replay(result["certificate"])


def test_failed_nonlinear_source_is_inconclusive() -> None:
    source = su2_wilson_polar_vacuum(14, family="cubic", correction_radius=Q(1, 10))
    assert source["actual_vacuum_verified"] is False
    result = certify(source["certificate"], block_size=7)
    assert result["status"] == "INCONCLUSIVE"
    assert result["conditional_poincare_lower"] is None
    assert result["actual_block_conditionals_verified"] is False
    assert result["all_cardinalities_conditional_gap_verified"] is False
    assert replay(result["certificate"])


def test_strip_chart_cannot_be_substituted_for_original_cubic_edges() -> None:
    source = su2_wilson_polar_vacuum(20, family="strip", correction_radius=Q(1, 10))
    assert source["actual_vacuum_verified"] is True
    with pytest.raises(ValueError, match="cubic original-edge"):
        certify(source["certificate"], block_size=7)


@pytest.mark.parametrize("omega,requested", [
    (Q(0), 1), (Q(1), 1), (Q(99, 100), 1), (Q(3, 2), 1),
    (Q(5), 4), (Q(1001, 100), 4), (Q(1, 2), 17),
])
def test_exact_exponential_floor_uses_strict_domain(omega: Q, requested: int) -> None:
    steps, value = _exp_floor(omega, requested)
    assert steps == max(requested, omega.numerator // omega.denominator + 1)
    assert steps > omega
    assert value == (1 - omega / steps)**steps > 0


def test_local_union_bound_independent_rational_regression() -> None:
    # Algebra regression for the proof: not a sampled certification of U.
    rng = Random(889173)
    for _ in range(64):
        modes = []
        for _ in range(20):
            support = rng.sample(range(12), rng.randint(4, 8))
            spins = [Q(rng.randint(1, 6), 2) if e in support else Q(0) for e in range(12)]
            energy = sum((j * (j + 1) for j in spins), Q(0))
            coefficient = Q(rng.randint(1, 8), rng.randint(1, 20))
            assert energy >= 3
            modes.append((spins, energy, coefficient))
        anchors = [sum((j[e] * energy * c for j, energy, c in modes), Q(0)) for e in range(12)]
        radius = max(anchors)
        block = set(rng.sample(range(12), rng.randint(1, 8)))
        local_l1 = sum((c for j, _, c in modes if any(j[e] for e in block)), Q(0))
        assert 4 * local_l1 <= Q(8, 3) * sum((anchors[e] for e in block), Q(0))
        assert 4 * local_l1 <= Q(8, 3) * len(block) * radius


@pytest.mark.parametrize("field", [
    "selected_correction_radius", "source_actual_vacuum_verified", "invariant_electric_floor",
    "plaquettes_per_edge_upper", "touching_plaquettes_upper",
    "block_correction_log_oscillation_upper", "block_seed_log_oscillation_upper",
    "block_log_density_oscillation_upper", "single_edge_log_density_oscillation_upper",
    "unweighted_mixed_hessian_row_upper", "original_coordinate_curvature_lower",
    "block_effective_exponent_steps", "single_edge_effective_exponent_steps",
    "block_exp_negative_lower", "single_edge_exp_negative_lower", "direct_block_poincare_lower",
    "single_edge_poincare_lower", "conditional_schur_margin_lower",
    "conditional_schur_poincare_lower", "conditional_curvature_poincare_lower",
    "all_cardinalities_conditional_poincare_lower", "conditional_poincare_lower",
    "conditional_energy_units_lower",
])
def test_rehashed_arithmetic_forgery_is_rejected(good: dict[str, Any], field: str) -> None:
    certificate = deepcopy(good["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = "123"
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay(certificate)


@pytest.mark.parametrize("field", [
    "family_scope", "coordinate_scope", "block_quantifier", "exterior_quantifier",
    "actual_conditional", "reference_conditional", "correction_source",
    "local_oscillation_lemma", "weighted_neighborhood_tail", "plaquette_count_lemma",
    "direct_comparison", "schur_comparison", "curvature_comparison", "energy_scaling",
    "frozen_hamiltonian_identification", "exponential_rule",
])
def test_rehashed_scope_forgery_is_rejected(good: dict[str, Any], field: str) -> None:
    certificate = deepcopy(good["certificate"])
    certificate["payload"]["witness"][field] = "unearned"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("field", [
    "actual_vacuum_verified", "actual_block_conditionals_verified",
    "uniform_over_all_exteriors_verified", "volume_uniform_conditional_block_verified",
    "all_cardinalities_conditional_gap_verified", "conditional_schur_gap_verified",
    "conditional_curvature_gap_verified", "spatial_correction_tail_verified",
    "actual_arbitrary_graph_membership_verified", "isolated_block_gap_substitution",
    "frozen_wilson_hamiltonian_claim", "all_scale_refinement_claim", "coarse_wilson_family_closed",
    "infinite_volume_claim", "uniform_in_a_claim", "continuum_claim",
    "yang_mills_claim", "yang_mills_mass_gap_claim", "theorem_prover_verified", "mathlib_verified",
])
def test_rehashed_honesty_forgery_is_rejected(good: dict[str, Any], field: str) -> None:
    certificate = deepcopy(good["certificate"])
    certificate["honesty"][field] = not certificate["honesty"].get(field, False)
    assert not replay(seal_certificate(certificate))


def test_twice_resealed_source_forgery_is_rejected(good: dict[str, Any]) -> None:
    certificate = deepcopy(good["certificate"])
    witness = certificate["payload"]["witness"]
    parent = witness["source_certificate"]
    parent["payload"]["witness"]["arithmetic"]["selected_correction_radius"] = "1/1000000"
    witness["source_certificate"] = seal_certificate(parent)
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("parameter", ["block_size", "exponent_steps"])
@pytest.mark.parametrize("value", [0, -1, True, False, Q(3, 2), 1.0, "4", None])
def test_input_counts_are_exact_positive_integers(
    source: dict[str, Any], parameter: str, value: Any,
) -> None:
    inputs = {"block_size": 7, "exponent_steps": 4, parameter: value}
    with pytest.raises((ValueError, TypeError)):
        certify(source["certificate"], **inputs)


@pytest.mark.parametrize("value", [None, [], {}, {"payload": "bad"}, {"payload": {"type": "unknown"}}])
def test_malformed_or_unsupported_sources_refuse(value: Any) -> None:
    with pytest.raises(ValueError, match="actual-source"):
        certify(value, block_size=7)
    assert not replay(value)


def test_parent_claims_remain_unearned(good: dict[str, Any]) -> None:
    for field in (
        "actual_arbitrary_graph_membership_verified", "isolated_block_gap_substitution",
        "frozen_wilson_hamiltonian_claim", "all_scale_refinement_claim",
        "coarse_wilson_family_closed", "infinite_volume_claim", "uniform_in_a_claim",
        "continuum_claim", "yang_mills_claim", "yang_mills_mass_gap_claim",
        "theorem_prover_verified", "mathlib_verified",
    ):
        assert good[field] is False

