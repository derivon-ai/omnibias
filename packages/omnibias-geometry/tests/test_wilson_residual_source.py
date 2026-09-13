# SPDX-License-Identifier: Apache-2.0
"""Independent exact gates for original-edge Wilson residual identities."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.wilson_residual_source import (
    replay_su2_wilson_residual_vacuum_certificate as replay,
)
from omnibias.geometry.gauge.transfer.wilson_residual_source import (
    su2_wilson_residual_vacuum as certify,
)


def _good() -> dict[str, Any]:
    return certify(11, family="strip", correction_radius=Q(1, 12))


@pytest.mark.parametrize("family,kappa,base,radius", [
    ("strip", Q(11), Q(1), Q(1, 12)),
    ("strip", Q(12), Q(17, 16), Q(1, 12)),
    ("cubic", Q(17), Q(1), Q(1, 12)),
])
def test_exact_source_and_selected_radius(family: str, kappa: Q, base: Q, radius: Q) -> None:
    result = certify(kappa, family=family, correction_radius=radius, decay_base=base)
    a = result["witness"]["arithmetic"]
    g, quadratic = 4 / kappa**2, Q(4, 3)
    seed = (8 if family == "strip" else 16) * g * base**2
    residual = (6 * base**2 + 8 * base**3 if family == "strip"
                else 12 * base**2 + 112 * base**3) * g**2
    linear = 2 * quadratic * seed
    discriminant = (1 - linear)**2 - 4 * quadratic * residual
    slack = radius - residual - linear * radius - quadratic * radius**2
    contraction = linear + 2 * quadratic * radius
    assert Q(a["seed_norm_upper"]) == seed
    assert Q(a["residual_norm_upper"]) == residual
    assert Q(a["linear_upper"]) == linear
    assert Q(a["quadratic_constant"]) == quadratic
    assert Q(a["radius_feasibility_discriminant"]) == discriminant > 0
    assert Q(a["selected_correction_radius"]) == radius
    assert Q(a["self_map_slack"]) == slack >= 0
    assert Q(a["contraction_upper"]) == contraction < 1
    assert a["source_radius_exists_for_criterion"] is True
    assert a["fixed_point_verified"] is True
    assert a["old_source_radius_exists_for_criterion"] is False
    assert result["actual_vacuum_verified"] is True
    assert result["status"] == "PASS"
    assert replay(result["certificate"])


def test_strip_kappa11_near_boundary_numbers_are_exact() -> None:
    a = _good()["witness"]["arithmetic"]
    assert a["seed_norm_upper"] == "32/121"
    assert a["residual_norm_upper"] == "224/14641"
    assert a["linear_upper"] == "256/363"
    assert a["radius_feasibility_discriminant"] == "697/131769"
    assert a["self_map_slack"] == "2/395307"
    assert a["contraction_upper"] == "1010/1089"


@pytest.mark.parametrize("family,kappa,base", [("strip", Q(11), Q(1)), ("strip", Q(12), Q(17, 16)),
                                             ("cubic", Q(17), Q(1))])
def test_actual_conditional_and_curvature_comparisons_independent_constants(
    family: str, kappa: Q, base: Q,
) -> None:
    radius, steps, g, alpha = Q(1, 12), 4, 4 / kappa**2, kappa / 2
    result = certify(kappa, family=family, correction_radius=radius, decay_base=base, exponent_steps=steps)
    a = result["witness"]["arithmetic"]
    cap = 2 if family == "strip" else 4
    # One incident fundamental has range four, coefficient g/3 in S,
    # and the actual density is exp(2S). The correction contributes8r/3.
    omega = 2 * 4 * cap * g / 3 + 8 * radius / 3
    exponential = (1 - omega / steps)**steps
    gamma = Q(3, 4) * exponential
    if family == "strip":
        # Two adjacent vertical mixed blocks; inherited site distance is
        # one less than line-graph distance for every distinct vertical pair.
        mixed = 2 * (g / 6) * base + 2 * radius / (3 * base)
        seed_full_row = 4 * (g / 6)
    else:
        # Per incident square: two distance-one neighbors and one opposite.
        mixed = cap * (g / 6) * (2 * base + base**2) + 2 * radius / 3
        seed_full_row = cap * 4 * (g / 6)
    conditional_margin = gamma - 2 * mixed
    curvature = Q(1, 2) - 2 * (seed_full_row + 2 * radius / 3)
    assert Q(a["conditional_log_density_oscillation_upper"]) == omega
    assert Q(a["exp_negative_rational_lower"]) == exponential
    assert Q(a["conditional_poincare_lower"]) == gamma
    assert Q(a["weighted_mixed_hessian_row_upper"]) == mixed
    assert Q(a["conditional_comparison_margin_lower"]) == conditional_margin > 0
    assert Q(a["conditional_physical_gap_lower"]) == alpha * conditional_margin
    assert Q(a["curvature_lower"]) == curvature > 0
    assert Q(a["curvature_physical_gap_lower"]) == alpha * curvature
    assert Q(result["physical_gap_lower"]) == max(alpha * conditional_margin, alpha * curvature)
    assert Q(a["weighted_covariance_kernel_row_upper"]) * conditional_margin == 1
    assert result["actual_conditional_hierarchy_verified"] is True
    assert result["spatial_exponential_covariance_bound_verified"] is (base > 1)


def test_curvature_route_survives_failed_conditional_dominance() -> None:
    result = certify(100, family="strip", correction_radius=Q(1, 3))
    a = result["witness"]["arithmetic"]
    assert result["actual_vacuum_verified"] is True
    assert result["actual_conditional_poincare_verified"] is True
    assert Q(a["conditional_comparison_margin_lower"]) < 0
    assert result["actual_conditional_hierarchy_verified"] is False
    assert a["conditional_physical_gap_lower"] is None
    assert result["physical_gap_lower"] == a["curvature_physical_gap_lower"] == "619/225"
    assert result["status"] == "PASS"
    assert replay(result["certificate"])


def test_rational_exponential_refinement_does_not_change_source_proof() -> None:
    gammas = []
    first_source = None
    for steps in (1, 2, 4, 8):
        result = certify(11, family="strip", correction_radius=Q(1, 12), exponent_steps=steps)
        a = result["witness"]["arithmetic"]
        gammas.append(Q(a["conditional_poincare_lower"]))
        source = tuple(a[key] for key in ("seed_norm_upper", "residual_norm_upper", "linear_upper",
                                          "self_map_slack", "contraction_upper", "fixed_point_verified"))
        if first_source is None:
            first_source = source
        assert source == first_source
        assert replay(result["certificate"])
    assert all(x < y for x, y in zip(gammas, gammas[1:], strict=False))


@pytest.mark.parametrize("family,kappa,base", [("strip", 11, Q(1)), ("strip", 12, Q(17, 16)),
                                             ("cubic", 17, Q(1)), ("cubic", 24, Q(2))])
def test_automatic_radius_uses_rational_discriminant_witness(family: str, kappa: int, base: Q) -> None:
    result = certify(kappa, family=family, decay_base=base)
    a = result["witness"]["arithmetic"]
    residual, linear, quadratic = Q(a["residual_norm_upper"]), Q(a["linear_upper"]), Q(a["quadratic_constant"])
    delta = (1 - linear)**2 - 4 * quadratic * residual
    assert result["witness"]["inputs"]["correction_radius"] is None
    if linear < 1 and delta > 0:
        radius = 2 * residual / (1 - linear)
        assert Q(a["selected_correction_radius"]) == radius
        assert Q(a["self_map_slack"]) == residual * delta / (1 - linear)**2 > 0
        assert Q(a["contraction_upper"]) < 1
        assert result["actual_vacuum_verified"] is True
    else:
        assert result["actual_vacuum_verified"] is False
        assert result["status"] == "INCONCLUSIVE"
    assert replay(result["certificate"])


@pytest.mark.parametrize("family,kappa,base", [("strip", 10, Q(1)), ("strip", 11, Q(17, 16)),
                                             ("cubic", 16, Q(1)), ("strip", 1, Q(1))])
def test_failed_source_does_not_earn_actual_vacuum_or_physical_gap(family: str, kappa: int, base: Q) -> None:
    result = certify(kappa, family=family, decay_base=base)
    a = result["witness"]["arithmetic"]
    assert a["source_radius_exists_for_criterion"] is False
    assert a["fixed_point_verified"] is False
    assert result["actual_vacuum_verified"] is False
    assert result["actual_conditional_hierarchy_verified"] is False
    assert result["physical_gap_lower"] is None
    assert result["status"] == "INCONCLUSIVE"
    assert a["conditional_poincare_lower"] is None
    assert a["weighted_mixed_hessian_row_upper"] is None
    assert a["curvature_lower"] is None
    assert replay(result["certificate"])


def test_unverified_caller_residual_cannot_be_a_source_input() -> None:
    with pytest.raises(TypeError):
        certify(11, residual_norm_upper=Q(0))  # type: ignore[call-arg]


@pytest.mark.parametrize("radius", [Q(1, 10000), Q(1)])
def test_bad_selected_radius_is_not_replaced_by_a_passing_automatic_radius(radius: Q) -> None:
    result = certify(11, family="strip", correction_radius=radius)
    a = result["witness"]["arithmetic"]
    assert a["source_radius_exists_for_criterion"] is True
    assert Q(a["selected_correction_radius"]) == radius
    assert a["fixed_point_verified"] is False
    assert result["status"] == "INCONCLUSIVE"
    assert result["physical_gap_lower"] is None
    assert replay(result["certificate"])


@pytest.mark.parametrize("kwargs", [
    {"kappa": True}, {"kappa": 11.0}, {"kappa": "11"}, {"kappa": 0}, {"kappa": -1},
    {"family": "su3"}, {"family": ""}, {"family": None},
    {"correction_radius": True}, {"correction_radius": 0.125}, {"correction_radius": "1/12"},
    {"correction_radius": 0}, {"correction_radius": -1},
    {"decay_base": True}, {"decay_base": 1.0}, {"decay_base": "1"}, {"decay_base": Q(1, 2)},
    {"exponent_steps": True}, {"exponent_steps": 4.0}, {"exponent_steps": Q(4)},
    {"exponent_steps": 0}, {"exponent_steps": -1},
])
def test_strict_exact_inputs(kwargs: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {"kappa": 11, "family": "strip", "correction_radius": Q(1, 12)}
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        certify(**arguments)


@pytest.mark.parametrize("value", [None, [], {}, 1, "certificate", {"payload": None}])
def test_malformed_replay_returns_false(value: Any) -> None:
    assert replay(value) is False


@pytest.mark.parametrize("key", ["seed_norm_upper", "residual_norm_upper", "linear_upper", "quadratic_constant",
                               "radius_feasibility_discriminant", "self_map_slack", "contraction_upper",
                               "selected_correction_radius", "conditional_physical_gap_lower",
                               "curvature_physical_gap_lower"])
def test_resealed_arithmetic_forgery_is_rejected(key: str) -> None:
    certificate = deepcopy(_good()["certificate"])
    certificate["payload"]["witness"]["arithmetic"][key] = "999"
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay(certificate)


@pytest.mark.parametrize("key,value", [("family", "cubic"), ("kappa", "12"),
                                      ("correction_radius", None), ("decay_base", "17/16"),
                                      ("exponent_steps", 8)])
def test_resealed_inputs_are_recomputed(key: str, value: Any) -> None:
    certificate = deepcopy(_good()["certificate"])
    certificate["payload"]["witness"]["inputs"][key] = value
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("field", ["original_edge_tensor_budgets", "exact_centered_residual",
                                  "structural_caps", "orientation", "remainder_scope", "normalization"])
def test_resealed_physical_source_identity_or_scope_rewrites_are_rejected(field: str) -> None:
    certificate = deepcopy(_good()["certificate"])
    witness = certificate["payload"]["witness"]
    if field == "original_edge_tensor_budgets":
        witness[field]["fundamental_square"]["nuclear_norm"] = "16"
    elif field == "structural_caps":
        witness[field]["adjacent_plaquette_pairs_touching_edge"] = 42
    else:
        witness[field] = "unverified replacement"
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay(certificate)


def test_failed_source_cannot_be_promoted_by_resealing_earned_flags() -> None:
    certificate = deepcopy(certify(10)["certificate"])
    certificate["honesty"]["actual_vacuum_verified"] = True
    certificate["honesty"]["actual_conditional_hierarchy_verified"] = True
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("key", ["continuum_claim", "yang_mills_mass_gap_claim", "infinite_volume_claim",
                               "uniform_in_a_claim", "all_scale_refinement_claim"])
def test_parent_flags_stay_false_and_resealed_promotions_fail(key: str) -> None:
    result = _good()
    assert result[key] is False
    certificate = deepcopy(result["certificate"])
    certificate["honesty"][key] = True
    assert not replay(seal_certificate(certificate))


def test_formal_tiers_remain_unearned() -> None:
    result = _good()
    assert result["theorem_prover_verified"] is False
    assert result["mathlib_verified"] is False


def test_strong_coupling_family_does_not_claim_the_weak_coupling_limit() -> None:
    strong = _good()
    weak = certify(Q(1, 100), family="strip")
    assert strong["beyond_legacy_source_criterion_verified"] is True
    assert strong["actual_vacuum_verified"] is True
    assert "fixed microscopic kappa" in strong["witness"]["energy_units"]
    assert "no representation cutoff" in strong["witness"]["gauss_constraint"]
    assert "not assumed invariant under edge inversion" in strong["witness"]["orientation"]
    assert weak["status"] == "INCONCLUSIVE"
    assert weak["actual_vacuum_verified"] is False
    assert weak["physical_gap_lower"] is None
    for result in (strong, weak):
        assert result["uniform_in_a_claim"] is False
        assert result["all_scale_refinement_claim"] is False
        assert result["continuum_claim"] is False
        assert result["yang_mills_mass_gap_claim"] is False


def test_public_transfer_exports() -> None:
    from omnibias.geometry.gauge import transfer

    assert transfer.su2_wilson_residual_vacuum is certify
    assert transfer.replay_su2_wilson_residual_vacuum_certificate is replay
