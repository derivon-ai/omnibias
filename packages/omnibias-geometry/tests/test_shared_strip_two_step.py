# SPDX-License-Identifier: Apache-2.0
"""Independent exact checks for nested actual-marginal strip refinement."""

from __future__ import annotations

import copy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import shared_strip_two_step as module
from omnibias.geometry.gauge.transfer.shared_strip_refinement import su2_shared_strip_refinement
from omnibias.geometry.gauge.transfer.shared_strip_two_step import (
    replay_su2_shared_strip_two_step_certificate as replay,
)
from omnibias.geometry.gauge.transfer.shared_strip_two_step import (
    su2_shared_strip_two_step as certify,
)


def _good(**kwargs: Any) -> dict[str, Any]:
    return certify(19, correction_radius=Q(1, 9), **kwargs)


def _passes(a: Q, h: Q, beta2: Q, target: Q) -> bool:
    return 0 < target < min(a, h) and (a - target) * (h - target) >= beta2 * a


def _bisect(a: Q, h: Q, beta2: Q, bits: int = 40) -> Q:
    """Exact determinant bisection; independent of production square roots."""
    denominator = 1 << bits
    lo, hi = 0, int(min(a, h) * denominator) + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _passes(a, h, beta2, Q(mid, denominator)):
            lo = mid
        else:
            hi = mid
    return Q(lo, denominator)


def _check_step(step: dict[str, Any]) -> None:
    a = Q(step["compression_gap_lower"])
    h = Q(step["all_physical_fiber_modes_gap_lower"])
    beta2 = Q(step["relative_cross_form_beta_squared_upper"])
    z = Q(step["gap_lower"])
    discriminant = (a - h)**2 + 4 * a * beta2
    assert Q(step["discriminant"]) == discriminant
    assert Q(step["sqrt_upper"])**2 >= discriminant
    assert z == (a + h - Q(step["sqrt_upper"])) / 2
    assert _passes(a, h, beta2, z)


def test_kappa19_independent_constants_and_two_exact_schur_steps() -> None:
    r = _good()
    a = r["witness"]["arithmetic"]
    alpha, g, radius = Q(19, 2), Q(4, 361), Q(1, 9)
    gamma1 = Q(3, 4) * (1 - (32 * g + 16 * radius) / 24)**8
    gamma2 = Q(3, 4) * (1 - (16 * g + 8 * radius) / 24)**8
    covariance = 2 * (g / 6 + 2 * radius / 3)**2 / gamma1
    drift = 10 * radius / 3 + 20 * (g / 6 + 2 * radius / 3)**2 / gamma1
    second_coarse = Q(15, 2) * alpha * (1 - (g + radius) / 3)**8
    second_fiber = Q(7, 2) * alpha * gamma2
    second_beta2 = 2 * alpha * drift**2 / (5 * gamma2)
    first_fiber = Q(11, 5) * alpha * gamma1
    first_beta2 = 4 * alpha * (7 * g / 6 + 4 * radius / 3)**2 / (5 * gamma1)
    assert Q(a["first_conditional_pair_poincare_lower"]) == gamma1
    assert Q(a["second_conditional_loop_poincare_lower"]) == gamma2
    assert Q(a["conditional_mixed_covariance_upper"]) == covariance
    assert Q(a["marginal_drift_Z_derivative_upper"]) == drift
    assert Q(a["second_fiber_log_oscillation_upper"]) == Q(3464, 9747)
    assert Q(a["outer_marginal_log_oscillation_upper"]) == Q(3176, 9747)
    second = r["witness"]["second_step"]
    first = r["witness"]["composed_first_step"]
    assert Q(second["compression_gap_lower"]) == second_coarse
    assert Q(second["all_physical_fiber_modes_gap_lower"]) == second_fiber
    assert Q(second["relative_cross_form_beta_squared_upper"]) == second_beta2
    assert first["compression_gap_lower"] == second["gap_lower"]
    assert Q(first["all_physical_fiber_modes_gap_lower"]) == first_fiber
    assert Q(first["relative_cross_form_beta_squared_upper"]) == first_beta2
    _check_step(second)
    _check_step(first)
    assert _passes(second_coarse, second_fiber, second_beta2, Q(12))
    assert _passes(Q(12), first_fiber, first_beta2, Q(6))
    intermediate = _bisect(second_coarse, second_fiber, second_beta2)
    final = _bisect(intermediate, first_fiber, first_beta2)
    assert intermediate <= Q(r["intermediate_physical_gap_lower"]) < intermediate + Q(1, 2**40)
    # The first Schur root is increasing and 1-Lipschitz in its coarse floor.
    assert final <= Q(r["physical_gap_lower"]) < final + Q(1, 2**39)
    assert Q(1267, 100) < Q(r["intermediate_physical_gap_lower"]) < Q(12671, 1000)
    assert Q(6338, 1000) < Q(r["physical_gap_lower"]) < Q(6339, 1000)
    assert r["status"] == "PASS"
    assert replay(r["certificate"])


def test_composition_preserves_the_actual_marginal_and_anisotropic_units() -> None:
    r = _good()
    w = r["witness"]
    assert w["graph"]["n_vertices"] == 10
    assert len(w["graph"]["edges"]) == 13
    assert w["intermediate_electric_weights"] == ["5", "5", "1"]
    assert w["outer_loop_electric_weight"] == "10"
    assert w["second_physical_fiber_kinetic_floor"] == "7/2"
    assert "original dimensionless aH at both reductions" in w["energy_units"]
    assert "same actual fine density" in w["marginal_consistency"]
    assert "J01*J12(F)=psi0*F/phi2" in w["nested_haar_isometries"]
    assert "as closed forms" in w["compression_consistency"]
    # This certificate closes another actual reduction; its conservative fine
    # gap need not improve the earlier single-step comparison.
    source = su2_shared_strip_refinement(19, correction_radius=Q(1, 9))
    assert Q(r["physical_gap_lower"]) < Q(source["physical_gap_lower"])


def test_old_compression_full_gap_and_joint_log_ball_are_not_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _good()

    def stripped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        source = su2_shared_strip_refinement(*args, **kwargs)
        source["witness"] = copy.deepcopy(source["witness"])
        source["status"] = "INCONCLUSIVE"
        source["finite_gate_verified"] = False
        for key in ("physical_gap_lower", "physical_finite_graph_gap_verified",
                    "actual_joint_log_ball_verified", "actual_joint_density_bounds_verified",
                    "actual_joint_to_own_marginals_enclosed", "actual_joint_to_own_marginals_A2_upper"):
            source.pop(key, None)
        source["witness"].pop("joint_fourier")
        inherited = source["witness"]["arithmetic"]
        for key in ("compression_gap_lower", "physical_gap_lower", "gap_candidate_lower",
                    "schur_discriminant", "schur_sqrt_upper"):
            inherited.pop(key)
        # Keep the genuine source certificate unchanged. This strips only
        # unused summaries; no invented source certificate is being accepted.
        return source

    monkeypatch.setattr(module, "su2_shared_strip_refinement", stripped)
    assert _good() == expected


def test_second_step_can_pass_when_the_source_log_ball_is_inconclusive() -> None:
    source = su2_shared_strip_refinement(19, correction_radius=Q(1, 9), exponent_steps=1)
    assert not source["actual_joint_log_ball_verified"]
    r = _good(exponent_steps=1)
    assert r["status"] == "PASS"
    assert r["actual_marginal_derivative_bound_verified"]
    _check_step(r["witness"]["second_step"])
    _check_step(r["witness"]["composed_first_step"])


@pytest.mark.parametrize("kappa,radius,steps,failure,nested,derivative", [
    (18, Q(1, 9), 8, "actual_vacuum_fixed_point", False, False),
    (64, Q(1, 4), 1, "first_actual_conditional_estimates", True, False),
    (64, Q(1, 4), 8, "second_relative_form_or_gap_precision", True, True),
])
def test_refusal_preserves_exactly_the_established_stages(
    kappa: int, radius: Q, steps: int, failure: str, nested: bool, derivative: bool,
) -> None:
    r = certify(kappa, correction_radius=radius, exponent_steps=steps)
    assert r["status"] == "INCONCLUSIVE"
    assert failure in r["witness"]["failed_constraints"]
    assert r["actual_nested_marginals_verified"] is nested
    assert r["actual_marginal_derivative_bound_verified"] is derivative
    assert not r["intermediate_physical_gap_verified"]
    assert r["physical_gap_lower"] is None
    assert r["witness"]["composed_first_step"] is None
    assert replay(r["certificate"])


@pytest.mark.parametrize("offset,passes", [(Q(-1, 10**30), False), (Q(0), True),
                                           (Q(1, 10**30), True)])
def test_exact_source_boundary(offset: Q, passes: bool) -> None:
    r = certify(Q(64, 3), correction_radius=Q(3, 64) + offset)
    assert r["actual_nested_marginals_verified"] is passes
    assert r["two_step_refinement_verified"] is passes
    assert replay(r["certificate"])


def test_sqrt_precision_changes_bounds_without_breaking_validity() -> None:
    r0, r8, r64 = (_good(sqrt_bits=n) for n in (0, 8, 64))
    for key in ("intermediate_physical_gap_lower", "physical_gap_lower"):
        assert Q(r0[key]) <= Q(r8[key]) <= Q(r64[key])
    for r in (r0, r8, r64):
        _check_step(r["witness"]["second_step"])
        _check_step(r["witness"]["composed_first_step"])


def test_independent_three_block_hamiltonian_schur_composition() -> None:
    # Excited-space K0=[[12,2,1],[2,8,1],[1,1,5]], K1 is its upper
    # 2x2 compression, K2=12. Adjoin a zero vacuum block if desired.
    # The second relative bound is 2²/12; first is v^T K1^-1 v=4/23.
    assert _passes(Q(12), Q(8), Q(1, 3), Q(7))
    assert _passes(Q(7), Q(5), Q(4, 23), Q(4))
    # Direct exact Sylvester check on K0-4I, independent of both gates.
    diagonal = (Q(8), Q(4), Q(1))
    a, b, c = diagonal
    d, e, f = Q(2), Q(1), Q(1)
    assert a > 0
    assert a * b - d**2 > 0
    assert a * (b * c - f**2) - d * (d * c - e * f) + e * (d * f - b * e) == 20
    # A compatible coarse compression cannot see a new low fiber state.
    for n in (10, 100, 1000):
        epsilon = Q(1, n)
        assert not _passes(Q(8), epsilon, Q(0), Q(4))
        assert (Q(12) - 4) * (Q(8) - 4) * (epsilon - 4) < 0


@pytest.mark.parametrize("path,value", [
    (("arithmetic", "conditional_mixed_covariance_upper"), "0"),
    (("arithmetic", "marginal_drift_Z_derivative_upper"), "0"),
    (("arithmetic", "second_fiber_log_oscillation_upper"), "0"),
    (("arithmetic", "outer_marginal_log_oscillation_upper"), "0"),
    (("second_step", "gap_lower"), "100"),
    (("second_step", "relative_cross_form_beta_squared_upper"), "0"),
    (("second_step", "sqrt_upper"), "0"),
    (("composed_first_step", "compression_gap_lower"), "100"),
    (("composed_first_step", "gap_lower"), "100"),
    (("composed_first_step", "all_physical_fiber_modes_gap_lower"), "100"),
    (("graph", "n_vertices"), 9),
    (("inputs", "kappa"), "20"),
    (("inputs", "exponent_steps"), 2),
    (("inputs", "sqrt_bits"), 4),
    (("intermediate_electric_weights",), ["3", "3", "1"]),
    (("outer_loop_electric_weight",), "6"),
])
def test_resealed_witness_tampering_rejected(path: tuple[str, ...], value: Any) -> None:
    c = copy.deepcopy(_good()["certificate"])
    data = c["payload"]["witness"]
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value
    c = seal_certificate(c)
    assert verify_certificate_digest(c)
    assert not replay(c)


def test_nested_source_and_parent_flags_cannot_be_forged_by_resealing() -> None:
    c = copy.deepcopy(_good()["certificate"])
    s = c["payload"]["witness"]["source_certificate"]
    s["payload"]["witness"]["arithmetic"]["conditional_pair_poincare_lower"] = "100"
    c["payload"]["witness"]["source_certificate"] = seal_certificate(s)
    c = seal_certificate(c)
    assert verify_certificate_digest(c)
    assert verify_certificate_digest(c["payload"]["witness"]["source_certificate"])
    assert not replay(c)


@pytest.mark.parametrize("key", ["all_scale_refinement_claim", "continuum_claim",
                                 "yang_mills_mass_gap_claim", "theorem_prover_verified",
                                 "independent_cell_replacement_verified"])
def test_resealed_honesty_tampering_rejected(key: str) -> None:
    c = copy.deepcopy(_good()["certificate"])
    c["honesty"][key] = True
    c = seal_certificate(c)
    assert verify_certificate_digest(c)
    assert not replay(c)


@pytest.mark.parametrize("bad", [None, [], True, 1, "", {}, {"digest": "sha256:bad"}])
def test_malformed_replay_refuses(bad: Any) -> None:
    assert not replay(bad)


@pytest.mark.parametrize("key,value", [
    ("kappa", True), ("kappa", 19.0), ("kappa", "19"), ("kappa", 0),
    ("correction_radius", False), ("correction_radius", 0.1), ("correction_radius", 0),
    ("exponent_steps", True), ("exponent_steps", 8.0), ("exponent_steps", 0),
    ("sqrt_bits", True), ("sqrt_bits", 64.0), ("sqrt_bits", -1),
])
def test_strict_exact_input_guards(key: str, value: Any) -> None:
    arguments: dict[str, Any] = {
        "kappa": 19, "correction_radius": Q(1, 9), "exponent_steps": 8, "sqrt_bits": 64,
    }
    arguments[key] = value
    with pytest.raises((TypeError, ValueError)):
        certify(**arguments)


def test_public_exports_and_scope_flags() -> None:
    import omnibias.geometry.gauge.transfer as transfer

    assert transfer.su2_shared_strip_two_step is certify
    assert transfer.replay_su2_shared_strip_two_step_certificate is replay
    r = _good()
    for key in ("independent_cell_replacement_verified", "all_scale_refinement_claim",
                "coarse_wilson_family_closed", "infinite_volume_claim", "uniform_in_a_claim",
                "continuum_claim", "yang_mills_claim", "yang_mills_mass_gap_claim",
                "theorem_prover_verified", "mathlib_verified"):
        assert r[key] is False
