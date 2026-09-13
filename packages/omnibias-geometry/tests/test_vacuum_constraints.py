# SPDX-License-Identifier: Apache-2.0
"""Exact real radius decisions, independently checked algebra and replay scope."""

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_family,
)
from omnibias.geometry.gauge.transfer.vacuum_constraints import (
    _pair_mul,
    _pair_pow,
    _pair_sign,
    _rational_root,
    replay_vacuum_constraint_certificate,
    solve_vacuum_constraints,
)


def _solve(**changes):
    parameters = dict(group="su3", kappa=45, target_gap=Q(298, 45))
    parameters.update(changes)
    return solve_vacuum_constraints(**parameters)


def test_rational_root_attains_exact_target_boundary():
    report = _solve()
    assert report["status"] == "FEASIBLE"
    assert report["constraint_decision_verified"]
    assert report["feasible"]
    assert report["target_gap_lower"] == "298/45"
    assert replay_vacuum_constraint_certificate(report["certificate"])
    w = report["witness"]
    assert w["root_rational"] == "32/75"
    assert w["radius_root"] == ["52/75", "-4/3", "1/25"]
    assert w["target_bound_rational"] == "298/45"
    assert w["checks"]["fixed_point_slack"]["sign"] == 0
    assert w["checks"]["target_gap_margin"]["sign"] == 0
    assert w["checks"]["target_gap_margin"]["relation"] == ">=0"
    assert w["checks"]["strict_contraction_margin"]["sign"] == 1
    assert w["target_bound_domain_verified"]
    assert w["failed_constraints"] == []
    # An independent direct evaluation at the real root is entirely rational.
    r = Q(32, 75)
    assert Q(3, 8) * (Q(16, 25) + r) ** 2 == r
    assert Q(1) - Q(2, 75) - Q(3, 2) * r == Q(1, 3)
    assert Q(4, 225) + Q(3, 4) * r == Q(76, 225)
    assert Q(30) * Q(1, 3) * (1 - Q(76, 225)) == Q(298, 45)


def test_target_infinitesimally_above_root_bound_is_infeasible_not_gapless():
    report = _solve(target_gap=Q(298, 45) + Q(1, 10**80))
    assert report["status"] == "INFEASIBLE_CRITERION"
    assert report["constraint_decision_verified"]
    assert not report["feasible"]
    assert not report["finite_gate_verified"]
    assert not report["volume_uniform_target_gap_verified"]
    assert report["target_gap_lower"] == "0"
    assert report["witness"]["failed_constraints"] == ["target_gap_margin"]
    assert report["witness"]["checks"]["target_gap_margin"]["sign"] == -1
    assert report["witness"]["target_bound_domain_verified"]
    assert report["gap_absence_claim"] is False
    assert report["certificate"]["honesty"]["gap_absence_claim"] is False
    # Replay verifies the negative decision about this criterion itself.
    assert replay_vacuum_constraint_certificate(report["certificate"])


def test_irrational_root_is_decided_without_a_rational_grid():
    sp = pytest.importorskip("sympy")
    report = _solve(group="su2", kappa=27, target_gap=4, gap_method="curvature")
    assert report["feasible"]
    w = report["witness"]
    assert w["radius_root"] == ["1163/5832", "-3/8", "139/2187"]
    assert w["root_rational"] is None
    assert w["target_bound_rational"] is None
    assert w["target_bound_pair"] == ["224/81", "27/4"]
    x, y, discriminant = map(sp.Rational, w["radius_root"])
    r = x + y * sp.sqrt(discriminant)
    assert sp.simplify(sp.Rational(4, 3) * (sp.Rational(128, 729) + r) ** 2 - r) == 0
    assert sp.sign(r) == 1
    assert sp.sign(sp.Rational(1, 9) - r) == 1
    exact_bound = sp.Rational(224, 81) + sp.Rational(27, 4) * sp.sqrt(discriminant)
    assert sp.sign(exact_bound - 4) == 1
    assert replay_vacuum_constraint_certificate(report["certificate"])
    # A known rational witness independently confirms feasibility at this target.
    direct = invariant_vacuum_fourier_family("su2", 27, correction_radius=Q(1, 9))
    assert Q(direct["witness"]["arithmetic"]["neutral_gap_lower"]) > 4


@pytest.mark.parametrize(("kappa", "cap", "discriminant"), [(16, 4, Q(-5, 3)), (32, 6, Q(0))])
def test_nonpositive_discriminant_refuses_even_the_double_root(kappa, cap, discriminant):
    report = _solve(group="su2", kappa=kappa, target_gap=1, weighted_incidence_cap=cap)
    w = report["witness"]
    assert report["status"] == "INFEASIBLE_CRITERION"
    assert Q(w["derived_coefficients"]["fixed_point_discriminant"]) == discriminant
    assert w["radius_root"] is None
    assert w["root_rational"] is None
    assert w["target_bound_pair"] is None
    assert not w["target_bound_domain_verified"]
    assert w["failed_constraints"] == ["positive_fixed_point_discriminant"]
    assert replay_vacuum_constraint_certificate(report["certificate"])


def test_a_real_fixed_point_does_not_automatically_earn_factorization():
    report = _solve(kappa=Q(44091, 1000), target_gap=1)
    w = report["witness"]
    assert Q(w["derived_coefficients"]["fixed_point_discriminant"]) == Q(67, 8000067)
    assert w["radius_root"] is not None
    assert w["checks"]["fixed_point_slack"]["passed"]
    assert w["checks"]["strict_contraction_margin"]["passed"]
    assert not w["checks"]["influence_margin"]["passed"]
    assert not w["target_bound_domain_verified"]
    assert report["status"] == "INFEASIBLE_CRITERION"
    assert not report["volume_uniform_target_gap_verified"]
    assert replay_vacuum_constraint_certificate(report["certificate"])


@pytest.mark.parametrize("steps", [1, 2, 4, 7])
def test_selected_exponential_polynomial_has_an_explicit_rational_witness(steps):
    # This m is chosen by the caller, not floor(Omega)+1 from the older gate.
    direct_bound = 10 * (1 - Q(76, 225 * steps)) ** steps
    report = _solve(target_gap=direct_bound, exponent_steps=steps)
    w = report["witness"]
    assert report["feasible"]
    assert w["inputs"]["exponent_steps"] == steps
    assert w["factored_target_constraint"]["exponent_steps"] == steps
    assert Q(w["target_bound_rational"]) == direct_bound
    assert w["checks"]["exponent_margin"]["sign"] == 1
    assert w["checks"]["target_gap_margin"]["sign"] == 0
    assert "freely chosen positive integer" in w["exponent_scope"]
    assert replay_vacuum_constraint_certificate(report["certificate"])
    assert not _solve(target_gap=direct_bound + Q(1, 10**50), exponent_steps=steps)["feasible"]


def test_exponent_choice_can_change_the_decision_without_changing_the_vacuum():
    assert not _solve(target_gap=7, exponent_steps=1)["feasible"]
    improved = _solve(target_gap=7, exponent_steps=4)
    assert improved["feasible"]
    assert improved["witness"]["root_rational"] == "32/75"


@pytest.mark.parametrize(
    ("group", "kappa", "method", "norm"),
    [
        ("su2", 27, "curvature", "spin_weighted_N"),
        ("su3", 45, "factorization", "Casimir_weighted_M"),
    ],
)
def test_decision_keeps_coefficient_norm_physical_units_and_parent_scope(
    group, kappa, method, norm
):
    report = _solve(group=group, kappa=kappa, target_gap=1, gap_method=method)
    w = report["witness"]
    assert w["derived_coefficients"]["coefficient_norm"] == norm
    assert w["energy_units"] == "dimensionless aH"
    assert w["normalization"].startswith("aH=kappa/2*sum(C_e)")
    assert "all finite simple electric graphs" in w["family_class"]
    assert "at every vertex" in w["gauss_constraint"]
    assert "entire scalar product-group Hilbert space" in w["gap_scope"]
    assert "no dynamical fundamental matter" in w["charged_scope"]
    assert "radius-one probe is not gap evidence" in w["coefficient_source"]
    for flag in (
        "gap_absence_claim",
        "infinite_volume_claim",
        "uniform_in_a_claim",
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "static_confinement_claim",
        "string_tension_claim",
        "theorem_prover_verified",
        "mathlib_verified",
        "analytic_implication_formally_verified",
    ):
        assert report[flag] is False


def test_all_quadratic_surd_sign_branches_match_independent_exact_sympy():
    sp = pytest.importorskip("sympy")
    values = [Q(-3), Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1), Q(3)]
    discriminants = [Q(0), Q(1, 4), Q(1), Q(2, 3), Q(2), Q(9, 4)]
    signs = set()
    for x, y, d in product(values, values, discriminants):
        expected = int(
            sp.sign(
                sp.Rational(x.numerator, x.denominator)
                + sp.Rational(y.numerator, y.denominator)
                * sp.sqrt(sp.Rational(d.numerator, d.denominator))
            )
        )
        assert _pair_sign((x, y), d) == expected, (x, y, d)
        signs.add(expected)
    assert signs == {-1, 0, 1}
    # Exact cancellation is not rounded away, and perturbations keep their signs.
    assert _pair_sign((Q(3, 2), Q(-1)), Q(9, 4)) == 0
    assert _pair_sign((Q(3, 2) + Q(1, 10**80), Q(-1)), Q(9, 4)) == 1
    assert _pair_sign((Q(3, 2) - Q(1, 10**80), Q(-1)), Q(9, 4)) == -1
    with pytest.raises(ValueError):
        _pair_sign((Q(1), Q(1)), Q(-1))


def test_pair_multiplication_and_powers_match_symbolic_expansion():
    sp = pytest.importorskip("sympy")
    d = Q(5, 7)
    left, right = (Q(2, 3), Q(-4, 5)), (Q(-7, 4), Q(3, 2))

    def expression(pair):
        x, y = pair
        return sp.Rational(x.numerator, x.denominator) + sp.Rational(
            y.numerator, y.denominator
        ) * sp.sqrt(sp.Rational(5, 7))

    assert (
        sp.simplify(expression(_pair_mul(left, right, d)) - expression(left) * expression(right))
        == 0
    )
    for exponent in (0, 1, 2, 3, 4, 7, 12):
        assert (
            sp.simplify(expression(_pair_pow(left, exponent, d)) - expression(left) ** exponent)
            == 0
        )


def test_rational_part_is_recognized_even_in_an_irrational_quadratic_field():
    assert _rational_root((Q(7, 5), Q(0)), Q(2)) == "7/5"
    assert _rational_root((Q(1, 3), Q(2)), Q(9, 16)) == "11/6"
    assert _rational_root((Q(1, 3), Q(2)), Q(2)) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"kappa": True},
        {"kappa": 45.0},
        {"target_gap": False},
        {"target_gap": 1.0},
        {"target_gap": "1"},
        {"weighted_incidence_cap": True},
        {"weighted_incidence_cap": 4.0},
        {"decay_base": 1.0},
        {"exponent_steps": True},
        {"exponent_steps": Q(1)},
        {"exponent_steps": 1.0},
        {"minimum_girth": 4.0},
        {"max_cycle_length": True},
        {"max_cycle_diameter": Q(2)},
    ],
)
def test_exact_input_types_are_required(changes):
    with pytest.raises(TypeError):
        _solve(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"weighted_incidence_cap": 0},
        {"weighted_incidence_cap": -1},
        {"target_gap": 0},
        {"target_gap": -1},
        {"kappa": 0},
        {"exponent_steps": 0},
        {"exponent_steps": -1},
        {"minimum_girth": 2},
        {"max_cycle_length": 2},
        {"max_cycle_diameter": -1},
        {"decay_base": Q(1, 2)},
        {"group": "su4"},
        {"gap_method": "automatic"},
    ],
)
def test_invalid_ranges_and_methods_are_refused(changes):
    with pytest.raises(ValueError):
        _solve(**changes)


@pytest.mark.parametrize("value", [None, [], (), "certificate", 1, True, {}])
def test_replay_malformed_top_level_returns_false(value):
    assert replay_vacuum_constraint_certificate(value) is False


@pytest.mark.parametrize("target", [Q(1), Q(298, 45) + Q(1, 10**40)])
@pytest.mark.parametrize(
    "path",
    [
        ("honesty", "yang_mills_mass_gap_claim"),
        ("honesty", "continuum_claim"),
        ("honesty", "gap_absence_claim"),
        ("honesty", "theorem_prover_verified"),
        ("payload", "witness", "inputs", "kappa"),
        ("payload", "witness", "inputs", "target_gap"),
        ("payload", "witness", "radius_root"),
        ("payload", "witness", "root_rational"),
        ("payload", "witness", "target_bound_pair"),
        ("payload", "witness", "target_bound_rational"),
        ("payload", "witness", "target_bound_domain_verified"),
        ("payload", "witness", "checks", "target_gap_margin", "sign"),
        ("payload", "witness", "checks", "target_gap_margin", "passed"),
        ("payload", "witness", "derived_coefficients", "bilinear_B"),
        ("payload", "witness", "energy_units"),
        ("payload", "witness", "gauss_constraint"),
    ],
)
def test_rehashed_forgery_is_rejected_for_both_decisions(target, path):
    certificate = deepcopy(_solve(target_gap=target)["certificate"])
    node = certificate
    for key in path[:-1]:
        node = node[key]
    original = node.get(path[-1])
    node[path[-1]] = not original if type(original) is bool else "forged"
    if path[0] == "honesty":
        node[path[-1]] = True
    assert not replay_vacuum_constraint_certificate(certificate)
    forged = seal_certificate(certificate)
    assert verify_certificate_digest(forged)
    assert not replay_vacuum_constraint_certificate(forged)


@pytest.mark.parametrize(
    "path", [("payload",), ("payload", "witness"), ("payload", "witness", "inputs")]
)
def test_resealed_malformed_nested_inputs_do_not_throw(path):
    certificate = deepcopy(_solve()["certificate"])
    node = certificate
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = None
    assert not replay_vacuum_constraint_certificate(seal_certificate(certificate))
