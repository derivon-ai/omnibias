# SPDX-License-Identifier: Apache-2.0
"""Exact real feasibility of a finite invariant-vacuum radius criterion.

The chosen target is a lower bound in dimensionless Hamiltonian units.
An infeasible decision concerns this sufficient criterion, never absence
of a physical gap. Quadratic-field signs use rational arithmetic only.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from math import isqrt
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    _constants,
    invariant_vacuum_fourier_family,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

Pair = tuple[Q, Q]


def _pair_add(left: Pair, right: Pair) -> Pair:
    return left[0] + right[0], left[1] + right[1]


def _pair_scale(value: Pair, scale: Q) -> Pair:
    return value[0] * scale, value[1] * scale


def _pair_mul(left: Pair, right: Pair, discriminant: Q) -> Pair:
    return (
        left[0] * right[0] + discriminant * left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _pair_pow(value: Pair, exponent: int, discriminant: Q) -> Pair:
    power = _integer(exponent, "exponent")
    if power < 0:
        raise ValueError("the polynomial exponent must be nonnegative")
    result = (Q(1), Q(0))
    factor = value
    while power:
        if power & 1:
            result = _pair_mul(result, factor, discriminant)
        power >>= 1
        if power:
            factor = _pair_mul(factor, factor, discriminant)
    return result


def _pair_sign(value: Pair, discriminant: Q) -> int:
    """Sign of x+y*sqrt(D), including exact zero, without a floating root."""
    if discriminant < 0:
        raise ValueError("real quadratic-field signs require a nonnegative discriminant")
    x, y = value
    if discriminant == 0 or y == 0:
        return int(x > 0) - int(x < 0)
    if x == 0:
        return int(y > 0) - int(y < 0)
    if x * y > 0:
        return int(x > 0) - int(x < 0)
    comparison = x * x - discriminant * y * y
    return (int(x > 0) - int(x < 0)) * (
        int(comparison > 0) - int(comparison < 0)
    )


def _poly_at_pair(coefficients: Sequence[Q], value: Pair, discriminant: Q) -> Pair:
    result = (Q(0), Q(0))
    for coefficient in reversed(coefficients):
        result = _pair_add(_pair_mul(result, value, discriminant), (coefficient, Q(0)))
    return result


def _pair_payload(value: Pair) -> list[str]:
    return [str(value[0]), str(value[1])]


def _rational_root(value: Pair, discriminant: Q) -> str | None:
    if value[1] == 0:
        return str(value[0])
    numerator_root = isqrt(discriminant.numerator)
    denominator_root = isqrt(discriminant.denominator)
    if (
        numerator_root**2 != discriminant.numerator
        or denominator_root**2 != discriminant.denominator
    ):
        return None
    return str(value[0] + value[1] * Q(numerator_root, denominator_root))


def _constraint(value: Pair, discriminant: Q, *, strict: bool) -> dict[str, Any]:
    sign = _pair_sign(value, discriminant)
    return {
        "value_pair": _pair_payload(value),
        "sign": sign,
        "relation": ">0" if strict else ">=0",
        "passed": sign > 0 if strict else sign >= 0,
    }


def solve_vacuum_constraints(
    group: str,
    kappa: int | Q,
    target_gap: int | Q,
    *,
    gap_method: str = "factorization",
    exponent_steps: int = 1,
    weighted_incidence_cap: int | Q = 4,
    minimum_girth: int = 4,
    max_cycle_length: int = 4,
    max_cycle_diameter: int = 2,
    decay_base: int | Q = 1,
) -> dict[str, Any]:
    """Decide whether some real radius proves the requested fixed-family floor.

    For factorization, exponent_steps is a freely chosen positive integer,
    with the explicit strict constraint Omega(radius)<exponent_steps. It is
    not claimed to equal the existing API's floor(Omega)+1.

    A positive incidence cap is required in this first version, making the
    minimal fixed-point radius strictly positive whenever it exists.
    The real radius may be irrational; no rational-grid completeness is used.
    """
    target = _rational(target_gap, "target_gap")
    cap = _rational(weighted_incidence_cap, "weighted_incidence_cap")
    steps = _integer(exponent_steps, "exponent_steps")
    if target <= 0 or cap <= 0:
        raise ValueError("target_gap and weighted_incidence_cap must be strictly positive")
    if steps < 1:
        raise ValueError("exponent_steps must be a positive integer")
    if gap_method not in {"curvature", "factorization"}:
        raise ValueError("gap_method must be 'curvature' or 'factorization'")

    # A radius-one probe exposes the same coefficient bounds even when its
    # own gate fails. It is only a source of coefficients, never gap evidence.
    source = invariant_vacuum_fourier_family(
        group,
        kappa,
        correction_radius=Q(1),
        weighted_incidence_cap=cap,
        minimum_girth=minimum_girth,
        max_cycle_length=max_cycle_length,
        max_cycle_diameter=max_cycle_diameter,
        decay_base=decay_base,
    )["witness"]
    arithmetic = source["arithmetic"]
    coupling = Q(source["kappa"])
    decay = Q(source["decay_base"])
    _, casimir, ricci, hessian_factor, influence_factor = _constants(group)
    forcing = Q(arithmetic["seed_norm_upper"])
    bilinear = Q(arithmetic["bilinear_constant"])
    seed_hessian = Q(arithmetic["seed_hessian_row_upper"])
    seed_influence = Q(arithmetic["seed_tv_influence_row_upper"])
    seed_oscillation = Q(arithmetic["seed_conditional_log_density_oscillation_upper"])
    # Both norm conventions give zeta=2*beta/girth; beta comes from the
    # existing invariant-vacuum coefficient helper.
    oscillation_factor = 2 * influence_factor / minimum_girth
    discriminant = 1 - 4 * bilinear * forcing
    vertex = (1 - 2 * bilinear * forcing) / (2 * bilinear)

    polynomial_coefficients = {
        "positive_radius": [Q(0), Q(1)],
        "fixed_point_slack": [
            -bilinear * forcing**2, 1 - 2 * bilinear * forcing, -bilinear
        ],
        "strict_contraction_margin": [1 - 2 * bilinear * forcing, -2 * bilinear],
    }
    if gap_method == "curvature":
        polynomial_coefficients["target_gap_margin"] = [
            coupling * (ricci - 2 * seed_hessian) / 2 - target,
            -coupling * hessian_factor,
        ]
        target_factorization = None
    else:
        polynomial_coefficients["influence_margin"] = [
            1 - seed_influence, -influence_factor
        ]
        polynomial_coefficients["exponent_margin"] = [
            steps - seed_oscillation, -oscillation_factor
        ]
        target_factorization = {
            "scale": str(coupling * casimir / 2),
            "influence_coefficients": [str(1 - seed_influence), str(-influence_factor)],
            "exponential_base_coefficients": [
                str(1 - seed_oscillation / steps), str(-oscillation_factor / steps)
            ],
            "exponent_steps": steps,
            "subtract_target_gap": str(target),
            "relation": ">=0",
        }

    root: Pair | None = None
    rational_radius: str | None = None
    bound: Pair | None = None
    checks: dict[str, dict[str, Any]] = {
        "positive_fixed_point_discriminant": {
            "rational_value": str(discriminant),
            "sign": int(discriminant > 0) - int(discriminant < 0),
            "relation": ">0",
            "passed": discriminant > 0,
        }
    }
    if discriminant > 0:
        root = (vertex, -1 / (2 * bilinear))
        rational_radius = _rational_root(root, discriminant)
        for name, coefficients in polynomial_coefficients.items():
            value = _poly_at_pair(coefficients, root, discriminant)
            checks[name] = _constraint(
                value,
                discriminant,
                strict=name not in {"fixed_point_slack", "target_gap_margin"},
            )
        if gap_method == "curvature":
            bound = _poly_at_pair(
                [
                    coupling * (ricci - 2 * seed_hessian) / 2,
                    -coupling * hessian_factor,
                ],
                root,
                discriminant,
            )
        else:
            influence_margin = _poly_at_pair(
                [1 - seed_influence, -influence_factor], root, discriminant
            )
            exponential_base = _poly_at_pair(
                [1 - seed_oscillation / steps, -oscillation_factor / steps],
                root,
                discriminant,
            )
            bound = _pair_scale(
                _pair_mul(
                    influence_margin,
                    _pair_pow(exponential_base, steps, discriminant),
                    discriminant,
                ),
                coupling * casimir / 2,
            )
            checks["target_gap_margin"] = _constraint(
                _pair_add(bound, (-target, Q(0))), discriminant, strict=False
            )
    feasible = all(check["passed"] for check in checks.values())
    witness = {
        "inputs": {
            "group": group,
            "kappa": str(coupling),
            "target_gap": str(target),
            "gap_method": gap_method,
            "exponent_steps": steps,
            "weighted_incidence_cap": str(cap),
            "minimum_girth": minimum_girth,
            "max_cycle_length": max_cycle_length,
            "max_cycle_diameter": max_cycle_diameter,
            "decay_base": str(decay),
        },
        "normalization": source["normalization"],
        "energy_units": source["energy_units"],
        "family_class": source["family_class"],
        "gauss_constraint": source["gauss_constraint"],
        "gap_scope": source["gap_scope"],
        "charged_scope": source["charged_scope"],
        "boundary_scope": source["boundary_scope"],
        "coefficient_source": "invariant_vacuum_fourier_family coefficient bounds; radius-one probe is not gap evidence",
        "derived_coefficients": {
            "coefficient_norm": arithmetic["coefficient_norm"],
            "forcing_A": str(forcing),
            "bilinear_B": str(bilinear),
            "fundamental_casimir": str(casimir),
            "haar_ricci": str(ricci),
            "seed_hessian_h0": str(seed_hessian),
            "hessian_norm_factor": str(hessian_factor),
            "seed_influence_eta0": str(seed_influence),
            "influence_norm_factor_beta": str(influence_factor),
            "seed_oscillation_Omega0": str(seed_oscillation),
            "oscillation_norm_factor_zeta": str(oscillation_factor),
            "fixed_point_discriminant": str(discriminant),
            "strict_contraction_vertex": str(vertex),
        },
        "radius_domain": "strictly positive real radius",
        "exponent_scope": (
            "freely chosen positive integer m; Omega(r)<m is required, not m=floor(Omega(r))+1"
            if gap_method == "factorization"
            else "exponent_steps is unused by the curvature criterion"
        ),
        "polynomial_coefficient_order": "ascending powers of r",
        "polynomial_constraints": {
            name: {
                "coefficients": list(map(str, coefficients)),
                "relation": ">=0" if name in {"fixed_point_slack", "target_gap_margin"} else ">0",
            }
            for name, coefficients in polynomial_coefficients.items()
        },
        "factored_target_constraint": target_factorization,
        "quadratic_field_convention": "pair [x,y] represents x+y*sqrt(D); D is the fixed-point discriminant",
        "radius_root": [str(root[0]), str(root[1]), str(discriminant)] if root else None,
        "root_rational": rational_radius,
        "target_bound_pair": _pair_payload(bound) if bound else None,
        "target_bound_rational": _rational_root(bound, discriminant) if bound else None,
        "target_bound_pair_scope": "polynomial value at the candidate root; licensed as a gap floor only when its domain constraints pass",
        "target_bound_domain_verified": all(
            check["passed"] for name, check in checks.items() if name != "target_gap_margin"
        ),
        "checks": checks,
        "failed_constraints": [name for name, check in checks.items() if not check["passed"]],
        "decision_basis": "all remaining margins and the selected positive-domain gap floor decrease with radius, so feasibility is decided at the least fixed-point radius",
        "feasible": feasible,
    }
    certificate = make_certificate(
        claim="exact real feasibility decision for a sufficient invariant-vacuum target-gap criterion",
        payload={"type": "vacuum_constraints_v1", "witness": witness},
        honesty={
            "constraint_decision_verified": True,
            "volume_uniform_target_gap_verified": feasible,
            "gap_absence_claim": False,
            "yang_mills_claim": False,
            "yang_mills_mass_gap_claim": False,
            "continuum_claim": False,
            "infinite_volume_claim": False,
        },
        meta={
            "analytic_implication": "docs/api/gauge-vacuum-constraints.md",
            "scope": "finite exact semialgebraic decision plus a written fixed-coupling finite-family implication",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": "FEASIBLE" if feasible else "INFEASIBLE_CRITERION",
        "constraint_decision_verified": True,
        "feasible": feasible,
        "finite_gate_verified": feasible,
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        "volume_uniform_finite_graph_family_verified": feasible,
        "volume_uniform_target_gap_verified": feasible,
        "target_gap_lower": str(target) if feasible else "0",
        "gap_absence_claim": False,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "static_confinement_claim": False,
        "string_tension_claim": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "analytic_implication_formally_verified": False,
    }


def replay_vacuum_constraint_certificate(certificate: dict[str, Any]) -> bool:
    """Verify either exact decision; an infeasible criterion is not a gap disproof."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "vacuum_constraints_v1":
            return False
        inputs = payload["witness"]["inputs"]
        report = solve_vacuum_constraints(
            inputs["group"],
            Q(inputs["kappa"]),
            Q(inputs["target_gap"]),
            gap_method=inputs["gap_method"],
            exponent_steps=inputs["exponent_steps"],
            weighted_incidence_cap=Q(inputs["weighted_incidence_cap"]),
            minimum_girth=inputs["minimum_girth"],
            max_cycle_length=inputs["max_cycle_length"],
            max_cycle_diameter=inputs["max_cycle_diameter"],
            decay_base=Q(inputs["decay_base"]),
        )
        return bool(report["constraint_decision_verified"] and report["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False
