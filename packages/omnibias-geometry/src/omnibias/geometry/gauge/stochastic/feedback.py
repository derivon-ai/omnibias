# SPDX-License-Identifier: Apache-2.0
"""Finite obstructions and conditional arithmetic for cube feedback trials.

The written group-theoretic implications live in the consumer proof note.
Replay checks the finite central sign problem; it does not check a proposed
Hamiltonian isometry or establish its missing derivative estimates.
"""

from __future__ import annotations

from fractions import Fraction as Q
from itertools import product
from math import prod
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.stochastic.finite import rational


def central_factor_obstruction(factor_count: int = 5) -> dict[str, Any]:
    """Enumerate central SU2 factorizations of -I for a specified finite count.

    Boundary-only conjugation-equivariance forces each selected factor at
    -I into the centre. Every such factorization costs at least A(-I)=4.
    A randomized equivariant conditional distribution is outside this class.
    """
    if type(factor_count) is not int or not 1 <= factor_count <= 16:
        raise ValueError("factor_count must be an integer in [1,16] for this enumerator")
    rows: list[dict[str, Any]] = [{"signs": list(signs), "action_sum": sum(2-2*s for s in signs)}
            for signs in product((-1, 1), repeat=factor_count) if prod(signs) == -1]
    minimum = min(row["action_sum"] for row in rows)
    coefficient = Q(121, 49*factor_count)
    payload = {
        "type": "central_su2_feedback_obstruction_v1", "factor_count": factor_count,
        "central_boundary": [-1, 0, 0, 0], "boundary_action": 4,
        "rows": rows, "minimum_central_action_sum": minimum,
        "geodesic_action_coefficient_upper": str(coefficient),
        "geodesic_bound_derivation": "sum A(exp(log(B)/n)) / A(B) <= pi^2/(4n) <= 121/(49n)",
        "deterministic_boundary_only_equivariant_contraction_refuted": minimum >= 4,
        "pointwise_balanced_trial_is_global_equivariant_section": False,
    }
    return {**payload, "certificate": make_certificate(
        claim="finite central obstruction to deterministic equivariant SU2 feedback", payload=payload, meta={"transcend_backend": "not_used"}),
        "method": "EXACT_RATIONAL", "theorem_prover_verified": False,
        "mathlib_verified": False, "physical_isometry_constructed": False,
        "physical_feedback_contraction_verified": False, "continuum_claim": False,
        "yang_mills_mass_gap_claim": False}


def conditional_feedback_iteration(
    constant: int | Q, theta: int | Q, initial: int | Q, *, steps: int = 6,
) -> dict[str, Any]:
    """Iterate C_(n+1)<=C0/2+theta*C_n, conditional on an unproved form bound."""
    c0, coefficient, first = (rational(x, name) for x, name in
                              ((constant, "constant"), (theta, "theta"), (initial, "initial")))
    if c0 < 0 or coefficient < 0 or first < 0:
        raise ValueError("constant, theta and initial must be nonnegative")
    if type(steps) is not int or not 0 <= steps <= 1000:
        raise ValueError("steps must be an integer in [0,1000]")
    values = [first]
    for _ in range(steps):
        values.append(c0/2+coefficient*values[-1])
    fixed = c0/(2*(1-coefficient)) if coefficient < 1 else None
    payload = {"type": "conditional_feedback_iteration_v1",
        "inputs": {"constant": str(c0), "theta": str(coefficient), "initial": str(first), "steps": steps},
        "upper_budgets": [str(v) for v in values],
        "contractive_coefficient": coefficient < 1,
        "fixed_budget": str(fixed) if fixed is not None else None,
        "uniform_budget": str(max(first, fixed)) if fixed is not None else None,
        "external_premises": ["normalized physical attachment isometry", "global all-link form comparison",
                              "uniform derivative and large-field estimates", "attachment family covers the desired graph sequence"]}
    return {**payload, "certificate": make_certificate(claim="conditional finite affine feedback arithmetic", payload=payload, meta={"transcend_backend": "not_used"}),
        "physical_feedback_contraction_verified": False, "theorem_prover_verified": False,
        "mathlib_verified": False, "continuum_claim": False, "yang_mills_mass_gap_claim": False}


def replay_feedback_certificate(certificate: dict[str, Any]) -> bool:
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] == "central_su2_feedback_obstruction_v1":
            expected = central_factor_obstruction(payload["factor_count"])
        elif payload["type"] == "conditional_feedback_iteration_v1":
            raw = payload["inputs"]
            expected = conditional_feedback_iteration(Q(raw["constant"]), Q(raw["theta"]), Q(raw["initial"]), steps=raw["steps"])
        else:
            return False
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = ["central_factor_obstruction", "conditional_feedback_iteration", "replay_feedback_certificate"]
