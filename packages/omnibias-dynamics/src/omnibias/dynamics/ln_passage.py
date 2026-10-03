# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Finite LN/exp checks for the coalescing Hilbert-XVI passage proposal.

The elementary coordinate identities and finite guard checks in this module
do not prove that the physical Dulac/first-hit map is Log-Noetherian.  In
particular, making an unbounded factor a coordinate does not bound its sup
norm, and a relative shrinking radius does not prove physical transversality.
Every report keeps those analytic obligations explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.asymptotic_jet import (
    verify_fixed_product_derivative,
    weighted_scale_derivative,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.log_noetherian import (
    ChainFunctionRef,
    LNCell,
    LNChain,
    LNFiber,
    NamedPolynomial,
    ln_format,
    log_chart_derivative_bound,
    monomial_eigenvalue_bound,
    seal_ln_certificate,
    seal_ln_chain_closure_obligation,
    seal_ln_format_bound_obligation,
    verify_ln_certificate,
    verify_ln_chain,
)
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    certify_stopped_event,
    polynomial_interval,
    verify_stopped_event,
)

GuardStatus = Literal["identically_zero", "nowhere_zero", "unresolved"]

KILL_A_LAMBDA1 = -3.0
KILL_B_LAMBDA1 = -2.0
EXISTING_SECTION_POWERS: tuple[int, ...] = (0, 3, 4)


@dataclass(frozen=True)
class PassageCell:
    """One declared coordinate cell, with physical membership kept separate."""

    name: str
    coordinates: tuple[str, ...]
    purpose: str
    physical_map_membership_proved: bool = False


@dataclass(frozen=True)
class GuardCheck:
    """Finite interval classification of one declared cell guard."""

    cell: str
    name: str
    status: GuardStatus
    enclosure: Interval


@dataclass(frozen=True)
class KillPlacement:
    """Replayable placement or refusal of one named degeneration."""

    name: str
    cell: str
    pointwise_in_cell: bool
    uniform_format_bounded: bool
    positive_radius_margin: bool
    failing_chain_function: str
    reason: str
    values: dict[str, float | bool]


def canonical_L(lambda1: float, sep: float) -> float:
    """Return the unique ``L`` satisfying ``sep² = lambda1² - 4L``."""
    if not (math.isfinite(lambda1) and math.isfinite(sep)) or sep < 0:
        raise ValueError("lambda1 and nonnegative sep must be finite")
    return (lambda1 * lambda1 - sep * sep) / 4.0


def quadratic_relation_residual(
    L: float, lambda1: float, sep: float
) -> float:
    """Residual of the exact quadratic-family relation."""
    if not all(math.isfinite(value) for value in (L, lambda1, sep)):
        raise ValueError("quadratic parameters must be finite")
    return sep * sep - (lambda1 * lambda1 - 4.0 * L)


def first_root(lambda1: float, sep: float) -> float:
    """The first root ``(-lambda1-sep)/2``."""
    if not (math.isfinite(lambda1) and math.isfinite(sep)) or sep < 0:
        raise ValueError("lambda1 and nonnegative sep must be finite")
    return (-lambda1 - sep) / 2.0


def stable_first_root(L: float, lambda1: float, sep: float) -> float:
    """Cancellation-free first root on the negative-``lambda1`` branch."""
    if L <= 0 or lambda1 >= 0 or sep < 0:
        raise ValueError("requires L > 0, lambda1 < 0, and sep >= 0")
    denominator = abs(lambda1) + sep
    if denominator <= 0:
        raise ValueError("first-root denominator must be positive")
    return 2.0 * L / denominator


def declared_cells() -> tuple[PassageCell, ...]:
    """The candidate finite ledger; this is not a physical-cover theorem."""
    return (
        PassageCell(
            "cellA",
            ("epsilon", "log_sep", "W"),
            "super-small root separation and logarithmic height",
        ),
        PassageCell(
            "cellB",
            ("L", "lambda1", "sep", "r1", "a"),
            "shrinking first root with a base-dependent radial wall",
        ),
        PassageCell(
            "cellAB",
            ("epsilon", "L", "lambda1", "sep", "r1"),
            "coordinate overlap where both candidate radii are positive",
        ),
    )


def coordinate_ln_chain() -> LNChain:
    """Build the exact-Q coordinate chain used by the finite substrate check.

    The functions are coordinate/radius functions only.  In particular, this
    chain does not contain ``tau=epsilon*log(1/sep)`` or the physical first-hit
    map: their membership is exactly the missing analytic obligation.
    """
    radius = ChainFunctionRef("radius")
    cell = LNCell(
        (
            LNFiber("Point"),
            LNFiber("PuncturedDisc", (Fraction(1),)),
            LNFiber("Annulus", (radius, radius.scaled(4))),
        ),
        real_part=True,
    )
    epsilon = SparsePolynomial.variable(3, 0)
    sep = SparsePolynomial.variable(3, 1)
    w_coord = SparsePolynomial.variable(3, 2)
    radius_value = SparsePolynomial.constant(3, 1)
    functions = (
        NamedPolynomial("epsilon", epsilon),
        NamedPolynomial("sep", sep),
        NamedPolynomial("W", w_coord),
        NamedPolynomial("radius", radius_value),
    )
    zero_base = SparsePolynomial.constant(3, 0)
    zero_chain = SparsePolynomial.constant(4, 0)
    chain_variables = tuple(SparsePolynomial.variable(4, axis) for axis in range(4))
    closure = (
        (chain_variables[0], zero_chain, zero_chain),
        (zero_chain, chain_variables[1], zero_chain),
        (zero_chain, zero_chain, chain_variables[2]),
        (zero_chain, zero_chain, zero_chain),
    )
    claimed = (
        (epsilon, zero_base, zero_base),
        (zero_base, sep, zero_base),
        (zero_base, zero_base, w_coord),
        (zero_base, zero_base, zero_base),
    )
    return LNChain(
        cell=cell,
        functions=functions,
        closure_matrix=closure,
        claimed_derivatives=claimed,
    )


def coordinate_sup_enclosures(
    *,
    delta: Fraction = Fraction(1, 2),
) -> dict[str, Interval]:
    """Enclose every coordinate-chain function on a declared delta box.

    These are enclosures of the finite coordinate model, not of ``D_phys``.
    """
    if not 0 < delta < 1:
        raise ValueError("delta must satisfy 0 < delta < 1")
    chain = coordinate_ln_chain()
    chain.cell.delta_extension(delta)
    box = (
        Interval(0.025, 0.125),
        Interval(1.0e-6, float(1 / delta)),
        Interval(float(delta), float(4 / delta)),
    )
    return {
        function.name: polynomial_interval(function.polynomial, box)
        for function in chain.functions
    }


def coordinate_ln_evidence() -> dict[str, object]:
    """Seal exact coordinate closure, format, and conditional Cauchy bounds."""
    chain = coordinate_ln_chain()
    sup_enclosures = coordinate_sup_enclosures()
    sup_hi = sum(max(abs(value.lo), abs(value.hi)) for value in sup_enclosures.values())
    supplied_sup = Interval(0.0, sup_hi)
    format_report = ln_format(
        chain,
        cell_format=Fraction(7),
        sup_bound=supplied_sup,
    )
    certificate = seal_ln_certificate(
        chain,
        cell_format=Fraction(7),
        sup_bound=supplied_sup,
    )
    closure_obligation = seal_ln_chain_closure_obligation(chain)
    format_obligation = seal_ln_format_bound_obligation(
        format_report,
        strict_upper=format_report.combinatorial_part + 1,
    )
    cauchy = log_chart_derivative_bound(
        supplied_sup,
        Fraction(1, 2),
        2,
    )
    eigen = monomial_eigenvalue_bound(
        Fraction(1),
        2,
        supplied_sup,
    )
    return {
        "chain_valid": verify_ln_chain(chain),
        "sup_enclosures": sup_enclosures,
        "format": format_report,
        "certificate": certificate,
        "certificate_valid": verify_ln_certificate(certificate),
        "closure_obligation": closure_obligation,
        "format_obligation": format_obligation,
        "conditional_c2_bound": cauchy + eigen,
        "weighted_coordinate_replay": coordinate_monomial_replay(),
        "physical_phi_member": False,
        "physical_uniform_c2": False,
        "reason": (
            "the exact chain contains coordinate polynomials only; tau/log, "
            "the physical first-hit map, and their holomorphic sup bounds are absent"
        ),
    }


def corrected_kill_a(epsilon: float) -> KillPlacement:
    """Place the algebraically valid super-small-separation path.

    The plan's fixed ``L=1, lambda1=-3`` tuple is rejected separately by
    :func:`invalid_plan_kill_a`.  Here ``L`` is corrected to
    ``(9-sep²)/4``.  Log-domain formulas avoid underflow.
    """
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0,1)")
    log_sep = -1.0 / (epsilon * epsilon)
    sep = math.exp(log_sep) if log_sep >= math.log(float.fromhex("0x0.0000000000001p-1022")) else 0.0
    L = canonical_L(KILL_A_LAMBDA1, sep)
    tau = 1.0 / epsilon
    outgoing = {
        f"log_W_ratio_eps_{power}": (
            2.0 / epsilon
            + (power - 3) * epsilon * math.log(epsilon)
        )
        for power in EXISTING_SECTION_POWERS
    }
    return KillPlacement(
        name="kill_super_small_sep_corrected",
        cell="cellA",
        pointwise_in_cell=True,
        uniform_format_bounded=False,
        positive_radius_margin=True,
        failing_chain_function="physical_outgoing_matching_W_ratio",
        reason=(
            "tau=epsilon*log(1/sep)=1/epsilon and every existing-section "
            "log(W_max/W_e)=2/epsilon+O(epsilon*|log epsilon|) diverges"
        ),
        values={
            "epsilon": epsilon,
            "log_sep": log_sep,
            "sep": sep,
            "lambda1": KILL_A_LAMBDA1,
            "L": L,
            "tau": tau,
            "quadratic_relation_holds": abs(
                quadratic_relation_residual(L, KILL_A_LAMBDA1, sep)
            )
            <= 8.0 * math.ulp(max(1.0, L)),
            **outgoing,
        },
    )


def invalid_plan_kill_a(epsilon: float) -> dict[str, float | bool]:
    """Reject the printed ``L=1, lambda1=-3, sep=exp(-1/eps²)`` tuple."""
    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0,1)")
    sep = math.exp(-1.0 / (epsilon * epsilon))
    residual = quadratic_relation_residual(1.0, KILL_A_LAMBDA1, sep)
    return {
        "epsilon": epsilon,
        "L": 1.0,
        "lambda1": KILL_A_LAMBDA1,
        "sep": sep,
        "relation_residual": residual,
        "is_quadratic_family_point": residual == 0.0,
    }


def shrinking_root_radius(
    L: float, *, theta: float, lambda1: float = KILL_B_LAMBDA1
) -> tuple[float, float, float]:
    """Return ``(sep, r1, r1-theta*sep)`` on the shrinking-root family."""
    if not (math.isfinite(theta) and theta > 0):
        raise ValueError("theta must be finite and positive")
    if not (math.isfinite(L) and 0 < L < lambda1 * lambda1 / 4.0):
        raise ValueError("L must lie strictly inside the first-root regime")
    sep = math.sqrt(lambda1 * lambda1 - 4.0 * L)
    r1 = stable_first_root(L, lambda1, sep)
    return sep, r1, r1 - theta * sep


def kill_b(n: int, *, theta: float = 0.125) -> KillPlacement:
    """Test the proposed wall ``a=r1-theta*sep`` on ``L=1/n``."""
    if type(n) is not int or n < 2:
        raise ValueError("n must be an integer at least two")
    L = 1.0 / n
    sep, r1, proposed = shrinking_root_radius(L, theta=theta)
    relative = (1.0 - theta) * r1 if theta < 1 else float("nan")
    positive = proposed > 0
    return KillPlacement(
        name="kill_shrinking_root",
        cell="cellB",
        pointwise_in_cell=positive,
        uniform_format_bounded=False,
        positive_radius_margin=positive,
        failing_chain_function="a(L,lambda1)=r1-theta*sep",
        reason=(
            "the proposed radius tends to -2*theta and is eventually negative; "
            "the positive relative replacement (1-theta)*r1 still loses the "
            "absolute physical transversality margin as r1 tends to zero"
        ),
        values={
            "n": float(n),
            "L": L,
            "lambda1": KILL_B_LAMBDA1,
            "sep": sep,
            "r1": r1,
            "proposed_radius": proposed,
            "relative_radius": relative,
            "quadratic_relation_holds": abs(
                quadratic_relation_residual(L, KILL_B_LAMBDA1, sep)
            )
            <= 8.0 * math.ulp(1.0),
        },
    )


def coordinate_monomial_replay() -> bool:
    """Replay one exact logarithmic-frame identity over Q.

    This is a substrate check for a declared coordinate monomial, not closure
    of the physical first-hit map.
    """
    omega = SparsePolynomial.variable(2, 0)
    u_scale = SparsePolynomial.variable(2, 1)
    monomial = omega * u_scale**2
    claimed = monomial
    return (
        weighted_scale_derivative(monomial, order=2).terms == claimed.terms
        and verify_fixed_product_derivative(monomial, claimed, order=2)
    )


def classify_guard(
    cell: str,
    name: str,
    polynomial: SparsePolynomial,
    box: tuple[Interval, ...],
) -> GuardCheck:
    """Classify a finite guard without treating ambiguity as exclusion."""
    value = (
        Interval.point(0.0)
        if not polynomial.terms
        else polynomial_interval(polynomial, box)
    )
    status: GuardStatus
    if not polynomial.terms:
        status = "identically_zero"
    elif value.lo > 0 or value.hi < 0:
        status = "nowhere_zero"
    else:
        status = "unresolved"
    return GuardCheck(cell, name, status, value)


def admission_compatibility() -> dict[str, object]:
    """Finite guard compatibility on model cells, not physical first-hit."""
    z = SparsePolynomial.variable(1, 0)
    zero = SparsePolynomial.constant(1, 0)
    positive_box = (Interval(0.25, 0.5),)
    checks = (
        classify_guard("cellA", "algebraic_relation_after_substitution", zero, positive_box),
        classify_guard("cellA", "positive_coordinate", z, positive_box),
        classify_guard("cellB", "algebraic_relation_after_substitution", zero, positive_box),
        classify_guard("cellB", "positive_relative_radius", z, positive_box),
        classify_guard("cellAB", "overlap_coordinate", z, positive_box),
    )
    ambiguous = classify_guard(
        "negative_control",
        "crossing_guard",
        z - Fraction(3, 8),
        positive_box,
    )
    finite_compatible = all(
        check.status in ("identically_zero", "nowhere_zero") for check in checks
    )
    return {
        "checks": checks,
        "ambiguous_control": ambiguous,
        "finite_compatible": finite_compatible,
        "small_label_tube": {
            "u": 1.0,
            "tbox": 0.4,
            "strict": 0.4 < 1.0**2 / 2.0,
        },
        "physical_first_hit_complete": False,
        "reason": (
            "guard compatibility does not prove existence, no earlier hit, or "
            "transversality of the singular physical continuation"
        ),
    }


def certify_regular_arc_model() -> dict[str, object]:
    """Certify a bounded polynomial regular arc with the stopped-event engine.

    The arc demonstrates the reusable finite-time seam.  It is not the missing
    singular tail of the quadratic passage.
    """
    x = SparsePolynomial.variable(3, 0)
    flow = PolynomialFlow((SparsePolynomial.constant(3, 1),), parameter_count=1)
    request = StoppedEventRequest(
        flow=flow,
        initial=(SparsePolynomial.constant(1, 0),),
        parameters=(Interval(-0.01, 0.01),),
        target=PolynomialEvent(x - 1, direction=1, name="regular_arc_exit"),
        step=0.125,
        max_steps=12,
        order=6,
        derivative_order=2,
    )
    result = certify_stopped_event(request)
    return {
        "certified": result.certified,
        "replayed": verify_stopped_event(result) if result.certified else False,
        "status": result.status,
        "reason": result.reason,
        "physical_singular_tail": False,
    }


def overlap_matching() -> dict[str, bool | str]:
    """Record exact coordinate matching and the still-open physical match."""
    z = SparsePolynomial.variable(1, 0)
    coordinate_identity = (z + 1 - 1).terms == z.terms
    return {
        "coordinate_identity": coordinate_identity,
        "physical_map_matching": False,
        "reason": (
            "physical uniqueness applies only after both first-hit "
            "continuations exist on the overlap"
        ),
    }


def assert_g1_passed(items: tuple[bool, bool, bool, bool]) -> None:
    """Refuse a G1 assertion unless all four analytic items are closed."""
    if len(items) != 4 or not all(items):
        raise ValueError("G1 requires all four written analytic items")


def report(*, epsilon: float = 0.05, kill_b_n: int = 32) -> dict[str, object]:
    """Return the finite LN-passage assessment with an explicit blocked verdict."""
    kill_a = corrected_kill_a(epsilon)
    shrinking = kill_b(kill_b_n)
    admission = admission_compatibility()
    regular = certify_regular_arc_model()
    overlap = overlap_matching()
    items = (
        False,  # finite physical chart cover
        False,  # complete physical first-hit
        False,  # physical C2 remainder
        False,  # physical endpoint matching
    )
    return {
        "schema": "hilbert16-ln-passage-finite-replay-v1",
        "cells": declared_cells(),
        "coordinate_evidence": coordinate_ln_evidence(),
        "coordinate_chain_closure": coordinate_monomial_replay(),
        "kill_a": kill_a,
        "invalid_plan_kill_a": invalid_plan_kill_a(epsilon),
        "kill_b": shrinking,
        "admission": admission,
        "regular_arc": regular,
        "overlap": overlap,
        "g1_items_closed": items,
        "g1_passed": all(items),
        "full_hilbert16_solved": False,
        "sharper_failure": (
            kill_a.failing_chain_function,
            shrinking.failing_chain_function,
        ),
        "scope": (
            "Finite exact/interval checks and two named obstruction replays. "
            "Not physical LN membership, not G1, and not Hilbert XVI."
        ),
    }


__all__ = [
    "EXISTING_SECTION_POWERS",
    "GuardCheck",
    "GuardStatus",
    "KILL_A_LAMBDA1",
    "KILL_B_LAMBDA1",
    "KillPlacement",
    "PassageCell",
    "admission_compatibility",
    "assert_g1_passed",
    "canonical_L",
    "certify_regular_arc_model",
    "classify_guard",
    "coordinate_ln_chain",
    "coordinate_ln_evidence",
    "coordinate_monomial_replay",
    "coordinate_sup_enclosures",
    "corrected_kill_a",
    "declared_cells",
    "first_root",
    "invalid_plan_kill_a",
    "kill_b",
    "overlap_matching",
    "quadratic_relation_residual",
    "report",
    "shrinking_root_radius",
    "stable_first_root",
]
