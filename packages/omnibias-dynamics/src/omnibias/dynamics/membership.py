# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Sound collar-membership certificates on ``x in [delta, delta0]``.

A graphic's *declared* finite Dulac model (``GraphicTarget.return_map``,
already the input to
:func:`~omnibias.dynamics.graphic.certify_graphic_cyclicity`) is a hand-
written truncation with an unproved remainder. The *derived* corner
expansion from :mod:`omnibias.dynamics.saddle_normal_form` is instead read
mechanically off the field's resonant normal form. This module proves the
two soundly **agree** -- as interval enclosures, never a single float -- on a
collar bounded away from the corner itself, via the same
:func:`~omnibias.dynamics.dulac.enclose_dulac_term` interval machinery
already sealed for the Dulac-model certificates (reused, not
reimplemented).

**Scope, read before quoting a result.** This certificate is restricted to
``x in [delta, delta0]`` with ``delta > 0`` strictly: the corner window
``x -> 0`` itself is explicitly *not* covered (``corner_window_external``
stays ``True`` on every certificate this module produces), so a
``PROVED_COLLAR`` status earns the narrow ``collar_return_membership_proved``
flag while ``physical_return_membership_proved`` stays ``False``. When the
collar proof also isolates a unique zero of the declared displacement (via
:func:`~omnibias.core.verified.rootfind.interval_newton`), a genuine limit
cycle inside the collar is proved to exist and be unique -- a real positive
result, but again local to the collar, not a graphic-wide finite-cyclicity
theorem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.verified.asymptotic_jet import power_compensator
from omnibias.core.verified.interval import Interval, sum_intervals
from omnibias.core.verified.rootfind import NewtonResult, interval_newton
from omnibias.core.verified.transcend import certificate_mode, exp_iv, ln_iv
from omnibias.dynamics.dulac import (
    DulacExpansion,
    DulacTerm,
    RationalInterval,
    displacement_expansion,
    enclose_dulac_term,
)
from omnibias.dynamics.graphic import GraphicTarget
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.saddle_normal_form import DerivedDulacExpansion
from omnibias.dynamics.uniform_remainder import (
    UniformFlatRemainderCertificate,
    verify_uniform_flat_remainder,
)

__all__ = [
    "CollarGridNode",
    "CollarMembershipCertificate",
    "CollarSequenceCertificate",
    "MembershipStatus",
    "UniqueCycleResult",
    "certify_collar_membership",
    "certify_collar_sequence",
    "verify_collar_membership",
    "verify_collar_sequence",
]

MembershipStatus = Literal["PROVED_COLLAR", "BLOCKED"]


def _q(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _enclose_expansion(expansion: DulacExpansion, box: RationalInterval) -> Interval:
    return sum_intervals([enclose_dulac_term(term, box) for term in expansion.terms])


def _enclose_power_log(exponent: Interval, log_power: int, x: Interval) -> Interval:
    """``x**exponent * log(x)**log_power``; ``exponent`` may be any real (not
    just the strictly-positive exponents :class:`~omnibias.dynamics.dulac.DulacTerm`
    requires), which is exactly what differentiating a term needs."""
    if log_power == 0:
        with certificate_mode():
            return exp_iv(exponent * ln_iv(x))
    return log_power * power_compensator(exponent, exponent, x, a_order=log_power - 1)


def _enclose_term_derivative(term: DulacTerm, x: Interval) -> Interval:
    """Sound enclosure of ``d/dx [coefficient * x**exponent * log(x)**log_power]``."""
    exponent = term.exponent.to_interval()
    coefficient = term.coefficient.to_interval()
    one = Interval.point(1.0)
    shifted = exponent - one
    total = coefficient * exponent * _enclose_power_log(shifted, term.log_power, x)
    if term.log_power >= 1:
        total = total + coefficient * Interval.point(float(term.log_power)) * _enclose_power_log(
            shifted, term.log_power - 1, x
        )
    return total


def _enclose_expansion_derivative(expansion: DulacExpansion, x: Interval) -> Interval:
    return sum_intervals([_enclose_term_derivative(term, x) for term in expansion.terms])


@dataclass(frozen=True)
class CollarGridNode:
    """One sub-box's declared-vs-derived enclosure comparison."""

    lo: Fraction
    hi: Fraction
    declared: tuple[float, float]
    derived: tuple[float, float]
    contained: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "lo": _q(self.lo),
            "hi": _q(self.hi),
            "declared": [self.declared[0], self.declared[1]],
            "derived": [self.derived[0], self.derived[1]],
            "contained": self.contained,
        }


@dataclass(frozen=True)
class UniqueCycleResult:
    """A certified unique zero of the declared displacement on the collar, if any."""

    attempted: bool
    enclosure: tuple[float, float] | None
    proved_unique_cycle: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "enclosure": None if self.enclosure is None else [self.enclosure[0], self.enclosure[1]],
            "proved_unique_cycle": self.proved_unique_cycle,
        }


@dataclass(frozen=True)
class CollarMembershipCertificate:
    """A sound collar agreement certificate, or an explicit disjoint-box block."""

    target: GraphicTarget
    derived: DerivedDulacExpansion
    delta: Fraction
    delta0: Fraction
    grid: tuple[CollarGridNode, ...]
    status: MembershipStatus
    unique_cycle: UniqueCycleResult
    source_digest: str
    seal: Cert


@dataclass(frozen=True)
class CollarSequenceCertificate:
    """A monotone collar sequence ``delta_k -> 0`` with a summable remainder tail."""

    target: GraphicTarget
    derived: DerivedDulacExpansion
    delta0: Fraction
    deltas: tuple[Fraction, ...]
    certificates: tuple[CollarMembershipCertificate, ...]
    tail_bound: Fraction
    corner_window_external: bool
    source_digest: str
    seal: Cert


def _grid_breakpoints(delta: Fraction, delta0: Fraction, n: int) -> list[Fraction]:
    return [delta + (delta0 - delta) * Fraction(i, n) for i in range(n + 1)]


def _enclose_expansion_by_float(expansion: DulacExpansion, x: Interval) -> Interval:
    """Evaluate a Dulac expansion directly in float-interval space (no
    rational re-boxing), reusing the same power/log primitives as
    :func:`~omnibias.dynamics.dulac.enclose_dulac_term`."""
    total = Interval.point(0.0)
    for term in expansion.terms:
        exponent = term.exponent.to_interval()
        coefficient = term.coefficient.to_interval()
        total = total + coefficient * _enclose_power_log(exponent, term.log_power, x)
    return total


def _attempt_unique_cycle(target: GraphicTarget, delta: Fraction, delta0: Fraction) -> UniqueCycleResult:
    displacement = displacement_expansion(target.return_map)
    try:
        result: NewtonResult = interval_newton(
            lambda x: _enclose_expansion_by_float(displacement, x),
            lambda x: _enclose_expansion_derivative(displacement, x),
            (float(delta), float(delta0)),
        )
    except (ArithmeticError, ValueError, ZeroDivisionError):
        return UniqueCycleResult(True, None, False)
    proved = result["status"] == "unique_root" and result["unique"]
    return UniqueCycleResult(True, result["enclosure"], proved)


def certify_collar_membership(
    target: GraphicTarget,
    derived: DerivedDulacExpansion,
    *,
    delta: Fraction,
    delta0: Fraction,
    grid_points: int = 8,
    attempt_unique_cycle: bool = True,
) -> CollarMembershipCertificate:
    """Prove the declared and field-derived return maps soundly agree on the collar.

    Splits ``[delta, delta0]`` into ``grid_points`` exact-rational sub-boxes.
    On each box both the declared model (``target.return_map``) and the
    derived model (``derived.expansion``) are soundly enclosed via
    :func:`~omnibias.dynamics.dulac.enclose_dulac_term`; ``status`` is
    ``PROVED_COLLAR`` only when *every* sub-box's two enclosures overlap --
    one disjoint box blocks the whole certificate rather than being silently
    averaged away. When the collar is proved, an optional interval-Newton
    pass on the declared displacement additionally checks for a genuine
    unique zero (a certified limit cycle) inside the collar.
    """
    if not (0 < delta < delta0 <= 1):
        raise ValueError("the collar must satisfy 0 < delta < delta0 <= 1")
    if type(grid_points) is not int or grid_points < 1:
        raise ValueError("grid_points must be a positive integer")
    breakpoints = _grid_breakpoints(delta, delta0, grid_points)
    nodes: list[CollarGridNode] = []
    all_contained = True
    for lo, hi in zip(breakpoints, breakpoints[1:], strict=False):
        box = RationalInterval(lo, hi)
        declared_iv = _enclose_expansion(target.return_map, box)
        derived_iv = _enclose_expansion(derived.expansion, box)
        try:
            declared_iv.intersect(derived_iv)
            contained = True
        except ValueError:
            contained = False
        all_contained = all_contained and contained
        nodes.append(
            CollarGridNode(
                lo, hi, (declared_iv.lo, declared_iv.hi), (derived_iv.lo, derived_iv.hi), contained
            )
        )
    status: MembershipStatus = "PROVED_COLLAR" if all_contained else "BLOCKED"

    unique_cycle = UniqueCycleResult(False, None, False)
    if attempt_unique_cycle and status == "PROVED_COLLAR":
        unique_cycle = _attempt_unique_cycle(target, delta, delta0)

    seal = make_certificate(
        claim=(
            "Sound interval agreement of a declared and a field-derived Dulac "
            "model on a collar bounded away from the corner."
            if status == "PROVED_COLLAR"
            else "A declared and a field-derived Dulac model disjointly disagree on a collar sub-box."
        ),
        payload={
            "type": "collar_membership",
            "source_digest": target.digest,
            "derived_ratio": _q(derived.ratio),
            "delta": _q(delta),
            "delta0": _q(delta0),
            "grid": [node.to_payload() for node in nodes],
            "status": status,
            "unique_cycle": unique_cycle.to_payload(),
        },
        honesty=build_honesty(
            collar_return_membership_proved=status == "PROVED_COLLAR",
            physical_return_membership_proved=False,
            corner_window_external=True,
            graphic_finite_cyclicity_proved=False,
            full_hilbert16_solved=False,
            uniform_remainder_proved=False,
        ),
        meta={"transcend_backend": "not_used"},
    )
    return CollarMembershipCertificate(
        target, derived, delta, delta0, tuple(nodes), status, unique_cycle, target.digest, seal
    )


def certify_collar_sequence(
    target: GraphicTarget,
    derived: DerivedDulacExpansion,
    deltas: Sequence[Fraction],
    *,
    delta0: Fraction,
    grid_points: int = 8,
    uniform_remainder: UniformFlatRemainderCertificate | None = None,
) -> CollarSequenceCertificate:
    """Close a monotone collar sequence toward the corner for hyperbolic saddles only."""
    if not deltas:
        raise ValueError("deltas must be nonempty")
    ordered = tuple(deltas)
    if not all(0 < delta < delta0 for delta in ordered):
        raise ValueError("every delta must satisfy 0 < delta < delta0")
    if not all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)):
        raise ValueError("deltas must be strictly decreasing")
    if not derived.ratio.denominator == 1 or derived.ratio <= 0:
        raise ValueError("collar sequences are supported for rational-ratio hyperbolic saddles only")
    certificates = tuple(
        certify_collar_membership(
            target,
            derived,
            delta=delta,
            delta0=delta0,
            grid_points=grid_points,
            attempt_unique_cycle=False,
        )
        for delta in ordered
    )
    all_proved = all(cert.status == "PROVED_COLLAR" for cert in certificates)
    remainder = derived.expansion.remainder_bound
    tail_bound = sum(remainder * delta for delta in ordered)
    uniform_ok = uniform_remainder is not None and verify_uniform_flat_remainder(uniform_remainder)
    seal = make_certificate(
        claim="Monotone collar sequence toward a hyperbolic corner (compatibility only).",
        payload={
            "type": "collar_sequence",
            "source_digest": target.digest,
            "delta0": _q(delta0),
            "deltas": [_q(delta) for delta in ordered],
            "tail_bound": _q(tail_bound),
            "all_collars_proved": all_proved,
            "remainder_bound": _q(remainder),
            "grid_points": grid_points,
            "uniform_remainder_digest": uniform_remainder.source_digest if uniform_remainder else "",
        },
        honesty=build_honesty(
            collar_return_membership_proved=all_proved,
            physical_return_membership_proved=False,
            corner_window_external=True,
            graphic_finite_cyclicity_proved=False,
            full_hilbert16_solved=False,
            uniform_remainder_proved=uniform_ok,
        ),
        meta={"transcend_backend": "not_used"},
    )
    return CollarSequenceCertificate(
        target, derived, delta0, ordered, certificates, tail_bound, True, target.digest, seal
    )


def verify_collar_sequence(certificate: CollarSequenceCertificate) -> bool:
    if certificate.source_digest != certificate.target.digest or not verify_certificate_digest(certificate.seal):
        return False
    try:
        expected = certify_collar_sequence(
            certificate.target,
            certificate.derived,
            certificate.deltas,
            delta0=certificate.delta0,
        )
    except (ArithmeticError, TypeError, ValueError):
        return False
    return (
        expected.deltas == certificate.deltas
        and expected.delta0 == certificate.delta0
        and expected.tail_bound == certificate.tail_bound
        and expected.corner_window_external == certificate.corner_window_external
        and all(a.status == b.status for a, b in zip(expected.certificates, certificate.certificates, strict=True))
    )


def verify_collar_membership(certificate: CollarMembershipCertificate) -> bool:
    """Replay the collar grid comparison and unique-cycle attempt from source."""
    if (
        certificate.source_digest != certificate.target.digest
        or not verify_certificate_digest(certificate.seal)
    ):
        return False
    try:
        expected = certify_collar_membership(
            certificate.target,
            certificate.derived,
            delta=certificate.delta,
            delta0=certificate.delta0,
            grid_points=len(certificate.grid),
            attempt_unique_cycle=certificate.unique_cycle.attempted,
        )
    except (ArithmeticError, TypeError, ValueError):
        return False
    return expected == certificate
