# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Fixed-order weighted coefficient bounds (theory 07-11).

Definitions 6.4–6.5 of the forced-blowup construction are seminorms
``W_alpha``, ``M_alpha``, ``S_alpha`` with edge weights ``zeta``,
``delta``, a pulse envelope ``P(v)``, and ``eps = Q^h``. This module
checks a **finite-order** pointwise bound and an optional finite-order
Gevrey majorant ``|a_k| <= C k!^s rho^k`` only for ``k <= k_max``.

It is not a Gevrey-class theorem. Infinite-class membership stays out
of Lean. :mod:`omnibias.core.verified.sequence_space` remains the
geometric ``nu^k`` path and is not replaced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Any

from omnibias.core.verified.interval import Interval, IntervalLike

_FORBIDDEN = (
    "gevrey_class_claim",
    "navier_stokes_proof_claim",
    "forced_blowup_reproof_claim",
    "continuum_pde_claim",
)


def _as_frac(value: Fraction | int | str) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def honesty_payload() -> dict[str, bool]:
    return {
        "unproven_claim": False,
        "gevrey_class_claim": False,
        "navier_stokes_proof_claim": False,
        "forced_blowup_reproof_claim": False,
        "continuum_pde_claim": False,
        "continuum_navier_stokes_claim": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def assert_honesty(payload: Mapping[str, Any] | None = None) -> dict[str, bool]:
    flags = honesty_payload() if payload is None else dict(payload)
    for key in _FORBIDDEN:
        if bool(flags.get(key, False)):
            raise ValueError(f"honesty.{key} must stay False on this fragment")
    return {str(k): bool(v) for k, v in flags.items()}


@dataclass(frozen=True)
class EdgeWeights:
    """Positive edge weights ``zeta``, ``delta`` of the pulse metric."""

    zeta: Fraction
    delta: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "zeta", _as_frac(self.zeta))
        object.__setattr__(self, "delta", _as_frac(self.delta))
        if self.zeta <= 0 or self.delta <= 0:
            raise ValueError("edge weights must be strictly positive")


@dataclass(frozen=True)
class PulseWeight:
    """Envelope ``P(v)`` on ``0 <= v <= L_s``. Locked plant is monomial."""

    L_s: Fraction
    coeffs: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "L_s", _as_frac(self.L_s))
        cleaned = tuple(_as_frac(c) for c in self.coeffs)
        object.__setattr__(self, "coeffs", cleaned)
        if self.L_s <= 0:
            raise ValueError("L_s must be positive")

    def eval(self, v: Fraction) -> Fraction:
        if v < 0 or v > self.L_s:
            raise ValueError("v is outside [0, L_s]")
        acc = Fraction(0)
        power = Fraction(1)
        for coeff in self.coeffs:
            acc += coeff * power
            power *= v
        return acc


@dataclass(frozen=True)
class WeightedBound:
    """Finite-order bound ``C * eps^alpha`` times a stored envelope factor."""

    alpha: Fraction
    eps: Fraction
    S_star: Fraction
    C: Fraction
    b: Fraction
    d: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _as_frac(self.alpha))
        object.__setattr__(self, "eps", _as_frac(self.eps))
        object.__setattr__(self, "S_star", _as_frac(self.S_star))
        object.__setattr__(self, "C", _as_frac(self.C))
        object.__setattr__(self, "b", _as_frac(self.b))
        object.__setattr__(self, "d", _as_frac(self.d))
        if self.eps <= 0 or self.C < 0:
            raise ValueError("eps must be positive and C non-negative")
        if self.alpha.denominator != 1 or self.alpha < 0:
            raise ValueError("alpha must be a non-negative integer for this fragment")

    def rhs(self, *, envelope: Fraction = Fraction(1)) -> Fraction:
        return self.C * (self.eps ** int(self.alpha)) * envelope


@dataclass(frozen=True)
class WeightedSample:
    """One named sample of a coefficient seminorm."""

    name: str
    value: Interval
    envelope: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        boxed = (
            self.value
            if isinstance(self.value, Interval)
            else Interval.from_value(self.value)
        )
        object.__setattr__(self, "value", boxed)
        object.__setattr__(self, "envelope", _as_frac(self.envelope))


@dataclass(frozen=True)
class BoundReport:
    holds: bool
    failing: tuple[str, ...]
    bound: Fraction


def check_pointwise_bound(
    samples: Sequence[WeightedSample],
    bound: WeightedBound,
) -> BoundReport:
    """Return whether every sample magnitude lies in ``[0, rhs]``."""
    failing: list[str] = []
    rhs = bound.rhs()
    rhs_iv = Interval.from_rational(rhs)
    for sample in samples:
        allowed = bound.rhs(envelope=sample.envelope)
        allowed_iv = Interval.from_rational(allowed)
        mag = Interval.point(sample.value.mag)
        if mag.lo < 0:
            failing.append(sample.name)
            continue
        if mag.hi > allowed_iv.hi:
            failing.append(sample.name)
            continue
        if not (rhs_iv.lo >= 0.0 or allowed >= 0):
            failing.append(sample.name)
    return BoundReport(holds=not failing, failing=tuple(failing), bound=rhs)


@dataclass(frozen=True)
class GevreyReport:
    holds: bool
    k_max: int
    failing_k: tuple[int, ...]
    detail: str


def gevrey_majorant(
    coeffs: Sequence[IntervalLike | Fraction | int],
    *,
    s: Fraction | int,
    rho: Fraction | int | str,
    k_max: int,
    C: Fraction | int | str = 1,
) -> GevreyReport:
    """Check ``|a_k| <= C k!^s rho^k`` for ``k <= k_max`` only.

    Refuses ``k > k_max`` as out-of-fragment. Integer ``s`` only, so the
    comparison stays exact over ``Q``. This is not a Gevrey-class claim.
    """
    if k_max < 0:
        raise ValueError(f"k_max must be >= 0, got {k_max}")
    s_frac = _as_frac(s)
    if s_frac.denominator != 1 or s_frac < 0:
        raise ValueError("s must be a non-negative integer on this fragment")
    s_int = int(s_frac)
    rho_f = _as_frac(rho)
    C_f = _as_frac(C)
    if rho_f <= 0 or C_f < 0:
        raise ValueError("rho must be positive and C non-negative")
    if len(coeffs) > k_max + 1:
        raise ValueError(
            f"out-of-fragment: {len(coeffs) - 1} > k_max={k_max}; "
            "infinite Gevrey-class membership is not claimed"
        )
    failing: list[int] = []
    for k, raw in enumerate(coeffs):
        if k > k_max:
            raise ValueError(f"out-of-fragment: k={k} > k_max={k_max}")
        mag = Interval.from_value(raw).mag if not isinstance(raw, Fraction | int) else abs(_as_frac(raw))
        if isinstance(mag, float):
            left = Interval.point(mag)
            right = Interval.from_rational(
                C_f * (factorial(k) ** s_int) * (rho_f**k)
            )
            if left.hi > right.hi:
                failing.append(k)
        else:
            right = C_f * (factorial(k) ** s_int) * (rho_f**k)
            if _as_frac(mag) > right:
                failing.append(k)
    return GevreyReport(
        holds=not failing,
        k_max=k_max,
        failing_k=tuple(failing),
        detail="finite-order majorant; not a Gevrey-class theorem",
    )


def locked_samples() -> tuple[WeightedSample, ...]:
    """Monomial envelope on a rational grid; bound ``C=1`` is obvious by hand."""
    half = Interval.from_rational(Fraction(1, 2))
    return (
        WeightedSample("grid_0", half),
        WeightedSample("grid_1", Interval.from_rational(Fraction(1, 3))),
        WeightedSample("random_0", Interval.from_rational(Fraction(2, 5))),
    )


def violating_sample() -> WeightedSample:
    return WeightedSample("too_large", Interval.from_rational(Fraction(2)))


def locked_bound() -> WeightedBound:
    return WeightedBound(
        alpha=Fraction(0),
        eps=Fraction(1),
        S_star=Fraction(0),
        C=Fraction(1),
        b=Fraction(0),
        d=Fraction(0),
    )


__all__ = [
    "BoundReport",
    "EdgeWeights",
    "GevreyReport",
    "PulseWeight",
    "WeightedBound",
    "WeightedSample",
    "assert_honesty",
    "check_pointwise_bound",
    "gevrey_majorant",
    "honesty_payload",
    "locked_bound",
    "locked_samples",
    "violating_sample",
]
