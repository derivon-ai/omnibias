# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Pulse envelope with exact-D logistic rates (theory 07-12).

``P = s t`` in occupancy coordinates ``s = sigma(v)``, ``t = sigma(L-v)``
rises then viscously decays. ``P'`` is the product rule plus the Riccati
identity ``sigma' = sigma (1 - sigma)``, read from the closed-form
sigmoid tower, never finite differences.

Cutoff tails vanishing with all derivatives at ``tau = 0`` stay an
external premise. The certified tail is :func:`omnibias.core.mollifier.tail_bound`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.mollifier import MollifierSpec, tail_bound, true_outside_mass
from omnibias.core.multipack import PackSpec
from omnibias.core.verified.coeffs import sigmoid_poly_coeffs_exact
from omnibias.core.verified.interval import Interval

LOCKED_L = Fraction(2)
GROWTH_RATE = Fraction(1, 2)
DECAY_RATE = Fraction(-1, 2)

_FORBIDDEN = (
    "navier_stokes_proof_claim",
    "forced_blowup_reproof_claim",
    "continuum_pde_claim",
)


def _as_frac(value: Fraction | int | str) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def honesty_payload() -> dict[str, bool]:
    return {
        "unproven_claim": False,
        "navier_stokes_proof_claim": False,
        "forced_blowup_reproof_claim": False,
        "continuum_pde_claim": False,
        "cutoff_tail_external": True,
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


def tower_sigma_prime(s: Fraction) -> Fraction:
    """``P_1(s) = sigma'`` as a polynomial in ``s``, exact over ``Q``."""
    coeffs = sigmoid_poly_coeffs_exact(1)
    acc = Fraction(0)
    power = Fraction(1)
    ss = _as_frac(s)
    for coeff in coeffs:
        acc += Fraction(coeff) * power
        power *= ss
    return acc


def riccati_sigma_prime(s: Fraction) -> Fraction:
    ss = _as_frac(s)
    return ss * (1 - ss)


def pulse_value(s: Fraction, t: Fraction) -> Fraction:
    return _as_frac(s) * _as_frac(t)


def pulse_derivative(s: Fraction, t: Fraction) -> Fraction:
    """``d/dv [sigma(v) sigma(L-v)]`` in ``(s, t)`` coordinates."""
    ss = _as_frac(s)
    tt = _as_frac(t)
    return riccati_sigma_prime(ss) * tt - ss * riccati_sigma_prime(tt)


@dataclass(frozen=True)
class PulseEnvelope:
    """Named logistic product envelope on occupancy coordinates."""

    s: Fraction
    t: Fraction
    name: str = "locked"

    def value(self) -> Fraction:
        return pulse_value(self.s, self.t)

    def derivative(self) -> Fraction:
        return pulse_derivative(self.s, self.t)

    def tower_derivative(self) -> Fraction:
        ss = self.s
        tt = self.t
        return tower_sigma_prime(ss) * tt - ss * tower_sigma_prime(tt)


def locked_growth_envelope() -> PulseEnvelope:
    """``s = 1/4``, ``t = 3/4``: ``P' = 3/32 = (1/2) P``."""
    return PulseEnvelope(Fraction(1, 4), Fraction(3, 4), name="growth")


def locked_decay_envelope() -> PulseEnvelope:
    """``s = 3/4``, ``t = 1/4``: ``P' = -3/32 = (-1/2) P``."""
    return PulseEnvelope(Fraction(3, 4), Fraction(1, 4), name="decay")


def locked_mid_envelope() -> PulseEnvelope:
    return PulseEnvelope(Fraction(1, 2), Fraction(1, 2), name="mid")


def locked_mollifier() -> MollifierSpec:
    return MollifierSpec(
        base="logistic",
        scale=1.0,
        packs=(PackSpec(order=0, mean=0.0, weight=1.0),),
        pack_scales=(1.0,),
    )


def mollifier_tail_contains_truth(*, half_width: float = 3.0) -> bool:
    spec = locked_mollifier()
    bound = tail_bound(spec, half_width=half_width)
    truth = true_outside_mass(spec, half_width=half_width)
    return bound.contains(truth)


def mollifier_tail_interval(*, half_width: float = 3.0) -> Interval:
    return tail_bound(locked_mollifier(), half_width=half_width)


__all__ = [
    "DECAY_RATE",
    "GROWTH_RATE",
    "LOCKED_L",
    "PulseEnvelope",
    "assert_honesty",
    "honesty_payload",
    "locked_decay_envelope",
    "locked_growth_envelope",
    "locked_mid_envelope",
    "locked_mollifier",
    "mollifier_tail_contains_truth",
    "mollifier_tail_interval",
    "pulse_derivative",
    "pulse_value",
    "riccati_sigma_prime",
    "tower_sigma_prime",
]
