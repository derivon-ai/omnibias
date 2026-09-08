# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Radial ``m = 1`` swirl-heat identity (theory 07-12).

The exterior swirl ``K = r^{-1-2h} H(tau / r^2)`` solves the radial
``m = 1`` heat equation when ``H`` is a heat profile. The CI plant is
``h = 0``, ``H == 1``, so ``K = 1/r`` and the radial Laplacian vanishes
identically over ``Q``.

The paper uses ``h > 0``. Anisotropic scaling ``h = 1/200`` is recorded
as leftover if it leaves the holonomic fragment; the fragment is the
radial operator, not the anisotropic scaling. Not a 3-D heat theorem,
not Osterwalder–Schrader, not a continuum mass gap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.verified.interval import Interval

LOCKED_H = Fraction(0)
LOCKED_R = Fraction(1)
LOCKED_TAU = Fraction(1)
ANISOTROPIC_H = Fraction(1, 200)

_FORBIDDEN = (
    "navier_stokes_proof_claim",
    "forced_blowup_reproof_claim",
    "continuum_pde_claim",
    "three_d_heat_theorem",
)


def _as_frac(value: Fraction | int | str) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def honesty_payload() -> dict[str, bool]:
    return {
        "unproven_claim": False,
        "navier_stokes_proof_claim": False,
        "forced_blowup_reproof_claim": False,
        "continuum_pde_claim": False,
        "three_d_heat_theorem": False,
        "continuum_navier_stokes_claim": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "paper_uses_h_positive": True,
        "ci_plant_h_zero": True,
    }


def assert_honesty(payload: Mapping[str, Any] | None = None) -> dict[str, bool]:
    flags = honesty_payload() if payload is None else dict(payload)
    for key in _FORBIDDEN:
        if bool(flags.get(key, False)):
            raise ValueError(f"honesty.{key} must stay False on this fragment")
    return {str(k): bool(v) for k, v in flags.items()}


def isotropic_swirl(r: Fraction, tau: Fraction, *, h: Fraction = LOCKED_H) -> Fraction:
    """``K = r^{-1-2h}`` with ``H == 1`` (constant heat profile)."""
    del tau
    rr = _as_frac(r)
    hh = _as_frac(h)
    if rr == 0:
        raise ZeroDivisionError("swirl heat at r = 0")
    exponent = 1 + 2 * hh
    if exponent.denominator != 1:
        raise ValueError("h must make 1+2h an integer on this fragment")
    power = int(exponent)
    acc = Fraction(1)
    for _ in range(power):
        acc *= rr
    return 1 / acc


def radial_m1_laplacian(r: Fraction, K: Fraction, Kr: Fraction, Krr: Fraction) -> Fraction:
    """``(partial_r^2 + r^{-1} partial_r - r^{-2}) K``."""
    rr = _as_frac(r)
    if rr == 0:
        raise ZeroDivisionError("radial Laplacian at r = 0")
    return Krr + Kr / rr - K / (rr * rr)


def swirl_heat_residual(
    r: Fraction = LOCKED_R,
    tau: Fraction = LOCKED_TAU,
    *,
    h: Fraction = LOCKED_H,
) -> Fraction:
    """``-partial_tau K - radial_m1(K)`` for the constant-``H`` plant.

    ``H == 1`` is independent of ``tau``, so ``partial_tau K = 0``.
    """
    del tau
    rr = _as_frac(r)
    hh = _as_frac(h)
    K = isotropic_swirl(rr, Fraction(0), h=hh)
    exponent = int(1 + 2 * hh)
    # K = r^{-m}, K_r = -m r^{-m-1}, K_rr = m(m+1) r^{-m-2}.
    m = exponent
    Kr = Fraction(-m) / (rr ** (m + 1))
    Krr = Fraction(m * (m + 1)) / (rr ** (m + 2))
    return -radial_m1_laplacian(rr, K, Kr, Krr)


def swirl_heat_residual_interval(
    r: Fraction = LOCKED_R,
    tau: Fraction = LOCKED_TAU,
    *,
    h: Fraction = LOCKED_H,
) -> Interval:
    residual = swirl_heat_residual(r, tau, h=h)
    return Interval.from_rational(residual)


def anisotropic_heat_in_fragment() -> bool:
    """``h = 1/200`` leaves ``K = r^{-1-2h}`` outside integer-power ``Q``."""
    return False


@dataclass(frozen=True)
class SwirlHeatPlant:
    h: Fraction
    r: Fraction
    tau: Fraction

    def residual(self) -> Fraction:
        return swirl_heat_residual(self.r, self.tau, h=self.h)


def locked_swirl_heat_plant() -> SwirlHeatPlant:
    return SwirlHeatPlant(h=LOCKED_H, r=LOCKED_R, tau=LOCKED_TAU)


__all__ = [
    "ANISOTROPIC_H",
    "LOCKED_H",
    "LOCKED_R",
    "LOCKED_TAU",
    "SwirlHeatPlant",
    "anisotropic_heat_in_fragment",
    "assert_honesty",
    "honesty_payload",
    "isotropic_swirl",
    "locked_swirl_heat_plant",
    "radial_m1_laplacian",
    "swirl_heat_residual",
    "swirl_heat_residual_interval",
]
