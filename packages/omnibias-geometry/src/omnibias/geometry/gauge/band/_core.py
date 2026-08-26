# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wilson-line holonomy band (theory 02-14).

Closed form only in the abelian and transverse-constant regimes. Open
lines are gauge-dependent. No Yang-Mills / mass-gap / continuum claim.
The gap is held finite (band), the opposite of founding ``delta -> 0``.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from enum import StrEnum

from omnibias.core.verified.interval import Interval
from omnibias.geometry.gauge._core.lie_algebra import LieAlgebra, su, u1


class BandRegime(StrEnum):
    ABELIAN = "abelian"
    TRANSVERSE_CONSTANT = "transverse_constant"
    MAGNUS = "magnus"
    PRODUCT = "product"


@dataclass(frozen=True)
class HolonomyBand:
    normal: tuple[float, ...]
    lo: float
    hi: float
    algebra: LieAlgebra
    coupling: float = 1.0

    def __post_init__(self) -> None:
        if not self.normal:
            raise ValueError("normal must be non-empty")
        if self.hi == self.lo:
            raise ValueError("holonomy band needs a finite gap (hi != lo)")
        if not math.isfinite(self.coupling):
            raise ValueError("coupling must be finite")

    @property
    def width(self) -> float:
        return float(self.hi) - float(self.lo)


def classify_regime(
    algebra: LieAlgebra,
    *,
    transverse_constant: bool,
    request_magnus: bool = False,
) -> BandRegime:
    """Conservative: never a closed-form regime for a path-ordered connection."""
    if algebra.name == "u(1)":
        return BandRegime.ABELIAN
    if transverse_constant:
        return BandRegime.TRANSVERSE_CONSTANT
    if request_magnus:
        return BandRegime.MAGNUS
    return BandRegime.PRODUCT


_U1_GENERATOR = 1.0 / math.sqrt(2.0)  # matches ``LieAlgebra.generators`` for u(1)


def abelian_holonomy(*, a0: float, lo: float, hi: float, coupling: float) -> complex:
    """``A(z) = a0 sigma'(z)`` with ``sigma = tanh``: flux is the antiderivative window.

    The u(1) generator in this package is ``1/sqrt(2)`` (fundamental
    normalization ``tr(T^2) = 1/2``), so the phase matches
    :func:`parallel_transport_from_arrays`.
    """
    flux = float(a0) * (math.tanh(float(hi)) - math.tanh(float(lo)))
    return cmath.exp(-1j * float(coupling) * flux * _U1_GENERATOR)


def su2_transverse_constant(
    components: tuple[float, float, float],
    *,
    length: float,
    coupling: float,
) -> tuple[complex, complex, complex, complex]:
    """Rodrigues formula for ``U = exp(-i g (A^a T^a) L)`` with ``T^a = sigma^a / 2``.

    Returns ``(U00, U01, U10, U11)``.
    """
    ax, ay, az = (float(v) for v in components)
    mag = math.hypot(ax, math.hypot(ay, az))
    theta = 0.5 * float(coupling) * mag * float(length)
    if mag == 0.0:
        return (1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j)
    nx, ny, nz = ax / mag, ay / mag, az / mag
    c, s = math.cos(theta), math.sin(theta)
    # U = c I - i s (n · sigma)
    u00 = c - 1j * s * nz
    u01 = -1j * s * (nx - 1j * ny)
    u10 = -1j * s * (nx + 1j * ny)
    u11 = c + 1j * s * nz
    return (u00, u01, u10, u11)


def magnus_truncation_bound(*, a_norm: float, length: float, order: int) -> Interval:
    """Lagrange-style remainder. Refuses outside the classical Magnus radius."""
    if order < 1:
        raise ValueError("Magnus order must be >= 1")
    ml = abs(float(a_norm)) * abs(float(length))
    if ml >= math.pi:
        raise ValueError(
            "Magnus series is outside its convergence radius "
            f"(||A|| L = {ml} >= pi); refusing rather than truncating"
        )
    fact = 1.0
    for i in range(1, order + 2):
        fact *= float(i)
    cap = (ml ** (order + 1)) / fact * math.exp(ml)
    return Interval(-cap, cap)


def open_line_is_gauge_dependent() -> bool:
    """Open Wilson lines are gauge dependent. Features must close the loop."""
    return True


@dataclass(frozen=True)
class RandomU1Gauge:
    """Endpoint values of a U(1) gauge, ``g(x) = exp(-i coupling λ(x) T)``.

    ``T`` is the package's ``1/sqrt(2)`` generator, matching
    :func:`abelian_holonomy`.
    """

    lambda_lo: float
    lambda_hi: float


def random_u1_gauge(*, seed: int) -> RandomU1Gauge:
    """Random endpoint gauge on the open band (leftover #35 API)."""
    rng = random.Random(int(seed))
    return RandomU1Gauge(
        lambda_lo=rng.uniform(-math.pi, math.pi),
        lambda_hi=rng.uniform(-math.pi, math.pi),
    )


def u1_gauge_element(*, lam: float, coupling: float) -> complex:
    """``g = exp(-i coupling λ T)`` with ``T = 1/sqrt(2)``."""
    return cmath.exp(-1j * float(coupling) * float(lam) * _U1_GENERATOR)


def conjugate_open_holonomy(
    u: complex, gauge: RandomU1Gauge, *, coupling: float
) -> complex:
    """``g(x_hi) U g(x_lo)^{-1}`` for an open U(1) holonomy."""
    g_hi = u1_gauge_element(lam=gauge.lambda_hi, coupling=coupling)
    g_lo = u1_gauge_element(lam=gauge.lambda_lo, coupling=coupling)
    return g_hi * complex(u) * g_lo.conjugate()


def abelian_holonomy_gauged(
    *,
    a0: float,
    lo: float,
    hi: float,
    coupling: float,
    gauge: RandomU1Gauge,
) -> complex:
    """Holonomy of ``A' = A + dλ``. Extra flux is ``λ(hi) - λ(lo)``."""
    u = abelian_holonomy(a0=a0, lo=lo, hi=hi, coupling=coupling)
    extra = u1_gauge_element(lam=gauge.lambda_hi - gauge.lambda_lo, coupling=coupling)
    return u * extra


def random_gauge_covariance_ulps(
    *,
    a0: float,
    lo: float,
    hi: float,
    coupling: float,
    seed: int,
) -> float:
    """Ulp error of ``U'`` versus ``g(hi) U g(lo)^{-1}`` on a random gauge."""
    gauge = random_u1_gauge(seed=seed)
    u = abelian_holonomy(a0=a0, lo=lo, hi=hi, coupling=coupling)
    transformed = abelian_holonomy_gauged(
        a0=a0, lo=lo, hi=hi, coupling=coupling, gauge=gauge
    )
    conjugated = conjugate_open_holonomy(u, gauge, coupling=coupling)
    err = abs(transformed - conjugated)
    return float(err / math.ulp(1.0))


__all__ = [
    "BandRegime",
    "HolonomyBand",
    "RandomU1Gauge",
    "abelian_holonomy",
    "abelian_holonomy_gauged",
    "classify_regime",
    "conjugate_open_holonomy",
    "magnus_truncation_bound",
    "open_line_is_gauge_dependent",
    "random_gauge_covariance_ulps",
    "random_u1_gauge",
    "su",
    "su2_transverse_constant",
    "u1",
    "u1_gauge_element",
]
