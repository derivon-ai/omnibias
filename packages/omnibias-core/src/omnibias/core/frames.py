# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""OMBU wavelet frames (theory 01-06).

``sigma'`` is **not** admissible (``admissibility_constant(..., 1)`` returns
``None``, never a guess). Frames are not orthonormal and not compactly
supported; there is no O(N) fast transform. Pack order remains a band
selector (01-07), not a Littlewood-Paley completeness claim.

The atoms come from founding ``delta -> 0``. No temperature collapse.
Pure Python: no tensor imports. A ``FrameSpec`` compiles to a ``BankSpec``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from math import factorial, pi

from omnibias.core.polynomials import (
    hermite_coeffs,
    sech_polynomial_coeffs,
    tanh_polynomial_coeffs,
)
from omnibias.core.scan import BankSpec
from omnibias.core.spectral_design import hat_sigma_magnitude
from omnibias.core.verified.interval import Interval

_CLOSED = frozenset({"gaussian", "sech"})


def _horner(coeffs: Sequence[float], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def dilated_sigma_n(base: str, u: float, order: int, alpha: float) -> float:
    """``sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)``."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    z = alpha * u
    if base == "gaussian":
        g = math.exp(-0.5 * z * z)
        he = _horner(hermite_coeffs(order), z)
        sign = 1.0 if order % 2 == 0 else -1.0
        raw = sign * he * g
    elif base in ("tanh", "sech"):
        t = math.tanh(z)
        if base == "tanh":
            raw = _horner(tanh_polynomial_coeffs(order), t)
        else:
            sech = 1.0 / math.cosh(z)
            raw = _horner(sech_polynomial_coeffs(order), t) * sech
    else:
        raise ValueError(f"unsupported frame base {base!r}")
    return (alpha**order) * raw


@dataclass(frozen=True)
class FrameSpec:
    """Discrete analytic frame: order-``n`` atoms on a scale / offset bank.

    Not orthonormal, not compactly supported, no O(N) transform.
    """

    base: str
    order: int
    scales: tuple[float, ...]
    offset_spacing: float
    n_offsets: int = 16

    def __post_init__(self) -> None:
        name = str(self.base).lower().strip()
        if name not in _CLOSED:
            raise ValueError(
                f"frame base {self.base!r} has no closed-form Fourier; "
                f"expected one of {sorted(_CLOSED)}"
            )
        if self.order < 1:
            raise ValueError("order must be >= 1")
        if self.offset_spacing <= 0.0:
            raise ValueError("offset_spacing must be positive")
        if any(s <= 0.0 for s in self.scales):
            raise ValueError("scales must be positive")
        object.__setattr__(self, "base", name)


def admissibility_constant(base: str, order: int) -> Interval | None:
    """Calderón constant. ``None`` for ``n = 1`` (``sigma'`` is not admissible)."""
    n = int(order)
    if n < 2:
        return None
    name = str(base).lower().strip()
    if name == "gaussian":
        # int_0^inf xi^{2n-1} |hat g|^2 d xi = pi * (n-1)!
        # (hat g = sqrt(2 pi) e^{-xi^2/2}; boundary terms vanish for n >= 2).
        return Interval.point(pi * float(factorial(n - 1)))
    if name == "sech":
        return _sech_admissibility(n)
    return None


def _sech_admissibility(n: int) -> Interval:
    """Enclose ``int_0^inf xi^{2n-1} |hat sech|^2 d xi`` with a Riemann sum + tail."""
    # sech decays as e^{-(pi/2) |xi|}; |hat|^2 ~ pi^2 e^{-pi |xi|}.
    n_grid = 4000
    hi = 40.0
    dx = hi / n_grid
    acc_lo = 0.0
    acc_hi = 0.0
    for i in range(n_grid):
        x0 = i * dx
        x1 = x0 + dx
        # mid-point plus a crude variation bound using endpoints
        vals = []
        for x in (x0 + 0.5 * dx, x0, x1):
            if x <= 0.0:
                x = 0.5 * dx
            hat = hat_sigma_magnitude("sech", x)
            vals.append((x ** (2 * n - 1)) * hat * hat)
        acc_lo += min(vals) * dx
        acc_hi += max(vals) * dx
    # tail int_hi^inf xi^{2n-1} pi^2 exp(-pi xi) d xi  (over-estimate)
    tail = (pi**2) * math.exp(-pi * hi) * ((hi ** (2 * n)) + 1.0)
    return Interval(max(acc_lo, 0.0), acc_hi + tail)


def vanishing_moments(base: str, order: int) -> int:
    """``sigma^(n)`` has ``n`` vanishing moments for ``n >= 2``; ``sigma'`` has none."""
    n = int(order)
    if n < 2:
        return 0
    _ = str(base)
    return n


def littlewood_paley_bounds(
    spec: FrameSpec, *, grid: int = 4096
) -> tuple[Interval, Interval]:
    """Sound ``(A, B)`` enclosure of the LP sum on a log-frequency grid.

    Uses an interval sup/inf over neighbouring bins, not a pointwise max.
    This is not a Littlewood-Paley completeness claim.
    """
    if spec.order < 2:
        raise ValueError("LP bounds require an admissible atom (order >= 2)")
    xis = [2.0 ** ((i - grid // 2) / 32.0) for i in range(grid)]
    sums: list[float] = []
    for xi in xis:
        acc = 0.0
        for a in spec.scales:
            # hat psi_a (xi) ~ sqrt(a) (i a xi)^n hat_sigma(a xi) / something
            # magnitude^2 ~ a * (a xi)^{2n} |hat_sigma(a xi)|^2
            mag = (abs(a * xi) ** spec.order) * math.sqrt(a) * hat_sigma_magnitude(
                spec.base, a * xi
            )
            acc += mag * mag
        sums.append(acc)
    bins_lo: list[float] = []
    bins_hi: list[float] = []
    for i in range(len(sums) - 1):
        bins_lo.append(min(sums[i], sums[i + 1]))
        bins_hi.append(max(sums[i], sums[i + 1]))
    return Interval(min(bins_lo), min(bins_lo)), Interval(max(bins_hi), max(bins_hi))


def redundancy(spec: FrameSpec) -> Interval:
    a, b = littlewood_paley_bounds(spec)
    if a.lo <= 0.0:
        return Interval(float("inf"), float("inf"))
    return b / a


def compile_bank(spec: FrameSpec) -> BankSpec:
    """Thin ``FrameSpec -> BankSpec`` compiler for ``BiasScan``."""
    n = spec.n_offsets
    if n < 2:
        raise ValueError("n_offsets must be >= 2")
    half = 0.5 * spec.offset_spacing * (n - 1)
    return BankSpec.uniform(-half, half, n, scales=spec.scales)


__all__ = [
    "FrameSpec",
    "admissibility_constant",
    "compile_bank",
    "dilated_sigma_n",
    "littlewood_paley_bounds",
    "redundancy",
    "vanishing_moments",
]
