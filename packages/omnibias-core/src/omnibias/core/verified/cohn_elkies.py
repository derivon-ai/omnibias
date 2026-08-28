# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Cohn-Elkies linear-programming bound in **fixed dimension** on the Hermite basis.

A Schwartz test function ``f`` that is a finite combination of physicists'
Hermite functions has an **exact** Fourier transform (the same coefficients
up to the diagonal ``(-i)^n`` phases of
:meth:`~omnibias.core.verified.hermite_basis.HermiteExpansion.fourier_transform_exact`).
The Cohn-Elkies theorem then says: if ``f(0) > 0``, ``hat f(0) > 0``,
``f(x) <= 0`` for ``|x| >= r``, and ``hat f >= 0`` everywhere, the
one-dimensional packing density is at most

.. math::

    \delta \;\le\; r \cdot \frac{f(0)}{\hat f(0)}.

This module certifies the sign constraints by interval evaluation on a
declared sample grid plus the exact transform, and compares the resulting
number to the **published 1-D packing density** ``1`` (the unique tiling of
the line by unit intervals).  It does **not** take a ``d -> inf`` high-
dimensional limit, does not claim Viazovska, and does not claim a new
record in any dimension ``d >= 2``.

The default test function is the two-coefficient combination
``f = psi_0 - (1/10) psi_2``, which is elementary to check by hand:
``f(x) = e^{-x^2/2}(6/5 - (2/5) x^2)`` so ``f <= 0`` for ``|x| >= sqrt(3)``,
and ``hat f = psi_0 + (1/10) psi_2 > 0`` everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omnibias.core.verified.hermite_basis import HermiteExpansion, hermite_function
from omnibias.core.verified.interval import Interval

#: The unique packing density of ``R`` by intervals of length 1.
PUBLISHED_1D_PACKING_DENSITY = 1.0


@dataclass(frozen=True)
class CohnElkiesBound:
    """A fixed-dimension (here: 1-D) Cohn-Elkies density upper bound."""

    dimension: int
    radius: float
    f_at_zero: tuple[float, float]
    hat_f_at_zero: tuple[float, float]
    density_upper: float
    published_comparison: float
    certified: bool
    detail: str
    high_dimension_limit_claim: bool = False

    def __post_init__(self) -> None:
        if self.high_dimension_limit_claim:
            raise ValueError("d->inf claims are refused by construction")
        if self.dimension != 1:
            raise ValueError("this module certifies the 1-D packing bound only")


def _even_hermite_test_function() -> HermiteExpansion:
    """``psi_0 - (1/10) psi_2`` -- even, self-dual up to a sign on the n=2 term."""
    return HermiteExpansion.from_coeffs([1.0, 0.0, -0.1])


def _grid(radius: float, n_points: int) -> tuple[float, ...]:
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    step = 0.5
    # Sample |x| > r (strictly outside, so a zero of f at the cutoff is not a refuse)
    # and a few interior hat-f points on [0, r].
    outside = tuple(radius + 0.25 + i * step for i in range(n_points))
    interior = tuple(i * radius / (n_points - 1) for i in range(n_points))
    return interior + outside


def cohn_elkies_hermite_bound(
    *,
    expansion: HermiteExpansion | None = None,
    radius: float = 2.0,
    n_samples: int = 8,
) -> CohnElkiesBound:
    """Certify a 1-D Cohn-Elkies bound from a finite even Hermite combination.

    Sign constraints are checked on a **declared finite grid**, not on the
    whole line: ``f <= 0`` at the outside nodes and ``hat f >= 0`` at the
    interior-plus-outside nodes.  Failure of any node refuses the bound
    (``certified=False``) rather than interpolating.
    """
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError(f"radius must be finite and > 0, got {radius!r}")
    fn = expansion if expansion is not None else _even_hermite_test_function()
    hat = fn.fourier_transform_exact()
    f0 = fn.evaluate_kept(0.0)
    h0 = hat.evaluate_kept(0.0)
    if f0.re.lo <= 0.0 or h0.re.lo <= 0.0:
        return CohnElkiesBound(
            dimension=1,
            radius=float(radius),
            f_at_zero=(f0.re.lo, f0.re.hi),
            hat_f_at_zero=(h0.re.lo, h0.re.hi),
            density_upper=math.inf,
            published_comparison=PUBLISHED_1D_PACKING_DENSITY,
            certified=False,
            detail="f(0) and hat f(0) must both be certified strictly positive",
        )
    nodes = _grid(radius, n_samples)
    for x in nodes:
        if abs(x) > radius:
            fx = fn.evaluate_kept(x)
            if fx.re.hi > 0.0:
                return CohnElkiesBound(
                    dimension=1,
                    radius=float(radius),
                    f_at_zero=(f0.re.lo, f0.re.hi),
                    hat_f_at_zero=(h0.re.lo, h0.re.hi),
                    density_upper=math.inf,
                    published_comparison=PUBLISHED_1D_PACKING_DENSITY,
                    certified=False,
                    detail=f"f({x}) is not certified <= 0 (hi={fx.re.hi})",
                )
        hx = hat.evaluate_kept(x)
        if hx.re.lo < 0.0:
            return CohnElkiesBound(
                dimension=1,
                radius=float(radius),
                f_at_zero=(f0.re.lo, f0.re.hi),
                hat_f_at_zero=(h0.re.lo, h0.re.hi),
                density_upper=math.inf,
                published_comparison=PUBLISHED_1D_PACKING_DENSITY,
                certified=False,
                detail=f"hat f({x}) is not certified >= 0 (lo={hx.re.lo})",
            )
    # Density <= r * f(0) / hat f(0); use the *upper* endpoint of the quotient.
    quotient = Interval(f0.re.lo, f0.re.hi) / Interval(h0.re.lo, h0.re.hi)
    density_upper = float((Interval.point(radius) * quotient).hi)
    return CohnElkiesBound(
        dimension=1,
        radius=float(radius),
        f_at_zero=(f0.re.lo, f0.re.hi),
        hat_f_at_zero=(h0.re.lo, h0.re.hi),
        density_upper=density_upper,
        published_comparison=PUBLISHED_1D_PACKING_DENSITY,
        certified=True,
        detail=(
            "1-D Cohn-Elkies bound from a finite Hermite combination on a "
            "declared sample grid; compared to the published packing density 1 "
            "(the bound is typically loose; d->inf is not claimed)"
        ),
    )


def hermite_sample_nonpositive(n: int, x: float) -> Interval:
    """Diagnostic enclosure of ``psi_n(x)`` (re-export for tests / docs)."""
    return hermite_function(n, x)


__all__ = [
    "CohnElkiesBound",
    "PUBLISHED_1D_PACKING_DENSITY",
    "cohn_elkies_hermite_bound",
    "hermite_sample_nonpositive",
]
