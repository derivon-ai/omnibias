# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""SU(3) Wilson character transfer from an enclosed Haar integral.

The single-link Wilson weight ``exp((β/3) Re χ_fund)`` is expanded on
SU(3) characters by integrating against the Weyl measure on the maximal
torus.  Coefficients are enclosed by interval arithmetic on a finite
box partition of ``[0, 2π]²``.  The β=0 (orthogonality) piece of a
non-trivial character is locked to zero rather than re-enclosed, so the
interval width tracks ``exp(a)-1`` instead of a cancelling oscillation.

This is one coupling and one irrep truncation.  It is not 4-D SU(3)
Yang-Mills and not a product of ordinary ``I_n``.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, cos_iv, exp_iv, sin_iv

TWO_PI = PI_IV + PI_IV
#: Weyl volume ``∫ ρ dθ dφ = 6 (2π)² = 24 π²`` for ``ρ = ∏_{i<j} |z_i-z_j|²``.
HAAR_VOLUME = Interval.from_value(24) * PI_IV * PI_IV


def _cell(index: int, n_cells: int) -> Interval:
    scale = TWO_PI * Interval.from_value(Fraction(1, n_cells))
    return scale * Interval.from_value(index) + scale * Interval(0.0, 1.0)


def _haar_density(theta: Interval, phi: Interval) -> Interval:
    """Unnormalized Weyl density ``∏_{i<j} |e^{iφ_i}-e^{iφ_j}|²``."""
    s1 = sin_iv((theta - phi) * Interval.from_value(Fraction(1, 2)))
    s2 = sin_iv((theta + theta + phi) * Interval.from_value(Fraction(1, 2)))
    s3 = sin_iv((theta + phi + phi) * Interval.from_value(Fraction(1, 2)))
    return Interval.from_value(64) * (s1 * s1) * (s2 * s2) * (s3 * s3)


def _re_fund(theta: Interval, phi: Interval) -> Interval:
    return cos_iv(theta) + cos_iv(phi) + cos_iv(theta + phi)


def _im_fund(theta: Interval, phi: Interval) -> Interval:
    return sin_iv(theta) + sin_iv(phi) - sin_iv(theta + phi)


def _weight_minus_one(beta: Interval, re_chi: Interval) -> Interval:
    argument = beta * Interval.from_value(Fraction(1, 3)) * re_chi
    return exp_iv(argument) - Interval.point(1.0)


def su3_wilson_haar_coefficient(
    dynkin: tuple[int, int],
    beta: float | Fraction,
    *,
    n_cells: int = 10,
) -> Interval:
    """Enclose ``∫ χ_R^* (e^{(β/3) Re χ_f}-1) ρ dθ dφ`` plus the trivial volume.

    For ``R ≠ 1`` the locked orthogonality ``∫ χ_R^* ρ = 0`` is used, so
    the returned interval is that difference integral.  For the trivial
    representation the locked volume ``24 π²`` is added.
    """
    if n_cells < 2:
        raise ValueError(f"n_cells must be >= 2, got {n_cells}")
    if dynkin[0] < 0 or dynkin[1] < 0:
        raise ValueError(f"dynkin labels must be non-negative, got {dynkin}")
    if dynkin[0] > 1 or dynkin[1] > 1:
        raise ValueError(
            "enclosed Haar characters are locked for (p,q) with p,q <= 1; "
            f"got {dynkin}"
        )
    argument = Interval.from_value(beta)
    if argument.lo <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta!r}")
    area = (TWO_PI * Interval.from_value(Fraction(1, n_cells))) ** 2
    total = Interval.point(0.0)
    for i in range(n_cells):
        theta = _cell(i, n_cells)
        for j in range(n_cells):
            phi = _cell(j, n_cells)
            density = _haar_density(theta, phi)
            re_chi = _re_fund(theta, phi)
            shift = _weight_minus_one(argument, re_chi)
            if dynkin == (0, 0):
                character = Interval.point(1.0)
            elif dynkin in ((1, 0), (0, 1)):
                character = re_chi
            else:
                imag = _im_fund(theta, phi)
                character = (re_chi * re_chi + imag * imag) - Interval.point(1.0)
            total = total + character * shift * density * area
    if dynkin == (0, 0):
        return total + HAAR_VOLUME
    return total


__all__ = [
    "HAAR_VOLUME",
    "su3_wilson_haar_coefficient",
]
