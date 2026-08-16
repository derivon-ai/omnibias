# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""SU(3) Wilson character transfer from an enclosed Haar integral.

The single-link Wilson weight ``exp((β/3) Re χ_fund)`` is expanded on
SU(3) characters by integrating against the Weyl measure on the maximal
torus.  Coefficients use a midpoint rule plus a locked Lipschitz
remainder on a finite box partition of ``[0, 2π]²``.  The β=0
(orthogonality) piece of a non-trivial character is locked to zero
rather than re-enclosed, so the interval width tracks ``exp(a)-1``
instead of a cancelling oscillation.

Characters with ``p, q ≤ 2`` are locked trigonometric polynomials in
``e^{iθ}, e^{iφ}``.  This is one coupling and one irrep truncation.  It
is not 4-D SU(3) Yang-Mills and not a product of ordinary ``I_n``.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, cos_iv, exp_iv, sin_iv

TWO_PI = PI_IV + PI_IV
#: Weyl volume ``∫ ρ dθ dφ = 6 (2π)² = 24 π²`` for ``ρ = ∏_{i<j} |z_i-z_j|²``.
HAAR_VOLUME = Interval.from_value(24) * PI_IV * PI_IV

#: ``|χ_{p,q}| ≤ dim(p,q)``.
_CHI_ABS: dict[tuple[int, int], int] = {
    (0, 0): 1,
    (1, 0): 3,
    (0, 1): 3,
    (1, 1): 8,
    (2, 0): 6,
    (0, 2): 6,
    (2, 1): 15,
    (1, 2): 15,
    (2, 2): 27,
}

#: Crude ``|∇χ|_1`` majorants (degree × dimension × 2 angles).
_CHI_GRAD: dict[tuple[int, int], int] = {
    (0, 0): 0,
    (1, 0): 6,
    (0, 1): 6,
    (1, 1): 24,
    (2, 0): 24,
    (0, 2): 24,
    (2, 1): 60,
    (1, 2): 60,
    (2, 2): 108,
}

_RHO_ABS = 64
_RHO_GRAD = 256
_RE_FUND_GRAD = 4


def _center(index: int, n_cells: int) -> Interval:
    """Cell midpoint ``π (2i+1) / n``, a rational multiple of ``π``."""
    return PI_IV * Interval.from_value(Fraction(2 * index + 1, n_cells))


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


def _re_character(dynkin: tuple[int, int], theta: Interval, phi: Interval) -> Interval:
    """Locked real SU(3) characters for ``p, q ≤ 2`` (Weyl polynomials)."""
    re_f = _re_fund(theta, phi)
    im_f = _im_fund(theta, phi)
    if dynkin == (0, 0):
        return Interval.point(1.0)
    if dynkin in ((1, 0), (0, 1)):
        return re_f
    if dynkin == (1, 1):
        return (re_f * re_f + im_f * im_f) - Interval.point(1.0)
    if dynkin in ((2, 0), (0, 2)):
        return re_f * re_f - im_f * im_f - re_f
    if dynkin in ((2, 1), (1, 2)):
        amp = re_f * re_f - im_f * im_f - re_f
        imag = (re_f + re_f) * im_f + im_f
        return amp * re_f + imag * im_f - re_f
    if dynkin == (2, 2):
        amp = re_f * re_f - im_f * im_f - re_f
        imag = (re_f + re_f) * im_f + im_f
        return amp * amp + imag * imag - re_f * re_f - im_f * im_f
    raise ValueError(
        "enclosed Haar characters are locked for (p,q) with p,q <= 2; "
        f"got {dynkin}"
    )


def _weight_minus_one(beta: Interval, re_chi: Interval) -> Interval:
    argument = beta * Interval.from_value(Fraction(1, 3)) * re_chi
    return exp_iv(argument) - Interval.point(1.0)


def _integrand_lipschitz(beta: Interval, dynkin: tuple[int, int]) -> Interval:
    """Crude ``|∇(χ (e^a-1) ρ)|_1`` majorant, locked from derivative bounds."""
    chi_max = Interval.from_value(_CHI_ABS[dynkin])
    dchi = Interval.from_value(_CHI_GRAD[dynkin])
    rho_max = Interval.from_value(_RHO_ABS)
    drho = Interval.from_value(_RHO_GRAD)
    exp_b = exp_iv(beta.abs())
    em1 = exp_b - Interval.point(1.0)
    da = beta.abs() * Interval.from_value(Fraction(_RE_FUND_GRAD, 3))
    return dchi * em1 * rho_max + chi_max * exp_b * da * rho_max + chi_max * em1 * drho


def su3_wilson_haar_coefficient(
    dynkin: tuple[int, int],
    beta: float | Fraction,
    *,
    n_cells: int = 8,
) -> Interval:
    """Enclose ``∫ χ_R^* (e^{(β/3) Re χ_f}-1) ρ dθ dφ`` plus the trivial volume.

    Midpoint evaluation at rational multiples of ``π``, plus a Lipschitz
    remainder ``Lip × (cell side) × area`` per cell.  For ``R ≠ 1`` the
    locked orthogonality ``∫ χ_R^* ρ = 0`` is used.  For the trivial
    representation the locked volume ``24 π²`` is added.
    """
    if n_cells < 2:
        raise ValueError(f"n_cells must be >= 2, got {n_cells}")
    if dynkin[0] < 0 or dynkin[1] < 0:
        raise ValueError(f"dynkin labels must be non-negative, got {dynkin}")
    if dynkin[0] > 2 or dynkin[1] > 2:
        raise ValueError(
            "enclosed Haar characters are locked for (p,q) with p,q <= 2; "
            f"got {dynkin}"
        )
    argument = Interval.from_value(beta)
    if argument.lo <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta!r}")
    side = TWO_PI * Interval.from_value(Fraction(1, n_cells))
    area = side * side
    lip = _integrand_lipschitz(argument, dynkin)
    rem_hi = (lip * side * area).hi
    remainder = Interval(-rem_hi, rem_hi)
    total = Interval.point(0.0)
    for i in range(n_cells):
        theta = _center(i, n_cells)
        for j in range(n_cells):
            phi = _center(j, n_cells)
            density = _haar_density(theta, phi)
            re_chi = _re_fund(theta, phi)
            shift = _weight_minus_one(argument, re_chi)
            character = _re_character(dynkin, theta, phi)
            total = total + character * shift * density * area + remainder
    if dynkin == (0, 0):
        return total + HAAR_VOLUME
    return total


__all__ = [
    "HAAR_VOLUME",
    "su3_wilson_haar_coefficient",
]
