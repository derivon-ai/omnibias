# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""SU(3) Wilson character transfer from an enclosed Haar integral.

The single-link Wilson weight ``exp((β/3) Re χ_fund)`` is expanded on
SU(3) characters by integrating against the Weyl measure on the maximal
torus.  Coefficients use a cellwise interval range times cell area on a
finite box partition of ``[0, 2π]²``.  The β=0 (orthogonality) piece of
a non-trivial character is locked to zero rather than re-enclosed, so
the interval width tracks ``exp(a)-1`` instead of a cancelling
oscillation.

Characters with ``p, q ≤ 2`` are locked trigonometric polynomials in
``e^{iθ}, e^{iφ}``.  Labels with ``p, q = 3`` are obtained from those
by the SU(3) Clebsch identity
``χ_{(p,q)} χ_{(1,0)} = χ_{(p+1,q)} + χ_{(p-1,q+1)} + χ_{(p,q-1)}``
(and ``(n,0) ⊗ (0,n)`` for ``(3,3)``).  This is one coupling and one
irrep truncation.  It is not 4-D SU(3) Yang-Mills and not a product of
ordinary ``I_n``.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import PI_IV, cos_iv, exp_iv, sin_iv

TWO_PI = PI_IV + PI_IV
#: Weyl volume ``∫ ρ dθ dφ = 6 (2π)² = 24 π²`` for ``ρ = ∏_{i<j} |z_i-z_j|²``.
HAAR_VOLUME = Interval.from_value(24) * PI_IV * PI_IV

def su3_dimension(p: int, q: int) -> int:
    """Weyl dimension ``(p+1)(q+1)(p+q+2)/2``."""
    return (int(p) + 1) * (int(q) + 1) * (int(p) + int(q) + 2) // 2


def _chi_abs(dynkin: tuple[int, int]) -> int:
    return su3_dimension(dynkin[0], dynkin[1])


def _chi_grad(dynkin: tuple[int, int]) -> int:
    """Crude ``|∇χ|_1`` majorant; locked tighter values for ``p, q ≤ 2``."""
    locked = {
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
    if dynkin in locked:
        return locked[dynkin]
    degree = dynkin[0] + dynkin[1]
    return 2 * degree * _chi_abs(dynkin)

_RHO_ABS = 64
_RHO_GRAD = 256
_RE_FUND_GRAD = 4


def _cell_box(index: int, n_cells: int) -> Interval:
    """Closed cell ``[2π i/n, 2π (i+1)/n]``, rational multiples of ``π``."""
    left = PI_IV * Interval.from_value(Fraction(2 * index, n_cells))
    right = PI_IV * Interval.from_value(Fraction(2 * index + 2, n_cells))
    return Interval(left.lo, right.hi)


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


def _mul(
    left: tuple[Interval, Interval], right: tuple[Interval, Interval]
) -> tuple[Interval, Interval]:
    re_l, im_l = left
    re_r, im_r = right
    return (re_l * re_r - im_l * im_r, re_l * im_r + im_l * re_r)


def _sub(
    left: tuple[Interval, Interval], right: tuple[Interval, Interval]
) -> tuple[Interval, Interval]:
    return (left[0] - right[0], left[1] - right[1])


def _zero_character() -> tuple[Interval, Interval]:
    return (Interval.point(0.0), Interval.point(0.0))


def _complex_character(
    dynkin: tuple[int, int], theta: Interval, phi: Interval
) -> tuple[Interval, Interval]:
    """Complex SU(3) character: locked ``p, q ≤ 2``, Clebsch above that."""
    p, q = dynkin
    if p < 0 or q < 0:
        return _zero_character()
    re_f = _re_fund(theta, phi)
    im_f = _im_fund(theta, phi)
    if dynkin == (0, 0):
        return (Interval.point(1.0), Interval.point(0.0))
    if dynkin == (1, 0):
        return (re_f, im_f)
    if dynkin == (0, 1):
        return (re_f, Interval.point(0.0) - im_f)
    if dynkin == (1, 1):
        return ((re_f * re_f + im_f * im_f) - Interval.point(1.0), Interval.point(0.0))
    if dynkin == (2, 0):
        return (re_f * re_f - im_f * im_f - re_f, (re_f + re_f) * im_f + im_f)
    if dynkin == (0, 2):
        re_c, im_c = _complex_character((2, 0), theta, phi)
        return (re_c, Interval.point(0.0) - im_c)
    if dynkin == (2, 1):
        # χ_{(1,1)} χ_{(1,0)} = χ_{(2,1)} + χ_{(0,2)} + χ_{(1,0)}
        adj = _complex_character((1, 1), theta, phi)
        fund = (re_f, im_f)
        return _sub(
            _sub(_mul(adj, fund), _complex_character((0, 2), theta, phi)),
            fund,
        )
    if dynkin == (1, 2):
        re_c, im_c = _complex_character((2, 1), theta, phi)
        return (re_c, Interval.point(0.0) - im_c)
    if dynkin == (2, 2):
        amp = re_f * re_f - im_f * im_f - re_f
        imag = (re_f + re_f) * im_f + im_f
        return (amp * amp + imag * imag - re_f * re_f - im_f * im_f, Interval.point(0.0))
    if p > 3 or q > 3:
        raise ValueError(
            "enclosed Haar characters are locked for (p,q) with p,q <= 3; "
            f"got {dynkin}"
        )
    if p < q:
        re_c, im_c = _complex_character((q, p), theta, phi)
        return (re_c, Interval.point(0.0) - im_c)
    if dynkin == (3, 3):
        # (3,0) ⊗ (0,3) = (3,3) ⊕ (2,2) ⊕ (1,1) ⊕ (0,0)
        chi30 = _complex_character((3, 0), theta, phi)
        chi03 = _complex_character((0, 3), theta, phi)
        return _sub(
            _sub(
                _sub(_mul(chi30, chi03), _complex_character((2, 2), theta, phi)),
                _complex_character((1, 1), theta, phi),
            ),
            _complex_character((0, 0), theta, phi),
        )
    # χ_{(p,q)} = χ_{(p-1,q)} χ_{(1,0)} - χ_{(p-2,q+1)} - χ_{(p-1,q-1)}
    fund = _complex_character((1, 0), theta, phi)
    return _sub(
        _sub(
            _mul(_complex_character((p - 1, q), theta, phi), fund),
            _complex_character((p - 2, q + 1), theta, phi),
        ),
        _complex_character((p - 1, q - 1), theta, phi),
    )


def _re_character(dynkin: tuple[int, int], theta: Interval, phi: Interval) -> Interval:
    """Real part of the SU(3) character (locked ≤2, Clebsch ≤3)."""
    return _complex_character(dynkin, theta, phi)[0]


def _weight_minus_one(beta: Interval, re_chi: Interval) -> Interval:
    argument = beta * Interval.from_value(Fraction(1, 3)) * re_chi
    return exp_iv(argument) - Interval.point(1.0)


def _integrand_lipschitz(beta: Interval, dynkin: tuple[int, int]) -> Interval:
    """Crude ``|∇(χ (e^a-1) ρ)|_1`` majorant, locked from derivative bounds."""
    chi_max = Interval.from_value(_chi_abs(dynkin))
    dchi = Interval.from_value(_chi_grad(dynkin))
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

    Cellwise interval range of the integrand times cell area on the
    boxes ``[2π i/n, 2π (i+1)/n]²``.  For ``R ≠ 1`` the locked
    orthogonality ``∫ χ_R^* ρ = 0`` is used.  For the trivial
    representation the locked volume ``24 π²`` is added.
    """
    if n_cells < 2:
        raise ValueError(f"n_cells must be >= 2, got {n_cells}")
    if dynkin[0] < 0 or dynkin[1] < 0:
        raise ValueError(f"dynkin labels must be non-negative, got {dynkin}")
    if dynkin[0] > 3 or dynkin[1] > 3:
        raise ValueError(
            "enclosed Haar characters are locked for (p,q) with p,q <= 3; "
            f"got {dynkin}"
        )
    argument = Interval.from_value(beta)
    if argument.lo <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta!r}")
    side = TWO_PI * Interval.from_value(Fraction(1, n_cells))
    area = side * side
    total = Interval.point(0.0)
    for i in range(n_cells):
        theta = _cell_box(i, n_cells)
        for j in range(n_cells):
            phi = _cell_box(j, n_cells)
            density = _haar_density(theta, phi)
            re_chi = _re_fund(theta, phi)
            shift = _weight_minus_one(argument, re_chi)
            character = _re_character(dynkin, theta, phi)
            total = total + character * shift * density * area
    if dynkin == (0, 0):
        return total + HAAR_VOLUME
    return total


__all__ = [
    "HAAR_VOLUME",
    "su3_dimension",
    "su3_wilson_haar_coefficient",
]
