# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous (interval) twin of the Walsh / multilinear spectra and the
certified linear- and differential-bias bounds built on them.

The exact ``{0,1}`` integer transforms in :mod:`omnibias.boolean._core.walsh` and
:mod:`omnibias.boolean._core.anf` are bit-exact and need no enclosure.  The
*verified* layer here exists for the **real-valued** spectra that omnibias makes
possible -- the Walsh / multilinear coefficients of a differentiable surrogate
(a ``tanh(beta x)`` / ``sigmoid(beta x)`` relaxation of a gate), of a noisy /
measured truth table, or of any function whose values are themselves intervals.
Every butterfly stage uses the outward-rounded
:class:`~omnibias.core.verified.interval.Interval`, so the returned coefficients
*provably contain* the true ones despite floating-point round-off, and the
cryptanalytic figures of merit (linearity, nonlinearity, linear bias,
autocorrelation, differential bias, absolute indicator) come with rigorous
two-sided bounds.

Conventions match the rest of :mod:`omnibias.boolean._core`: arrays are indexed
LSB-first by bitmask, the ``{+1,-1}`` ("spin") encoding is ``s = 1 - 2b``, and the
normalised Fourier coefficient is ``hat f(S) = 2^{-n} sum_x f(x) chi_S(x)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.boolean._core.truth_table import (
    TruthTable,
    check_truth_table,
    is_power_of_two,
    num_vars,
)
from omnibias.core.verified.interval import Interval, IntervalLike


def _check_pow2(size: int) -> int:
    if not is_power_of_two(size):
        raise ValueError(f"values length must be a power of two, got {size}")
    return size.bit_length() - 1


def _abs_iv(iv: Interval) -> Interval:
    """Enclosure of ``|iv|`` (collapses to ``[0, .]`` when the sign is undecided)."""
    if iv.lo >= 0.0:
        return iv
    if iv.hi <= 0.0:
        return Interval(-iv.hi, -iv.lo)
    return Interval(0.0, max(-iv.lo, iv.hi))


def _max_iv(values: Sequence[Interval]) -> Interval:
    """Rigorous enclosure of ``max`` of a list of interval-valued quantities.

    ``max_k [lo_k, hi_k]`` is enclosed by ``[max_k lo_k, max_k hi_k]`` (the true
    maximum lies between the largest lower bound and the largest upper bound).
    """
    return Interval(max(v.lo for v in values), max(v.hi for v in values))


# --------------------------------------------------------------------------- #
# Interval transforms (twins of walsh.py / multilinear.py).
# --------------------------------------------------------------------------- #
def walsh_hadamard_iv(values: Sequence[IntervalLike]) -> list[Interval]:
    """Outward-rounded fast Walsh-Hadamard transform (unnormalised).

    Twin of :func:`omnibias.boolean._core.walsh.walsh_hadamard`; every
    sum/difference is an interval operation, so ``result[k]`` encloses
    ``sum_x (-1)^{<k,x>} values[x]`` for every realisation inside the inputs.
    """
    a = [Interval.from_value(v) for v in values]
    size = len(a)
    _check_pow2(size)
    step = 1
    while step < size:
        for i in range(0, size, step << 1):
            for j in range(i, i + step):
                u = a[j]
                v = a[j + step]
                a[j] = u + v
                a[j + step] = u - v
        step <<= 1
    return a


def mobius_iv(values: Sequence[IntervalLike]) -> list[Interval]:
    r"""Outward-rounded real Mobius transform = interval multilinear coefficients.

    Twin of :func:`omnibias.boolean._core.multilinear.multilinear_coeffs` (the
    real-valued Reed-Muller / ANF extension); ``result[S]`` encloses the monomial
    coefficient ``m_S`` of the multilinear extension of ``values``.
    """
    a = [Interval.from_value(v) for v in values]
    size = len(a)
    _check_pow2(size)
    step = 1
    while step < size:
        for i in range(0, size, step << 1):
            for j in range(i, i + step):
                a[j + step] = a[j + step] - a[j]
        step <<= 1
    return a


def _pm1_values_iv(
    table: TruthTable | None,
    values: Sequence[IntervalLike] | None,
    encoding: str,
) -> tuple[list[Interval], int]:
    """Resolve the spin/encoded values from either an exact table or real inputs."""
    if (table is None) == (values is None):
        raise ValueError("provide exactly one of `table` or `values`")
    if table is not None:
        check_truth_table(table)
        n = num_vars(table)
        if encoding == "pm1":
            vals = [Interval.point(float(1 - 2 * v)) for v in table]
        elif encoding == "01":
            vals = [Interval.point(float(v)) for v in table]
        else:
            raise ValueError(f"encoding must be 'pm1' or '01', got {encoding!r}")
        return vals, n
    assert values is not None
    resolved = [Interval.from_value(v) for v in values]
    n = _check_pow2(len(resolved))
    return resolved, n


def fourier_coeffs_iv(
    table: TruthTable | None = None,
    *,
    values: Sequence[IntervalLike] | None = None,
    encoding: str = "pm1",
) -> tuple[Interval, ...]:
    r"""Rigorous Fourier coefficients ``hat f(S) = 2^{-n} (WHT f)(S)``.

    Pass an exact ``{0,1}`` ``table`` (with ``encoding`` ``"pm1"`` / ``"01"``) for
    the certified twin of :func:`omnibias.boolean._core.walsh.fourier_coeffs`, or
    pass real / interval ``values`` for the spectrum of a differentiable surrogate.
    """
    vals, n = _pm1_values_iv(table, values, encoding)
    wht = walsh_hadamard_iv(vals)
    scale = Interval.from_rational(Fraction(1, 1 << n))
    return tuple(c * scale for c in wht)


def walsh_spectrum_iv(
    table: TruthTable, encoding: str = "pm1"
) -> dict[frozenset[int], Interval]:
    """Interval Fourier coefficients as a ``{variable-set: enclosure}`` mapping."""
    n = num_vars(table)
    coeffs = fourier_coeffs_iv(table, encoding=encoding)
    return {
        frozenset(j for j in range(n) if (mask >> j) & 1): c
        for mask, c in enumerate(coeffs)
    }


def parseval_defect_iv(table: TruthTable) -> Interval:
    """Enclosure of ``sum_S hat f(S)^2 - 1`` (contains ``0`` for any Boolean ``f``)."""
    coeffs = fourier_coeffs_iv(table, encoding="pm1")
    acc = Interval.point(0.0)
    for c in coeffs:
        acc = acc + c * c
    return acc - Interval.point(1.0)


# --------------------------------------------------------------------------- #
# Certified linear-bias bounds.
# --------------------------------------------------------------------------- #
def walsh_at_iv(
    table: TruthTable | None = None,
    mask: int = 0,
    *,
    values: Sequence[IntervalLike] | None = None,
) -> Interval:
    """Enclosure of the unnormalised Walsh coefficient ``(WHT s)(mask)`` (spin)."""
    vals, _ = _pm1_values_iv(table, values, "pm1")
    if not 0 <= mask < len(vals):
        raise ValueError(f"mask {mask} out of range for length {len(vals)}")
    return walsh_hadamard_iv(vals)[mask]


def linear_bias_iv(table: TruthTable, mask: int) -> Interval:
    r"""Certified linear-approximation bias ``Pr_x[f(x) = <mask,x>] - 1/2``.

    Equals ``hat f(mask) / 2`` in the ``{+1,-1}`` encoding; ``|bias| <= 1/2``.
    """
    n = num_vars(table)
    w = walsh_at_iv(table, mask)
    scale = Interval.from_rational(Fraction(1, 1 << (n + 1)))
    return w * scale


def linearity_iv(
    table: TruthTable | None = None, *, values: Sequence[IntervalLike] | None = None
) -> Interval:
    """Enclosure of the linearity ``L(f) = max_S |(WHT s)(S)|`` (spin, unnormalised)."""
    vals, _ = _pm1_values_iv(table, values, "pm1")
    wht = walsh_hadamard_iv(vals)
    return _max_iv([_abs_iv(c) for c in wht])


def nonlinearity_iv(
    table: TruthTable | None = None, *, values: Sequence[IntervalLike] | None = None
) -> Interval:
    """Enclosure of the nonlinearity ``NL(f) = 2^{n-1} - L(f)/2``."""
    vals, n = _pm1_values_iv(table, values, "pm1")
    lin = linearity_iv(values=vals)
    half = Interval.point(0.5)
    return Interval.point(float(1 << (n - 1))) - lin * half


def max_linear_bias_iv(table: TruthTable) -> Interval:
    """Enclosure of ``max_{a != 0} |Pr_x[f(x) = <a,x>] - 1/2|`` (best linear approx)."""
    n = num_vars(table)
    wht = walsh_hadamard_iv([Interval.point(float(1 - 2 * v)) for v in table])
    scale = Interval.from_rational(Fraction(1, 1 << (n + 1)))
    return _max_iv([_abs_iv(wht[mask] * scale) for mask in range(1, len(wht))])


# --------------------------------------------------------------------------- #
# Certified differential-bias bounds (autocorrelation / derivative spectrum).
# --------------------------------------------------------------------------- #
def autocorrelation_iv(
    table: TruthTable | None = None,
    shift: int = 0,
    *,
    values: Sequence[IntervalLike] | None = None,
) -> Interval:
    r"""Enclosure of the autocorrelation ``C_f(a) = sum_x s(x) s(x xor a)`` (spin).

    For a Boolean ``f`` this is ``2^n (1 - 2 Pr_x[D_a f(x) = 1])`` where
    ``D_a f = f(x) xor f(x xor a)`` is the (first-order) Boolean derivative.
    """
    vals, _ = _pm1_values_iv(table, values, "pm1")
    size = len(vals)
    if not 0 <= shift < size:
        raise ValueError(f"shift {shift} out of range for length {size}")
    acc = Interval.point(0.0)
    for x in range(size):
        acc = acc + vals[x] * vals[x ^ shift]
    return acc


def differential_bias_iv(table: TruthTable, shift: int) -> Interval:
    r"""Certified differential bias ``Pr_x[D_a f(x) = 0] - 1/2 = C_f(a) / 2^{n+1}``."""
    n = num_vars(table)
    c = autocorrelation_iv(table, shift)
    return c * Interval.from_rational(Fraction(1, 1 << (n + 1)))


def absolute_indicator_iv(table: TruthTable) -> Interval:
    """Enclosure of the absolute indicator ``max_{a != 0} |C_f(a)|``.

    A small absolute indicator certifies good resistance to differential-style
    attacks (the function is far from having any high-bias derivative).
    """
    n = num_vars(table)
    size = 1 << n
    return _max_iv([_abs_iv(autocorrelation_iv(table, a)) for a in range(1, size)])


__all__ = [
    "absolute_indicator_iv",
    "autocorrelation_iv",
    "differential_bias_iv",
    "fourier_coeffs_iv",
    "linear_bias_iv",
    "linearity_iv",
    "max_linear_bias_iv",
    "mobius_iv",
    "nonlinearity_iv",
    "parseval_defect_iv",
    "walsh_at_iv",
    "walsh_hadamard_iv",
    "walsh_spectrum_iv",
]
