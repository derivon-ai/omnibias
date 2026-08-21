# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sparse n-variate polynomials over :class:`~fractions.Fraction`.

A :class:`PolyN` is a finite map from exponent tuples to rational coefficients.
Arithmetic is exact. The Jacobian determinant (``n <= 3``) and the univariate
Sylvester resultant are the algebra used by Keller replay / tangent-sweep and
the ``n=2`` finite lie; there is no Groebner basis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Self

from omnibias.holonomic._core.rational_poly import Poly, pderiv, peval, to_poly

Rational = Fraction | int
Monomial = tuple[int, ...]


def _as_fraction(c: Rational) -> Fraction:
    return c if isinstance(c, Fraction) else Fraction(c)


def _clean(terms: Mapping[Monomial, Fraction]) -> dict[Monomial, Fraction]:
    return {mon: coeff for mon, coeff in terms.items() if coeff != 0}


class PolyN:
    """Sparse polynomial in ``nvars`` indeterminates over ``Q``."""

    __slots__ = ("nvars", "terms")

    def __init__(self, nvars: int, terms: Mapping[Monomial, Rational] | None = None) -> None:
        if nvars < 0:
            raise ValueError(f"nvars must be >= 0, got {nvars}")
        cleaned: dict[Monomial, Fraction] = {}
        if terms:
            for mon, coeff in terms.items():
                if len(mon) != nvars:
                    raise ValueError(f"monomial {mon!r} is not length {nvars}")
                if any(e < 0 for e in mon):
                    raise ValueError(f"monomial {mon!r} has a negative exponent")
                value = _as_fraction(coeff)
                if value != 0:
                    cleaned[mon] = value
        self.nvars = nvars
        self.terms = cleaned

    @classmethod
    def zero(cls, nvars: int) -> Self:
        return cls(nvars)

    @classmethod
    def const(cls, nvars: int, value: Rational) -> Self:
        coeff = _as_fraction(value)
        if coeff == 0:
            return cls(nvars)
        return cls(nvars, {(0,) * nvars: coeff})

    @classmethod
    def var(cls, nvars: int, index: int) -> Self:
        if not 0 <= index < nvars:
            raise ValueError(f"variable index {index} out of range for nvars={nvars}")
        exp = [0] * nvars
        exp[index] = 1
        return cls(nvars, {tuple(exp): Fraction(1)})

    def is_zero(self) -> bool:
        return not self.terms

    def constant_value(self) -> Fraction | None:
        """The constant if ``self`` is a constant polynomial, else ``None``."""
        if not self.terms:
            return Fraction(0)
        if len(self.terms) == 1 and next(iter(self.terms)) == (0,) * self.nvars:
            return next(iter(self.terms.values()))
        return None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolyN):
            return NotImplemented
        return self.nvars == other.nvars and self.terms == other.terms

    def __hash__(self) -> int:
        return hash((self.nvars, frozenset(self.terms.items())))

    def __add__(self, other: PolyN | Rational) -> PolyN:
        if not isinstance(other, PolyN):
            return self + PolyN.const(self.nvars, other)
        if other.nvars != self.nvars:
            raise ValueError("nvars mismatch")
        out = dict(self.terms)
        for mon, coeff in other.terms.items():
            out[mon] = out.get(mon, Fraction(0)) + coeff
        return PolyN(self.nvars, _clean(out))

    def __radd__(self, other: Rational) -> PolyN:
        return self + other

    def __neg__(self) -> PolyN:
        return PolyN(self.nvars, {mon: -coeff for mon, coeff in self.terms.items()})

    def __sub__(self, other: PolyN | Rational) -> PolyN:
        if not isinstance(other, PolyN):
            return self + PolyN.const(self.nvars, -_as_fraction(other))
        return self + (-other)

    def __mul__(self, other: PolyN | Rational) -> PolyN:
        if not isinstance(other, PolyN):
            scale = _as_fraction(other)
            if scale == 0:
                return PolyN.zero(self.nvars)
            return PolyN(self.nvars, {mon: coeff * scale for mon, coeff in self.terms.items()})
        if other.nvars != self.nvars:
            raise ValueError("nvars mismatch")
        out: dict[Monomial, Fraction] = {}
        for mon_a, ca in self.terms.items():
            for mon_b, cb in other.terms.items():
                mon = tuple(i + j for i, j in zip(mon_a, mon_b, strict=True))
                out[mon] = out.get(mon, Fraction(0)) + ca * cb
        return PolyN(self.nvars, _clean(out))

    def __rmul__(self, other: Rational) -> PolyN:
        return self * other

    def __pow__(self, exp: int) -> PolyN:
        if exp < 0:
            raise ValueError("PolyN supports only non-negative powers")
        result = PolyN.const(self.nvars, 1)
        base = self
        while exp:
            if exp & 1:
                result = result * base
            exp >>= 1
            if exp:
                base = base * base
        return result

    def partial(self, index: int) -> PolyN:
        if not 0 <= index < self.nvars:
            raise ValueError(f"variable index {index} out of range for nvars={self.nvars}")
        out: dict[Monomial, Fraction] = {}
        for mon, coeff in self.terms.items():
            power = mon[index]
            if power == 0:
                continue
            new_mon = list(mon)
            new_mon[index] = power - 1
            out[tuple(new_mon)] = out.get(tuple(new_mon), Fraction(0)) + coeff * power
        return PolyN(self.nvars, _clean(out))

    def eval(self, values: Sequence[Rational]) -> Fraction:
        if len(values) != self.nvars:
            raise ValueError(f"expected {self.nvars} values, got {len(values)}")
        coords = [_as_fraction(v) for v in values]
        total = Fraction(0)
        for mon, coeff in self.terms.items():
            term = coeff
            for power, value in zip(mon, coords, strict=True):
                if power:
                    term *= value**power
            total += term
        return total

    def compose(self, subs: Sequence[PolyN]) -> PolyN:
        """Substitute ``subs[i]`` for variable ``i``."""
        if len(subs) != self.nvars:
            raise ValueError(f"expected {self.nvars} substitutions, got {len(subs)}")
        if any(p.nvars != self.nvars for p in subs):
            raise ValueError("substitution polynomials must share nvars")
        acc = PolyN.zero(self.nvars)
        for mon, coeff in self.terms.items():
            term = PolyN.const(self.nvars, coeff)
            for power, sub in zip(mon, subs, strict=True):
                if power:
                    term = term * (sub**power)
            acc = acc + term
        return acc

    def total_degree(self) -> int:
        """Total degree, or ``-1`` for the zero polynomial."""
        if not self.terms:
            return -1
        return max(sum(mon) for mon in self.terms)

    def homogeneous_part(self, degree: int) -> PolyN:
        """Sum of terms of exact total ``degree``."""
        if degree < 0:
            return PolyN.zero(self.nvars)
        return PolyN(
            self.nvars,
            {mon: coeff for mon, coeff in self.terms.items() if sum(mon) == degree},
        )


def jacobian_matrix(components: Sequence[PolyN]) -> list[list[PolyN]]:
    """Jacobian matrix of a map ``Q^n -> Q^n``."""
    if not components:
        return []
    nvars = components[0].nvars
    if len(components) != nvars:
        raise ValueError("jacobian_matrix expects as many components as variables")
    if any(f.nvars != nvars for f in components):
        raise ValueError("component nvars mismatch")
    return [[f.partial(j) for j in range(nvars)] for f in components]


def jacobian_det(components: Sequence[PolyN]) -> PolyN:
    """Determinant of the Jacobian (3×3 expansion, or 1×1 / 2×2)."""
    n = len(components)
    if n == 1:
        return components[0].partial(0)
    matrix = jacobian_matrix(components)
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if n != 3:
        raise ValueError("jacobian_det supports only n <= 3")
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def identical_jacobian_constant(components: Sequence[PolyN]) -> Fraction | None:
    """``det JF`` if it is a constant polynomial (possibly zero), else ``None``.

    This is an identity in ``Q[x]``, not a probe sample. A zero constant is
    returned as ``Fraction(0)`` so callers can reject singular maps.
    """

    return jacobian_det(components).constant_value()


def eval_map(
    components: Sequence[PolyN], point: Sequence[Rational]
) -> tuple[Fraction, ...]:
    """Evaluate every component at ``point``."""

    return tuple(component.eval(point) for component in components)


def sylvester_resultant(p: Poly, q: Poly) -> Fraction:
    """Sylvester resultant of two univariate polynomials over ``Q``."""
    p = to_poly(p)
    q = to_poly(q)
    if not p or not q:
        return Fraction(0)
    deg_p, deg_q = len(p) - 1, len(q) - 1
    size = deg_p + deg_q
    if size == 0:
        return Fraction(1)
    # Rows: deg_q shifts of p, then deg_p shifts of q. Columns high-to-low degree.
    matrix = [[Fraction(0)] * size for _ in range(size)]
    p_high = list(reversed(p))
    q_high = list(reversed(q))
    for shift in range(deg_q):
        for i, coeff in enumerate(p_high):
            matrix[shift][shift + i] = coeff
    for shift in range(deg_p):
        for i, coeff in enumerate(q_high):
            matrix[deg_q + shift][shift + i] = coeff
    return _det_fraction(matrix)


def _det_fraction(matrix: list[list[Fraction]]) -> Fraction:
    """Exact determinant by Gaussian elimination."""
    n = len(matrix)
    a = [row[:] for row in matrix]
    det = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            det = -det
        pivot_val = a[col][col]
        det *= pivot_val
        inv = Fraction(1) / pivot_val
        for r in range(col + 1, n):
            if a[r][col] == 0:
                continue
            factor = a[r][col] * inv
            for c in range(col, n):
                a[r][c] -= factor * a[col][c]
    return det


def integrate_poly(p: Poly) -> Poly:
    """Antiderivative of a univariate polynomial with zero constant of integration."""
    p = to_poly(p)
    if not p:
        return ()
    out = [Fraction(0)] * (len(p) + 1)
    for k, coeff in enumerate(p):
        out[k + 1] = coeff / (k + 1)
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)


def q_from_p(p: Poly) -> Poly:
    r"""Integrate ``q'(w) = (w/2) p'(w)`` with zero constant term."""
    p = to_poly(p)
    half_w_pprime = [Fraction(0)] * (len(p))
    # (w/2) p' = sum_{k>=1} (k p_k / 2) w^k
    for k, coeff in enumerate(p):
        if k == 0:
            continue
        idx = k
        if idx >= len(half_w_pprime):
            half_w_pprime.extend([Fraction(0)] * (idx + 1 - len(half_w_pprime)))
        half_w_pprime[idx] += coeff * k / 2
    return integrate_poly(tuple(half_w_pprime))


def eval_univariate(p: Poly, w: Rational) -> Fraction:
    return peval(to_poly(p), w)


def differentiate_univariate(p: Poly) -> Poly:
    return pderiv(to_poly(p))


def iter_exponents(nvars: int, max_degree: int) -> Iterable[Monomial]:
    """All exponent tuples of total degree ``<= max_degree``."""
    if nvars == 0:
        yield ()
        return

    def rec(prefix: list[int], remaining: int) -> Iterable[Monomial]:
        if len(prefix) == nvars:
            yield tuple(prefix)
            return
        for e in range(remaining + 1):
            prefix.append(e)
            yield from rec(prefix, remaining - e)
            prefix.pop()

    yield from rec([], max_degree)


__all__ = [
    "PolyN",
    "differentiate_univariate",
    "eval_map",
    "eval_univariate",
    "identical_jacobian_constant",
    "integrate_poly",
    "iter_exponents",
    "jacobian_det",
    "jacobian_matrix",
    "q_from_p",
    "sylvester_resultant",
]
