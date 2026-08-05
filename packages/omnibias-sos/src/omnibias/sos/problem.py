# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Multivariate polynomials and the SOS certificate container.

A :class:`Polynomial` is a sparse multi-index dictionary
``{(e_1, ..., e_n): coefficient}`` over ``n`` real variables, with just enough
algebra (``+``, ``-``, ``*``, scalar scaling, evaluation, partial derivatives) to
build the residual polynomials the Sum-of-Squares and auxiliary-functional
machinery needs -- e.g. ``C - Phi - grad(V) . f`` for the background method.

:class:`SOSCertificate` is the pure result container produced by
:func:`omnibias.sos.certify.certify_sos`; it is sealed into a tamper-evident v1
certificate by :mod:`omnibias.sos.honesty`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

Exponent = tuple[int, ...]
"""A monomial exponent multi-index; ``len == n_vars``."""


@dataclass(frozen=True)
class Polynomial:
    """A sparse multivariate polynomial with real (float) coefficients.

    Parameters
    ----------
    n_vars:
        Number of variables.
    coeffs:
        Mapping from exponent multi-index (length ``n_vars``) to coefficient.
        Zero coefficients and duplicate keys are normalised away on construction.
    """

    n_vars: int
    coeffs: Mapping[Exponent, float]

    def __post_init__(self) -> None:
        cleaned: dict[Exponent, float] = {}
        for exp, coeff in self.coeffs.items():
            key = tuple(int(e) for e in exp)
            if len(key) != self.n_vars:
                raise ValueError(
                    f"exponent {key!r} has length {len(key)}, expected {self.n_vars}"
                )
            if any(e < 0 for e in key):
                raise ValueError(f"negative exponent in {key!r}")
            value = float(coeff)
            if value != 0.0:
                cleaned[key] = cleaned.get(key, 0.0) + value
        # Drop terms that cancelled to exactly zero.
        object.__setattr__(self, "coeffs", {k: v for k, v in cleaned.items() if v != 0.0})

    # -- constructors ------------------------------------------------------- #

    @classmethod
    def zero(cls, n_vars: int) -> Polynomial:
        return cls(n_vars, {})

    @classmethod
    def constant(cls, value: float, n_vars: int) -> Polynomial:
        return cls(n_vars, {(0,) * n_vars: value})

    @classmethod
    def variable(cls, index: int, n_vars: int) -> Polynomial:
        r"""The degree-1 monomial ``x_index``."""
        exp = [0] * n_vars
        exp[index] = 1
        return cls(n_vars, {tuple(exp): 1.0})

    @classmethod
    def monomial(cls, exponent: Exponent, coefficient: float = 1.0) -> Polynomial:
        return cls(len(exponent), {tuple(exponent): coefficient})

    # -- structure ---------------------------------------------------------- #

    @property
    def support(self) -> frozenset[Exponent]:
        return frozenset(self.coeffs)

    def degree(self) -> int:
        """Total degree (``-1`` for the zero polynomial)."""
        return max((sum(exp) for exp in self.coeffs), default=-1)

    def coefficient(self, exponent: Exponent) -> float:
        return self.coeffs.get(tuple(exponent), 0.0)

    # -- evaluation & calculus ---------------------------------------------- #

    def evaluate(self, point: Sequence[float]) -> float:
        if len(point) != self.n_vars:
            raise ValueError(f"point has length {len(point)}, expected {self.n_vars}")
        total = 0.0
        for exp, coeff in self.coeffs.items():
            term = coeff
            for value, power in zip(point, exp, strict=True):
                if power:
                    term *= float(value) ** power
            total += term
        return total

    def partial(self, index: int) -> Polynomial:
        r"""Partial derivative ``d/dx_index``."""
        out: dict[Exponent, float] = {}
        for exp, coeff in self.coeffs.items():
            power = exp[index]
            if power == 0:
                continue
            lowered = list(exp)
            lowered[index] = power - 1
            key = tuple(lowered)
            out[key] = out.get(key, 0.0) + coeff * power
        return Polynomial(self.n_vars, out)

    def gradient(self) -> tuple[Polynomial, ...]:
        return tuple(self.partial(i) for i in range(self.n_vars))

    # -- algebra ------------------------------------------------------------ #

    def __add__(self, other: Polynomial | float) -> Polynomial:
        if isinstance(other, Polynomial):
            self._check_same_arity(other)
            out = dict(self.coeffs)
            for exp, coeff in other.coeffs.items():
                out[exp] = out.get(exp, 0.0) + coeff
            return Polynomial(self.n_vars, out)
        return self + Polynomial.constant(float(other), self.n_vars)

    __radd__ = __add__

    def __neg__(self) -> Polynomial:
        return Polynomial(self.n_vars, {exp: -coeff for exp, coeff in self.coeffs.items()})

    def __sub__(self, other: Polynomial | float) -> Polynomial:
        return self + (-other if isinstance(other, Polynomial) else -float(other))

    def __rsub__(self, other: float) -> Polynomial:
        return Polynomial.constant(float(other), self.n_vars) - self

    def __mul__(self, other: Polynomial | float) -> Polynomial:
        if isinstance(other, Polynomial):
            self._check_same_arity(other)
            out: dict[Exponent, float] = {}
            for exp_a, coeff_a in self.coeffs.items():
                for exp_b, coeff_b in other.coeffs.items():
                    key = tuple(a + b for a, b in zip(exp_a, exp_b, strict=True))
                    out[key] = out.get(key, 0.0) + coeff_a * coeff_b
            return Polynomial(self.n_vars, out)
        return Polynomial(
            self.n_vars, {exp: coeff * float(other) for exp, coeff in self.coeffs.items()}
        )

    __rmul__ = __mul__

    def _check_same_arity(self, other: Polynomial) -> None:
        if self.n_vars != other.n_vars:
            raise ValueError(
                f"arity mismatch: {self.n_vars} vs {other.n_vars} variables"
            )

    def __repr__(self) -> str:
        if not self.coeffs:
            return f"Polynomial(n_vars={self.n_vars}, 0)"
        terms = ", ".join(
            f"{exp}:{coeff:g}" for exp, coeff in sorted(self.coeffs.items())
        )
        return f"Polynomial(n_vars={self.n_vars}, {{{terms}}})"


@dataclass(frozen=True)
class RationalPolynomial:
    r"""A multivariate polynomial with **exact** :class:`~fractions.Fraction` coefficients.

    The auxiliary-functional residual ``C - Phi - grad(V) . f`` must be formed
    exactly: float polynomial arithmetic rounds (products of dyadic rationals can
    overflow the 53-bit mantissa), which would make the certified inequality about
    a slightly different polynomial than intended.  This type keeps every
    coefficient exact so the sealed bound is sound to the last bit.  It exposes the
    same ``coefficient`` / ``support`` surface as :class:`Polynomial`, so the
    rational rounding and certifier accept it directly.
    """

    n_vars: int
    coeffs: Mapping[Exponent, Fraction]

    def __post_init__(self) -> None:
        cleaned: dict[Exponent, Fraction] = {}
        for exp, coeff in self.coeffs.items():
            key = tuple(int(e) for e in exp)
            if len(key) != self.n_vars:
                raise ValueError(f"exponent {key!r} has length {len(key)}, expected {self.n_vars}")
            value = cleaned.get(key, Fraction(0)) + Fraction(coeff)
            cleaned[key] = value
        object.__setattr__(self, "coeffs", {k: v for k, v in cleaned.items() if v != 0})

    @classmethod
    def zero(cls, n_vars: int) -> RationalPolynomial:
        return cls(n_vars, {})

    @classmethod
    def constant(cls, value: Fraction | int, n_vars: int) -> RationalPolynomial:
        return cls(n_vars, {(0,) * n_vars: Fraction(value)})

    @classmethod
    def from_polynomial(cls, polynomial: Polynomial) -> RationalPolynomial:
        """Exact lift of a float :class:`Polynomial` (each float is its dyadic rational)."""
        return cls(polynomial.n_vars, {exp: Fraction(c) for exp, c in polynomial.coeffs.items()})

    @property
    def support(self) -> frozenset[Exponent]:
        return frozenset(self.coeffs)

    def degree(self) -> int:
        return max((sum(exp) for exp in self.coeffs), default=-1)

    def coefficient(self, exponent: Exponent) -> Fraction:
        return self.coeffs.get(tuple(exponent), Fraction(0))

    def to_float(self) -> Polynomial:
        return Polynomial(self.n_vars, {exp: float(c) for exp, c in self.coeffs.items()})

    def partial(self, index: int) -> RationalPolynomial:
        out: dict[Exponent, Fraction] = {}
        for exp, coeff in self.coeffs.items():
            power = exp[index]
            if power == 0:
                continue
            lowered = list(exp)
            lowered[index] = power - 1
            key = tuple(lowered)
            out[key] = out.get(key, Fraction(0)) + coeff * power
        return RationalPolynomial(self.n_vars, out)

    def __add__(self, other: RationalPolynomial) -> RationalPolynomial:
        if self.n_vars != other.n_vars:
            raise ValueError(f"arity mismatch: {self.n_vars} vs {other.n_vars} variables")
        out = dict(self.coeffs)
        for exp, coeff in other.coeffs.items():
            out[exp] = out.get(exp, Fraction(0)) + coeff
        return RationalPolynomial(self.n_vars, out)

    def __neg__(self) -> RationalPolynomial:
        return RationalPolynomial(self.n_vars, {exp: -c for exp, c in self.coeffs.items()})

    def __sub__(self, other: RationalPolynomial) -> RationalPolynomial:
        return self + (-other)

    def __mul__(self, other: RationalPolynomial | Fraction | int) -> RationalPolynomial:
        if isinstance(other, RationalPolynomial):
            if self.n_vars != other.n_vars:
                raise ValueError(f"arity mismatch: {self.n_vars} vs {other.n_vars} variables")
            out: dict[Exponent, Fraction] = {}
            for exp_a, coeff_a in self.coeffs.items():
                for exp_b, coeff_b in other.coeffs.items():
                    key = tuple(a + b for a, b in zip(exp_a, exp_b, strict=True))
                    out[key] = out.get(key, Fraction(0)) + coeff_a * coeff_b
            return RationalPolynomial(self.n_vars, out)
        scalar = Fraction(other)
        return RationalPolynomial(self.n_vars, {exp: c * scalar for exp, c in self.coeffs.items()})

    __rmul__ = __mul__


@dataclass(frozen=True)
class SOSCertificate:
    r"""Result of an SOS certification attempt.

    A ``status == "proved"`` certificate carries a rational Gram matrix whose
    outward-rounded interval ``LDL^T`` pivots are all strictly positive, so
    ``p(x) = z(x)^T Q z(x) >= 0`` for **all** ``x``.  A ``status ==
    "inconclusive"`` certificate makes **no** positivity claim -- the SDP was
    infeasible, the rational rounding failed, or the positive-definite margin was
    not met.

    Attributes
    ----------
    status:
        ``"proved"`` or ``"inconclusive"``.
    n_vars:
        Number of variables of the certified polynomial.
    basis:
        The monomial-basis exponents ``z(x)`` used for the Gram matrix.
    gram:
        The rational Gram matrix ``Q`` (as strings, exact) that matches the
        polynomial coefficients exactly, or ``None`` when inconclusive.
    pivots:
        The interval ``LDL^T`` pivots ``(lo, hi)`` of ``Q``; every ``lo > 0`` on a
        proved certificate.
    pd_margin:
        ``min_j pivot_j.lo`` -- the certified positive-definiteness margin
        (``> 0`` iff proved).
    coeff_residual:
        Max absolute polynomial-coefficient mismatch after exact rational
        rounding (``0`` on a proved certificate; the rounding is exact).
    detail:
        Human-readable explanation, especially of an inconclusive verdict.
    """

    status: str
    n_vars: int
    basis: tuple[Exponent, ...]
    gram: tuple[tuple[str, ...], ...] | None
    pivots: tuple[tuple[float, float], ...]
    pd_margin: float
    coeff_residual: float
    detail: str

    @property
    def certified(self) -> bool:
        """Whether this is a sound proof of global nonnegativity."""
        return self.status == "proved"


__all__ = ["Exponent", "Polynomial", "RationalPolynomial", "SOSCertificate"]
