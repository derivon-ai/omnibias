# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified ``d``-dimensional Fourier series in the weighted :math:`\ell^1_\nu` algebra.

This is the multivariate, *complex* generalisation of the one-sided
:class:`~omnibias.core.verified.sequence_space.ValidatedSeries`.  A
:class:`ValidatedFourierSeries` represents a periodic field

.. math::

    a(x) = \sum_{k \in \mathbb{Z}^d} a_k\, e^{i k\cdot x}

by a finite block of complex interval coefficients on the box
``K_N = \{k : \|k\|_\infty \le N\}`` plus a non-negative *tail* radius bounding the
weighted norm of everything truncated away,

.. math::

    \|a\|_\nu = \sum_{k} |a_k|\,\nu^{\|k\|_1},
    \qquad \text{tail} \ge \sum_{\|k\|_\infty > N} |a_k|\,\nu^{\|k\|_1}.

With ``nu >= 1`` the weight is sub-multiplicative
(``nu^{\|i+j\|_1} \le nu^{\|i\|_1}\nu^{\|j\|_1}``), so convolution is a bounded
bilinear operation and :math:`\ell^1_\nu` is a Banach *algebra* -- the property
the Newton-Kantorovich / radii-polynomial closure in
:mod:`omnibias.core.verified.kantorovich` relies on.

Why this unlocks the nonlocal operators
---------------------------------------
In Fourier space the operators that are *non-local and hard* in physical space
become *diagonal multipliers*:

* the Riesz transform ``R_j = \partial_j(-\Delta)^{-1/2}`` has symbol
  ``i k_j/|k|`` with ``|symbol| \le 1`` -- a **bounded** multiplier, so it acts
  coefficient-wise on the kept block and scales the tail by ``1``;
* the Leray (Helmholtz) projection ``P = I - \nabla\Delta^{-1}\mathrm{div}`` has
  symbol ``\delta_{ab} - k_a k_b/|k|^2``, again bounded by ``1``;
* the SQG velocity ``u = \nabla^\perp(-\Delta)^{-1/2}\theta`` is just
  ``(-R_2, R_1)\theta``.

So a band-limited spectral ansatz gets an *exact* nonlinear term (via the
truncated convolution, with the aliased tail folded in rigorously) and an *exact*
nonlocal velocity (via :meth:`riesz` / :meth:`leray`) -- the two pieces a
self-similar SQG/gSQG residual needs.

Unbounded multipliers
---------------------
Differentiation ``\partial_j`` (symbol ``i k_j``) and the fractional Laplacian
``(-\Delta)^s`` (symbol ``|k|^{2s}``) are **unbounded**: they do not map
``\ell^1_\nu`` into itself, so there is no finite tail factor.  They are provided
only on a *finite* series (:meth:`derivative` requires a zero tail) or through
:meth:`apply_multiplier` with a caller-supplied, separately-justified
``tail_factor``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.interval import Interval

#: An integer wavevector ``k`` (one entry per dimension).
Wavevector = tuple[int, ...]

#: A Fourier multiplier symbol ``k -> m(k)`` returning a complex enclosure.
Symbol = Callable[[Wavevector], ComplexInterval]


def _nonneg(iv: Interval) -> Interval:
    """Clamp a magnitude/tail enclosure to ``[0, inf)``.

    Outward rounding can push the lower endpoint of an exactly-zero magnitude one
    ulp below zero (``nextafter(0, -inf)``); since the enclosed quantity is a
    non-negative norm, raising the lower bound to ``0`` stays rigorous.
    """
    return Interval(max(iv.lo, 0.0), max(iv.hi, 0.0))


@dataclass
class ValidatedFourierSeries:
    r"""A rigorous element of the weighted :math:`\ell^1_\nu` Fourier algebra over ``Z^d``."""

    dim: int
    trunc: int
    nu: float
    coeffs: dict[Wavevector, ComplexInterval]
    tail: Interval

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.trunc < 0:
            raise ValueError("trunc N must be >= 0")
        if self.nu < 1.0:
            raise ValueError(
                "nu must be >= 1 for the weighted l1 convolution algebra to be sound"
            )
        if self.tail.lo < 0.0:
            raise ValueError("tail radius must be non-negative")
        for k in self.coeffs:
            if len(k) != self.dim:
                raise ValueError(f"wavevector {k} has wrong length for dim={self.dim}")
            if any(abs(kd) > self.trunc for kd in k):
                raise ValueError(f"wavevector {k} lies outside the box |k|_inf <= {self.trunc}")

    # ----- constructors -------------------------------------------------- #
    @classmethod
    def zero(cls, dim: int, trunc: int, nu: float) -> ValidatedFourierSeries:
        return cls(dim, trunc, nu, {}, Interval.point(0.0))

    @classmethod
    def constant(
        cls, value: ComplexLike, dim: int, trunc: int, nu: float
    ) -> ValidatedFourierSeries:
        """The constant field ``a(x) = value`` (only the zero mode is non-zero)."""
        z = ComplexInterval.from_value(value)
        return cls(dim, trunc, nu, {(0,) * dim: z}, Interval.point(0.0))

    @classmethod
    def from_coeffs(
        cls,
        coeffs: Mapping[Wavevector, ComplexLike],
        dim: int,
        trunc: int,
        nu: float,
        *,
        tail: float = 0.0,
    ) -> ValidatedFourierSeries:
        out: dict[Wavevector, ComplexInterval] = {
            tuple(k): ComplexInterval.from_value(v) for k, v in coeffs.items()
        }
        return cls(dim, trunc, nu, out, Interval.from_value(tail))

    # ----- queries ------------------------------------------------------- #
    def get(self, k: Wavevector) -> ComplexInterval:
        return self.coeffs.get(tuple(k), ComplexInterval.zero())

    def _weight(self, k: Wavevector) -> Interval:
        return Interval.point(self.nu).pow_int(sum(abs(kd) for kd in k))

    def low_norm(self) -> Interval:
        """Weighted norm of the finite (kept) block only."""
        acc = Interval.point(0.0)
        for k, c in self.coeffs.items():
            acc = acc + Interval.point(c.mag) * self._weight(k)
        return acc

    def norm(self) -> Interval:
        r"""Rigorous total weighted norm ``\|a\|_\nu`` (kept block + tail)."""
        return self.low_norm() + self.tail

    def _check(self, other: ValidatedFourierSeries) -> None:
        if self.dim != other.dim or self.trunc != other.trunc or self.nu != other.nu:
            raise ValueError("ValidatedFourierSeries operands must share dim, trunc, nu")

    # ----- arithmetic ---------------------------------------------------- #
    def __add__(self, other: ValidatedFourierSeries) -> ValidatedFourierSeries:
        self._check(other)
        coeffs = {k: v for k, v in self.coeffs.items()}
        for k, v in other.coeffs.items():
            coeffs[k] = coeffs.get(k, ComplexInterval.zero()) + v
        return ValidatedFourierSeries(
            self.dim, self.trunc, self.nu, coeffs, _nonneg(self.tail + other.tail)
        )

    def __neg__(self) -> ValidatedFourierSeries:
        coeffs = {k: -v for k, v in self.coeffs.items()}
        return ValidatedFourierSeries(self.dim, self.trunc, self.nu, coeffs, self.tail)

    def __sub__(self, other: ValidatedFourierSeries) -> ValidatedFourierSeries:
        return self.__add__(-other)

    def scale(self, factor: ComplexLike) -> ValidatedFourierSeries:
        """Multiply by a (complex) scalar rigorously."""
        c = ComplexInterval.from_value(factor)
        coeffs = {k: v * c for k, v in self.coeffs.items()}
        return ValidatedFourierSeries(
            self.dim,
            self.trunc,
            self.nu,
            coeffs,
            _nonneg(self.tail * Interval.point(c.mag)),
        )

    def __mul__(self, other: ValidatedFourierSeries) -> ValidatedFourierSeries:
        r"""Banach-algebra product (truncated convolution) ``(a b)_k = \sum_{i+j=k} a_i b_j``.

        Coefficients with ``\|k\|_\infty \le N`` are kept *exactly* (so genuine
        cancellation in the convolution is captured); products landing outside the
        box are folded into the tail by their weighted magnitude, and the
        kept/tail and tail/tail cross terms are bounded sub-multiplicatively.
        """
        self._check(other)
        n = self.trunc
        kept: dict[Wavevector, ComplexInterval] = {}
        overflow = Interval.point(0.0)
        nu_iv = Interval.point(self.nu)
        for i, ai in self.coeffs.items():
            for j, bj in other.coeffs.items():
                k = tuple(i[d] + j[d] for d in range(self.dim))
                prod = ai * bj
                if all(abs(kd) <= n for kd in k):
                    kept[k] = kept.get(k, ComplexInterval.zero()) + prod
                else:
                    w = nu_iv.pow_int(sum(abs(kd) for kd in k))
                    overflow = overflow + Interval.point(prod.mag) * w
        cross = (
            self.low_norm() * other.tail
            + self.tail * other.low_norm()
            + self.tail * other.tail
        )
        return ValidatedFourierSeries(
            self.dim, n, self.nu, kept, _nonneg(overflow + cross)
        )

    def banach_algebra_bound(self, other: ValidatedFourierSeries) -> Interval:
        """The sub-multiplicative bound ``\\|a\\|_\\nu \\|b\\|_\\nu`` on the product norm."""
        self._check(other)
        return self.norm() * other.norm()

    # ----- Fourier multipliers ------------------------------------------ #
    def apply_multiplier(
        self, symbol: Symbol, tail_factor: float
    ) -> ValidatedFourierSeries:
        r"""Apply a Fourier multiplier ``m`` coefficient-wise to the kept block.

        ``symbol(k)`` must rigorously enclose ``m(k)``; ``tail_factor`` must be a
        rigorous upper bound on ``sup_{\|k\|_\infty > N} |m(k)|`` so the truncated
        tail scales soundly (this is the caller's proof obligation -- the bounded
        helpers :meth:`riesz` and :meth:`leray` discharge it with factor ``1``).
        """
        if tail_factor < 0.0:
            raise ValueError("tail_factor must be non-negative")
        coeffs = {k: symbol(k) * v for k, v in self.coeffs.items()}
        return ValidatedFourierSeries(
            self.dim,
            self.trunc,
            self.nu,
            coeffs,
            _nonneg(self.tail * Interval.point(float(tail_factor))),
        )

    def riesz(self, j: int) -> ValidatedFourierSeries:
        r"""Riesz transform ``R_j = \partial_j(-\Delta)^{-1/2}`` (bounded, symbol ``i k_j/|k|``)."""
        if not 0 <= j < self.dim:
            raise ValueError(f"axis j={j} out of range for dim {self.dim}")
        return self.apply_multiplier(riesz_symbol(self.dim, j), 1.0)

    def leray(self, a: int, b: int) -> ValidatedFourierSeries:
        r"""Leray-projection entry ``P_{ab} = \delta_{ab} - k_a k_b/|k|^2`` (bounded by ``1``)."""
        if not (0 <= a < self.dim and 0 <= b < self.dim):
            raise ValueError("Leray indices out of range")
        return self.apply_multiplier(leray_symbol(self.dim, a, b), 1.0)

    def derivative(self, axis: int) -> ValidatedFourierSeries:
        r"""Exact partial derivative ``\partial_{axis}`` (symbol ``i k_{axis}``).

        This is an *unbounded* multiplier, so it is only sound on a finite series
        (zero tail).  Use it to differentiate a band-limited spectral ansatz
        exactly; for a series with a non-zero tail the derivative is not an
        ``\ell^1_\nu`` element and a :class:`ValueError` is raised.
        """
        if not 0 <= axis < self.dim:
            raise ValueError(f"axis {axis} out of range for dim {self.dim}")
        if not (self.tail.lo == 0.0 and self.tail.hi == 0.0):
            raise ValueError(
                "derivative is an unbounded multiplier; it requires a zero tail "
                "(a finite trigonometric polynomial)"
            )

        def sym(k: Wavevector) -> ComplexInterval:
            return ComplexInterval(Interval.point(0.0), Interval.from_rational(k[axis]))

        coeffs = {k: sym(k) * v for k, v in self.coeffs.items()}
        return ValidatedFourierSeries(
            self.dim, self.trunc, self.nu, coeffs, Interval.point(0.0)
        )

    def __repr__(self) -> str:
        return (
            f"ValidatedFourierSeries(dim={self.dim}, trunc={self.trunc}, "
            f"nu={self.nu!r}, modes={len(self.coeffs)}, tail={self.tail!r})"
        )


# --------------------------------------------------------------------------- #
# Bounded multiplier symbols.
# --------------------------------------------------------------------------- #
def riesz_symbol(dim: int, j: int) -> Symbol:
    r"""Symbol of the Riesz transform ``R_j``: ``k -> i k_j / |k|`` (``0`` at ``k=0``)."""

    def m(k: Wavevector) -> ComplexInterval:
        if all(kd == 0 for kd in k):
            return ComplexInterval.zero()
        modsq = sum(kd * kd for kd in k)
        mod = Interval.from_rational(modsq).sqrt()
        return ComplexInterval(Interval.point(0.0), Interval.from_rational(k[j]) / mod)

    return m


def leray_symbol(dim: int, a: int, b: int) -> Symbol:
    r"""Symbol of the Leray projection entry ``P_{ab} = \delta_{ab} - k_a k_b/|k|^2``.

    At ``k = 0`` the projection is the identity on the mean, so the symbol is
    ``\delta_{ab}`` there.
    """

    def m(k: Wavevector) -> ComplexInterval:
        if all(kd == 0 for kd in k):
            return ComplexInterval.point(1.0 if a == b else 0.0)
        modsq = Interval.from_rational(sum(kd * kd for kd in k))
        delta = Interval.from_rational(1 if a == b else 0)
        val = delta - Interval.from_rational(k[a] * k[b]) / modsq
        return ComplexInterval(val, Interval.point(0.0))

    return m


__all__ = [
    "Symbol",
    "ValidatedFourierSeries",
    "Wavevector",
    "leray_symbol",
    "riesz_symbol",
]
