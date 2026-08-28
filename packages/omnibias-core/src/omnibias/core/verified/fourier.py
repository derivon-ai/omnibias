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

What ``nu`` means: the strip of analyticity around the real torus
-------------------------------------------------------------------
This is the two-sided, periodic analogue of the one-sided
:class:`~omnibias.core.verified.sequence_space.ValidatedSeries`, whose ``nu > 1``
encloses a function analytic on the *disk* ``|z| <= nu``.  Here the analogous
statement is a **strip**, not a disk, and it is worth deriving precisely because
the two cases differ in a way that matters for soundness (see the next section).

Complexify each real coordinate ``x_d -> x_d + i s_d``.  Since
``e^{i k \cdot x} = e^{i k\cdot t}\,e^{-k\cdot s}`` for ``x = t + is``, a finite
weighted norm ``\|a\|_\nu = \sum_k |a_k|\,\nu^{\|k\|_1} < \infty`` with ``nu >= 1``
lets ``h = \ln(nu) \ge 0`` and, for every ``s`` with ``\|s\|_\infty \le h``,

.. math::

    |a_k\,e^{ik\cdot x}| = |a_k|\,e^{-k\cdot s}
        \le |a_k|\,e^{\|k\|_1 \|s\|_\infty}
        \le |a_k|\,e^{\|k\|_1 h} = |a_k|\,\nu^{\|k\|_1},

using ``|k \cdot s| \le \|k\|_1 \|s\|_\infty`` (Hoelder) then ``\|s\|_\infty \le h``.
The series is therefore absolutely and uniformly summable, hence ``a(x)`` extends
to a function holomorphic (in each variable) on the closed poly-strip
``\{x \in \mathbb{C}^d : |\mathrm{Im}(x_d)| \le h,\ d=1..d\}`` around the real torus
``\mathbb{R}^d/2\pi\mathbb{Z}^d``, with ``\sup_{\text{strip}} |a(x)| \le \|a\|_\nu``.
So ``nu`` plays *exactly* the role the one-sided disk radius plays -- it is
``e^{\text{half-width of the analyticity strip}}`` -- with two boundary cases:
``nu = 1`` (``h = 0``) degenerates the strip to the real torus itself, which is
precisely the classical **Wiener algebra** ``A(\mathbb{T}^d)`` of continuous
functions with absolutely summable Fourier coefficients (no extra analyticity, but
still genuinely *pointwise absolutely convergent*, not merely formal); ``nu < 1``
(``h < 0``) has no such meaning at all -- see below.

Why ``nu >= 1`` is not a stylistic default here (unlike the one-sided sibling)
-------------------------------------------------------------------------------
:class:`~omnibias.core.verified.sequence_space.ValidatedSeries` documents *two*
sound regimes, ``nu > 1`` (analytic disk) **and** ``nu in (0, 1]`` (formal
series) -- and both are sound *as an algebra* there, because its indices
``k >= 0`` are non-negative, so ``|i + j| = i + j = |i| + |j|`` is an *identity*:
``nu^{i+j} = nu^i\,nu^j`` holds exactly for every real ``nu > 0``, whatever the
sign of ``ln(nu)``. Submultiplicativity is automatic; only the *meaning* of
``nu`` (radius of convergence vs. bookkeeping weight) depends on which side of
``1`` it falls.

That identity **fails** here because two-sided wavevectors can cancel:
``\|i+j\|_1 \le \|i\|_1 + \|j\|_1`` is a genuine (sometimes strict) triangle
inequality, e.g. ``i=(1,)``, ``j=(-1,)`` gives ``\|i+j\|_1 = 0 < 2 = \|i\|_1 +
\|j\|_1``. Submultiplicativity ``nu^{\|i+j\|_1} \le nu^{\|i\|_1}\,nu^{\|j\|_1}``
then requires ``x \mapsto nu^x`` to be *non-decreasing*, i.e. ``nu >= 1`` --
and this is not merely sufficient but **tight**: for ``nu < 1`` the same pair
gives ``nu^{\|i+j\|_1} = nu^0 = 1`` while ``nu^{\|i\|_1}nu^{\|j\|_1} = nu^2 < 1``,
so the inequality provably *reverses* and the truncated-convolution proof in
:meth:`ValidatedFourierSeries.__mul__` (which relies on submultiplicativity to
fold overflow terms into a sound tail bound) becomes unsound. **Conclusion: the
``nu >= 1`` floor is exactly the sound boundary for the scalar isotropic weight,
not a conservative stand-in for something looser** -- there is no ``nu < 1``
"formal two-sided series" regime analogous to the one-sided case, because the
Banach-algebra property genuinely (not just by convention) depends on it.

A genuine, orthogonal generalisation: anisotropic per-axis weights
--------------------------------------------------------------------
The floor of ``1`` cannot move, but the requirement that a *single* ``nu`` apply
identically to every one of the ``d`` axes can be relaxed soundly. Passing
``nu`` as a length-``d`` sequence ``(nu_1, ..., nu_d)`` (each ``nu_d >= 1``)
switches the weight to the anisotropic product

.. math::

    w(k) = \prod_{d=1}^{D} nu_d^{|k_d|},

which reduces to the scalar weight exactly when every ``nu_d`` is equal.
Submultiplicativity still holds, axis by axis: for each ``d``,
``|i_d+j_d| \le |i_d|+|j_d|`` and ``nu_d >= 1`` give
``nu_d^{|i_d+j_d|} \le nu_d^{|i_d|+|j_d|} = nu_d^{|i_d|}\,nu_d^{|j_d|}``; taking
the product of these ``d`` non-negative inequalities gives ``w(i+j) \le
w(i)\,w(j)``. Geometrically each ``nu_d`` is an independent strip half-width
``h_d = \ln(nu_d)`` in coordinate ``d`` (an anisotropic poly-strip); forcing a
single isotropic ``nu`` when the true decay is direction-dependent is *sound*
but needlessly loose (it silently uses ``min_d nu_d``, inflating every tail
bound). The bounded-multiplier proofs (:meth:`apply_multiplier` and the
:meth:`riesz`/:meth:`hilbert`/:meth:`leray` helpers built on it) need no change
at all: their soundness argument -- ``\|m\cdot a_{\text{tail}}\|_w \le
\text{tail\_factor}\cdot\|a_{\text{tail}}\|_w`` from a coefficient-wise magnitude
bound on ``m`` -- only uses that ``w \ge 0``, never submultiplicativity, so it
holds for *any* per-axis weight unchanged.

This generalisation is implemented as a strict superset: passing a scalar
``nu`` (the only form accepted before) still runs the identical isotropic
formula through the identical code path, bit-for-bit, so every existing call
site and test is unaffected; only passing a length-``d`` sequence opts into the
anisotropic weight.

Why this unlocks the nonlocal operators
---------------------------------------
In Fourier space the operators that are *non-local and hard* in physical space
become *diagonal multipliers*:

* the Riesz transform ``R_j = \partial_j(-\Delta)^{-1/2}`` has symbol
  ``i k_j/|k|`` with ``|symbol| \le 1`` -- a **bounded** multiplier, so it acts
  coefficient-wise on the kept block and scales the tail by ``1``;
* the coordinate Hilbert transform has symbol ``-i sign(k_j)`` (zero on its
  axis-mean modes), also with modulus at most ``1``;
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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.interval import Interval

#: An integer wavevector ``k`` (one entry per dimension).
Wavevector = tuple[int, ...]

#: A Fourier multiplier symbol ``k -> m(k)`` returning a complex enclosure.
Symbol = Callable[[Wavevector], ComplexInterval]

#: A weight ``nu``: either a single isotropic scalar (broadcast to every axis,
#: the original behaviour) or a length-``dim`` sequence ``(nu_1, ..., nu_dim)``
#: of per-axis weights (see the module docstring for the soundness argument).
NuLike = float | Sequence[float]


def _normalize_nu(nu: NuLike, dim: int) -> tuple[float, ...]:
    r"""Broadcast/validate ``nu`` into a per-axis tuple ``(nu_1, ..., nu_dim)``.

    A scalar is broadcast to every axis (the isotropic weight
    ``nu^{\|k\|_1} = \prod_d nu^{|k_d|}``); a length-``dim`` sequence supplies
    one weight per axis (anisotropic). Every axis weight must be ``>= 1`` --
    this is the *tight* sound boundary for the two-sided weighted ``l1``
    convolution algebra (derived in the module docstring), not a conservative
    stand-in for something looser.
    """
    if isinstance(nu, int | float):
        axes: tuple[float, ...] = (float(nu),) * dim
    else:
        axes = tuple(float(v) for v in nu)
        if len(axes) != dim:
            raise ValueError(f"nu has {len(axes)} axis weight(s) but dim={dim}")
    for v in axes:
        if v < 1.0:
            raise ValueError(
                "every axis weight of nu must be >= 1 for the weighted l1 "
                "convolution algebra to be sound (see module docstring)"
            )
    return axes


def _nonneg(iv: Interval) -> Interval:
    """Clamp a magnitude/tail enclosure to ``[0, inf)``.

    Outward rounding can push the lower endpoint of an exactly-zero magnitude one
    ulp below zero (``nextafter(0, -inf)``); since the enclosed quantity is a
    non-negative norm, raising the lower bound to ``0`` stays rigorous.
    """
    return Interval(max(iv.lo, 0.0), max(iv.hi, 0.0))


@dataclass
class ValidatedFourierSeries:
    r"""A rigorous element of the weighted :math:`\ell^1_\nu` Fourier algebra over ``Z^d``.

    ``nu`` accepts either a single isotropic scalar (broadcast to every axis,
    the weight ``nu^{\|k\|_1}``) or a length-``dim`` sequence of per-axis
    weights ``(nu_1, ..., nu_dim)`` (the weight ``\prod_d nu_d^{|k_d|}``); see
    the module docstring for the derivation of what each regime means and why
    every axis weight must be ``>= 1``. Passing a scalar is a strict
    behavioural subset: it runs the exact pre-existing isotropic formula.
    """

    dim: int
    trunc: int
    nu: NuLike
    coeffs: dict[Wavevector, ComplexInterval]
    tail: Interval

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.trunc < 0:
            raise ValueError("trunc N must be >= 0")
        self._nu_axes: tuple[float, ...] = _normalize_nu(self.nu, self.dim)
        # Canonicalise storage (float stays a float; any sequence -- list,
        # tuple, generator-derived -- becomes the exact tuple in `_nu_axes`) so
        # `_check`/`__repr__`/equality compare like-with-like regardless of the
        # container type the caller happened to pass.
        self.nu = float(self.nu) if isinstance(self.nu, int | float) else self._nu_axes
        if self.tail.lo < 0.0:
            raise ValueError("tail radius must be non-negative")
        for k in self.coeffs:
            if len(k) != self.dim:
                raise ValueError(f"wavevector {k} has wrong length for dim={self.dim}")
            if any(abs(kd) > self.trunc for kd in k):
                raise ValueError(f"wavevector {k} lies outside the box |k|_inf <= {self.trunc}")

    # ----- constructors -------------------------------------------------- #
    @classmethod
    def zero(cls, dim: int, trunc: int, nu: NuLike) -> ValidatedFourierSeries:
        return cls(dim, trunc, nu, {}, Interval.point(0.0))

    @classmethod
    def constant(
        cls, value: ComplexLike, dim: int, trunc: int, nu: NuLike
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
        nu: NuLike,
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

    @property
    def nu_axes(self) -> tuple[float, ...]:
        """The per-axis weight tuple (a scalar ``nu`` broadcasts to every axis)."""
        return self._nu_axes

    def _weight(self, k: Wavevector) -> Interval:
        if isinstance(self.nu, int | float):
            # Isotropic path: identical to the original (pre-generalisation)
            # formula, bit-for-bit -- a single pow_int of the summed exponent.
            return Interval.point(float(self.nu)).pow_int(sum(abs(kd) for kd in k))
        w = Interval.point(1.0)
        for kd, nud in zip(k, self._nu_axes, strict=True):
            w = w * Interval.point(nud).pow_int(abs(kd))
        return w

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
        for i, ai in self.coeffs.items():
            for j, bj in other.coeffs.items():
                k = tuple(i[d] + j[d] for d in range(self.dim))
                prod = ai * bj
                if all(abs(kd) <= n for kd in k):
                    kept[k] = kept.get(k, ComplexInterval.zero()) + prod
                else:
                    overflow = overflow + Interval.point(prod.mag) * self._weight(k)
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
        helpers :meth:`hilbert`, :meth:`riesz`, and :meth:`leray` discharge it
        with factor ``1``).
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

    def hilbert(self, axis: int = 0) -> ValidatedFourierSeries:
        r"""Coordinate Hilbert transform (bounded, symbol ``-i sign(k_axis)``).

        Unlike differentiation, this multiplier has unit modulus away from its
        zero modes, so a non-zero Fourier tail remains in the same weighted
        sequence space.
        """
        if not 0 <= axis < self.dim:
            raise ValueError(f"axis={axis} out of range for dim {self.dim}")
        return self.apply_multiplier(hilbert_symbol(self.dim, axis), 1.0)

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


def hilbert_symbol(dim: int = 1, axis: int = 0) -> Symbol:
    r"""Symbol of a coordinate Hilbert transform: ``k -> -i sign(k_axis)``.

    The one-dimensional Hilbert transform is the default.  In higher
    dimensions, ``axis`` selects the coordinate Hilbert transform.  The
    multiplier is exactly zero for modes whose selected coordinate vanishes
    and otherwise has modulus one, so ``tail_factor=1`` is rigorous.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if not 0 <= axis < dim:
        raise ValueError(f"axis={axis} out of range for dim {dim}")

    def m(k: Wavevector) -> ComplexInterval:
        if len(k) != dim:
            raise ValueError(f"wavevector {k} has wrong length for dim={dim}")
        coordinate = k[axis]
        if coordinate == 0:
            return ComplexInterval.zero()
        sign = 1 if coordinate > 0 else -1
        return ComplexInterval(Interval.point(0.0), Interval.from_rational(-sign))

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
    "NuLike",
    "Symbol",
    "ValidatedFourierSeries",
    "Wavevector",
    "hilbert_symbol",
    "leray_symbol",
    "riesz_symbol",
]
