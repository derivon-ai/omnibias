# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified physicists' Hermite-function basis with an exact Fourier-transform tail.

Convention
----------
This module fixes the **physicists'** Hermite polynomials ``H_n``,

.. math::

    H_0(x) = 1,\quad H_1(x) = 2x,\quad H_{n+1}(x) = 2x\,H_n(x) - 2n\,H_{n-1}(x),

which have exact **integer** coefficients (:func:`hermite_poly_coeffs_exact`,
generated with the same iterative-recurrence idiom as
:func:`omnibias.core.polynomials.hermite_coeffs` /
:func:`omnibias.core.verified.coeffs.hermite_coeffs_exact` for the
*probabilists'* convention used by the Gaussian derivative tower -- this module
does **not** reuse those, because the physicists' and probabilists' Hermite
polynomials are different polynomials (``H_n(x) = 2^{n/2} He_n(x\sqrt2)``), and
only the physicists' family is the classical Fourier eigenbasis below). The
(unnormalized) **Hermite function** is

.. math::

    \psi_n(x) = H_n(x)\, e^{-x^2/2}

(:func:`hermite_function`), evaluated rigorously as an interval Horner
evaluation of the exact-integer polynomial (:func:`omnibias.core.verified.coeffs.horner_interval`)
times a rigorous enclosure of ``exp(-x^2/2)`` (:func:`gaussian_weight`, built on
:func:`omnibias.core.verified.transcend.exp_iv`).

Exact self-duality under the Fourier transform
-----------------------------------------------
Fix the **unitary** Fourier transform convention

.. math::

    \mathcal{F}[f](k) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} f(x)\,e^{-ikx}\,dx.

With *this* normalisation and *this* Gaussian weight (``e^{-x^2/2}``, not
``e^{-x^2}``), the Hermite functions are the classical Fourier eigenbasis:

.. math::

    \mathcal{F}[\psi_n] = (-i)^n\,\psi_n \qquad (n = 0, 1, 2, \dots).

The base case is a direct, elementary check that fixes the convention (rather
than merely asserting the general theorem): the Gaussian integral
``int e^{-x^2/2} e^{-ikx} dx = sqrt(2 pi) e^{-k^2/2}`` gives
``F[psi_0](k) = e^{-k^2/2} = psi_0(k) = (-i)^0 psi_0(k)`` -- ``psi_0`` is
self-dual under exactly this transform, which pins down both the ``1/sqrt(2
pi)`` prefactor and the ``e^{-x^2/2}`` (not ``e^{-x^2}``) weight as the unique
combination making the ``n=0`` case work; the general ``n`` case is the
standard fact that ``psi_n`` is built from ``psi_0`` by the raising operator
``a^dagger = (x - d/dx)/sqrt2``, which anticommutes appropriately with
``F`` (see e.g. Stein & Weiss, *Introduction to Fourier Analysis on Euclidean
Spaces*, or any quantum-mechanical treatment of the harmonic oscillator).

Because the multiplier ``(-i)^n`` is *diagonal* in this basis -- exactly the
same shape as the bounded Fourier multipliers ``riesz``/``hilbert``/``leray``
in :mod:`omnibias.core.verified.fourier`, which also transform a truncated
expansion by acting coefficient-wise -- :meth:`HermiteExpansion.fourier_transform_exact`
needs no numerical approximation at all: ``(-i)^n`` cycles through the four
*exactly representable* values ``{1, -i, -1, i}``, so the transform is applied
by **exact component permutation/sign-flip** (:func:`_apply_neg_i_power`), not
generic (outward-rounded) complex-interval multiplication -- the result is
bit-for-bit exact, not merely a sound enclosure. And because ``|(-i)^n| = 1``
this diagonal multiplier is an *isometry*, exactly like the Fourier-series
Riesz/Hilbert/Leray multipliers being bounded by ``1``: it does not change
which coefficient count ``N`` is kept, so any tail bound expressed in terms of
``N`` and the coefficient-decay hypothesis (below) applies unchanged to the
transformed expansion.

Why a future Cohn-Elkies sphere-packing bound wants exactly this
------------------------------------------------------------------
The Cohn-Elkies linear-programming bound for sphere packing needs a radial test
function ``f`` with ``f(0) > 0``, ``\hat f(0) > 0``, ``f(x) <= 0`` for
``|x| >= r``, and ``\hat f >= 0`` everywhere -- i.e. *simultaneous* sign
constraints on ``f`` **and** its Fourier transform. Candidate functions are
built as finite combinations of Hermite functions precisely because that makes
``\hat f`` computable **exactly** (no numerical FFT, no truncation error in the
transform itself) as the *same* finite coefficient vector up to the diagonal
``(-i)^n`` phases: the sign/positivity LP constraints on ``f`` and ``\hat f``
become constraints on one finite-dimensional coefficient vector rather than two
independently-truncated series. This module supplies exactly that primitive
(an exact, rigorous evaluator plus an exact, rigorous transform).  The
fixed-dimension Cohn-Elkies LP that consumes this primitive lives in
:mod:`omnibias.core.verified.cohn_elkies`.

Tail-bound contract (read before trusting a truncated evaluation)
---------------------------------------------------------------------
:meth:`HermiteExpansion.tail_bound` reuses the geometric-decay contract of
:func:`omnibias.core.verified.sequence_space.geometric_tail_bound` --
``|c_n| <= coeff_bound * ratio^n`` for ``n`` past the truncation, exactly as
that function already assumes -- multiplied by a *caller-supplied* uniform
bound ``psi_bound`` on ``|psi_n(x)|`` (or ``|\hat\psi_n(x)|``, see below) valid
for every kept-past-truncation ``n``:

.. math::

    \Bigl|\sum_{n>N} c_n\,\psi_n(x)\Bigr|
        \le \sum_{n>N} |c_n|\,|\psi_n(x)|
        \le \text{psi\_bound}\sum_{n>N} \text{coeff\_bound}\cdot\text{ratio}^n,

the last sum being exactly :func:`geometric_tail_bound` with weight ``nu=1``.
This composition is unconditionally sound *given* the two hypotheses; neither
is verified internally (exactly like :func:`geometric_tail_bound`'s own
``ratio``, or :class:`omnibias.core.verified.radii_spectral.SpectralProblem`'s
``tail_inverse_bound`` / ``quadratic_norm`` -- caller-supplied, caller-justified
hypotheses are the norm for this register). ``tail_bound`` only checks
``psi_bound >= 0`` and raises rather than silently under-certifying if it is
negative; it cannot check that ``psi_bound`` genuinely bounds ``|psi_n(x)|`` for
every ``n > N``, because that is a claim about unboundedly many ``n``.

This matters concretely because the two bases behave very differently as
``n -> infinity`` at a *fixed* ``x``:

* **Unnormalized** ``psi_n``: ``H_n`` has leading coefficient growing
  combinatorially, so ``|psi_n(x)|`` grows *unboundedly* in ``n`` for any fixed
  ``x`` (roughly like ``sqrt(n!)``, by the classical Cramer bound below). No
  finite constant ``psi_bound`` is a valid uniform bound "for all ``n > N``" on
  this basis -- a constant-``psi_bound`` hypothesis can only be honestly
  discharged over a *known finite* range of ``n`` (e.g. checked directly against
  :func:`hermite_function` up to some concrete cutoff), never unboundedly.
* **``L^2``-normalized** ``\hat\psi_n(x) = \psi_n(x) / sqrt(2^n n! sqrt(pi))``
  (:func:`hermite_function_normalized`): the classical **Cramer inequality**
  (Cramer, 1939; see e.g. Thangavelu, *Lectures on Hermite and Laguerre
  Expansions*, Lemma 1.5.1) gives the genuinely uniform bound
  ``sup_{x in R} |\hat\psi_n(x)| <= pi^{-1/4}`` for **every** ``n`` -- a fixed
  constant that legitimately discharges the "for all ``n > N``" hypothesis. This
  module does not re-derive or internally rely on Cramer's inequality (it is
  cited, not proved, here); :data:`CRAMER_UNIFORM_BOUND` is provided purely as a
  rigorously-outward-rounded value of ``pi^{-1/4}`` (computed from
  :data:`omnibias.core.verified.transcend.PI_IV`) for a caller who chooses to
  cite Cramer's inequality themselves as their ``psi_bound`` justification.

Use the normalized basis (``normalized=True``) with :data:`CRAMER_UNIFORM_BOUND`
when an unboundedly-sound tail claim matters; use the unnormalized basis when
only a finite-window comparison against a higher-order truncation is needed
(exactly the containment tests below).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from omnibias.core.verified.coeffs import horner_interval
from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.sequence_space import geometric_tail_bound
from omnibias.core.verified.transcend import PI_IV, exp_iv

#: Bound on each memo. The recurrence/normalisation are keyed on a caller-
#: supplied order, and an unbounded memo would let one hostile or mistyped call
#: pin arbitrarily much memory for the life of the process.
_CACHE_SIZE: int = 256

#: Rigorously-outward-rounded upper bound on ``pi**-0.25``, the classical
#: Cramer uniform sup-bound for the L2-normalized physicists' Hermite
#: functions (see the module docstring). Convenience value only -- never
#: enforced internally; a caller who wants the unboundedly-sound tail-bound
#: contract passes this explicitly as ``psi_bound`` together with
#: ``normalized=True``.
CRAMER_UNIFORM_BOUND: float = PI_IV.sqrt().sqrt().reciprocal().hi


@lru_cache(maxsize=_CACHE_SIZE)
def hermite_poly_coeffs_exact(n: int) -> tuple[int, ...]:
    """Exact integer coefficients of the physicists' Hermite polynomial ``H_n``.

    Returns ``(c_0, ..., c_n)`` with ``H_n(x) = sum_k c_k * x**k``. ``H_0 = 1``,
    ``H_1 = 2x``; recurrence ``H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)``. Iterative
    (not self-recursive), so a tall tower costs memory but never a
    ``RecursionError``.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    if n == 0:
        return (1,)
    if n == 1:
        return (0, 2)
    prev2: tuple[int, ...] = (1,)
    prev1: tuple[int, ...] = (0, 2)
    for m in range(2, n + 1):
        out = [0] * (m + 1)
        for k, c in enumerate(prev1):
            out[k + 1] += 2 * c
        for k, c in enumerate(prev2):
            out[k] -= 2 * (m - 1) * c
        prev2, prev1 = prev1, tuple(out)
    return prev1


def gaussian_weight(x: IntervalLike) -> Interval:
    r"""Rigorous enclosure of the Gaussian weight ``exp(-x^2/2)``."""
    xi = Interval.from_value(x)
    return exp_iv(Interval.point(-0.5) * xi.pow_int(2))


def hermite_function(n: int, x: IntervalLike) -> Interval:
    r"""Rigorous enclosure of the (unnormalized) Hermite function ``psi_n(x)``.

    ``psi_n(x) = H_n(x) * exp(-x^2/2)``, evaluated exactly via
    :func:`omnibias.core.verified.coeffs.horner_interval` on the exact integer
    coefficients of :func:`hermite_poly_coeffs_exact`, times :func:`gaussian_weight`.
    """
    xi = Interval.from_value(x)
    return horner_interval(hermite_poly_coeffs_exact(n), xi) * gaussian_weight(xi)


@lru_cache(maxsize=_CACHE_SIZE)
def _normalization_factor(n: int) -> Interval:
    r"""Rigorous enclosure of ``sqrt(2^n * n! * sqrt(pi))``, the ``L^2(R)`` norm of ``psi_n``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    two_n_n_factorial = Interval.from_rational(2**n * math.factorial(n))
    return (two_n_n_factorial * PI_IV.sqrt()).sqrt()


def hermite_function_normalized(n: int, x: IntervalLike) -> Interval:
    r"""Rigorous enclosure of the ``L^2``-normalized Hermite function ``\hat\psi_n(x)``.

    ``\hat\psi_n(x) = \psi_n(x) / sqrt(2^n n! sqrt(pi))``. Unlike the
    unnormalized :func:`hermite_function`, ``sup_x |\hat\psi_n(x)|`` is
    uniformly bounded in ``n`` (Cramer's inequality; see the module docstring
    and :data:`CRAMER_UNIFORM_BOUND`), which is what makes a constant
    ``psi_bound`` hypothesis in :meth:`HermiteExpansion.tail_bound` genuinely
    discharge-able for unboundedly many ``n``.
    """
    return hermite_function(n, x) / _normalization_factor(n)


#: The four exactly-representable values of ``(-i)^n`` (period 4), each built
#: with :meth:`~omnibias.core.verified.interval.Interval.point` (no rounding).
_NEG_I_POWERS: tuple[ComplexInterval, ComplexInterval, ComplexInterval, ComplexInterval] = (
    ComplexInterval(Interval.point(1.0), Interval.point(0.0)),
    ComplexInterval(Interval.point(0.0), Interval.point(-1.0)),
    ComplexInterval(Interval.point(-1.0), Interval.point(0.0)),
    ComplexInterval(Interval.point(0.0), Interval.point(1.0)),
)


def neg_i_power(n: int) -> ComplexInterval:
    r"""Exact enclosure of the Fourier-eigenvalue symbol ``(-i)^n`` (period 4).

    ``n=0,1,2,3 -> 1, -i, -1, i``, then repeats. Every value is an exact
    (degenerate, zero-width) :class:`~omnibias.core.verified.complex_interval.ComplexInterval`.
    :meth:`HermiteExpansion.fourier_transform_exact` applies this multiplier via
    exact component permutation (:func:`_apply_neg_i_power`), not by generic
    complex-interval multiplication with the value returned here, so that the
    transform stays bit-for-bit exact rather than merely soundly rounded.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return _NEG_I_POWERS[n % 4]


def _apply_neg_i_power(c: ComplexInterval, n: int) -> ComplexInterval:
    r"""Exact ``(-i)^n * c`` by component permutation/sign-flip -- zero rounding.

    Multiplying by ``1``, ``-i``, ``-1``, or ``i`` only permutes and/or negates
    the real/imaginary parts; :meth:`~omnibias.core.verified.interval.Interval.__neg__`
    is an exact sign flip (no rounding for any finite float), so this introduces
    *no* interval-width inflation at all, unlike generic
    :meth:`~omnibias.core.verified.complex_interval.ComplexInterval.__mul__`
    (which outward-rounds every product/sum even when the true result happens
    to be exact).
    """
    r = n % 4
    if r == 0:
        return c
    if r == 1:  # (-i) * (re + im*i) = im - re*i
        return ComplexInterval(c.im, -c.re)
    if r == 2:  # (-1) * (re + im*i) = -re - im*i
        return ComplexInterval(-c.re, -c.im)
    # r == 3: (i) * (re + im*i) = -im + re*i
    return ComplexInterval(-c.im, c.re)


@dataclass
class HermiteExpansion:
    r"""A finite physicists' Hermite-function expansion ``f = sum_{n=0}^{N} c_n psi_n``.

    ``coeffs`` are the (generally complex) :class:`~omnibias.core.verified.complex_interval.ComplexInterval`
    coefficients ``c_0, ..., c_N``; a real-valued target function has every
    ``c_n`` with an exactly-zero imaginary part. Complex coefficients (rather
    than real :class:`~omnibias.core.verified.interval.Interval`) are used so
    that :meth:`fourier_transform_exact`'s ``(-i)^n`` phases live in the same
    coefficient type without a second class -- exactly the design already used
    by :class:`omnibias.core.verified.fourier.ValidatedFourierSeries`.

    This class carries **no stored tail radius**: unlike
    :class:`~omnibias.core.verified.sequence_space.ValidatedSeries`
    (whose weighted-norm tail bounds ``sum_k |a_k| nu^k`` uniformly, independent
    of the evaluation point), a Hermite tail bound is intrinsically a function
    of the evaluation point ``x`` through ``|psi_n(x)|`` -- see
    :meth:`tail_bound`, computed on demand from an explicit, caller-supplied
    hypothesis rather than stored.
    """

    coeffs: list[ComplexInterval]

    def __post_init__(self) -> None:
        if not self.coeffs:
            raise ValueError("HermiteExpansion needs at least one coefficient (order >= 0)")

    @property
    def order(self) -> int:
        """The truncation order ``N`` (``len(coeffs) - 1``)."""
        return len(self.coeffs) - 1

    @classmethod
    def from_coeffs(cls, coeffs: Sequence[ComplexLike]) -> HermiteExpansion:
        return cls([ComplexInterval.from_value(c) for c in coeffs])

    def get(self, n: int) -> ComplexInterval:
        """The ``n``-th coefficient ``c_n`` (``0 <= n <= order``)."""
        return self.coeffs[n]

    def evaluate_kept(self, x: IntervalLike, *, normalized: bool = False) -> ComplexInterval:
        r"""Exact enclosure of the retained sum ``sum_{n=0}^{N} c_n psi_n(x)`` (no tail).

        Uses :func:`hermite_function_normalized` instead of :func:`hermite_function`
        when ``normalized=True``; the caller must then also use the normalized
        basis consistently in :meth:`tail_bound` / :meth:`evaluate`.
        """
        xi = Interval.from_value(x)
        basis = hermite_function_normalized if normalized else hermite_function
        acc = ComplexInterval.zero()
        for n, c in enumerate(self.coeffs):
            acc = acc + c * basis(n, xi)
        return acc

    def tail_bound(self, *, coeff_bound: float, ratio: float, psi_bound: float) -> Interval:
        r"""Rigorous bound on ``|sum_{n>N} c_n psi_n(x)|`` under an explicit decay hypothesis.

        Requires (both caller-supplied, caller-justified, **not** verified here
        -- see the module docstring's tail-bound contract):

        * a geometric coefficient-decay hypothesis ``|c_n| <= coeff_bound *
          ratio**n`` for every ``n > order`` (the exact hypothesis
          :func:`omnibias.core.verified.sequence_space.geometric_tail_bound`
          already takes, reused here with weight ``nu=1``);
        * a uniform hypothesis ``|psi_n(x)| <= psi_bound`` (or ``|\hat\psi_n(x)|``
          if the caller is using the normalized basis) for every ``n > order``,
          at whatever evaluation domain the caller intends.

        Raises :class:`ValueError` if ``psi_bound`` is negative (an admissibility
        check on the parameter, not a proof that it is the true bound) or if
        ``coeff_bound``/``ratio`` fail :func:`geometric_tail_bound`'s own checks.
        """
        if psi_bound < 0.0:
            raise ValueError("psi_bound must be non-negative")
        geo = geometric_tail_bound(coeff_bound, ratio, 1.0, self.order)
        return geo * Interval.point(float(psi_bound))

    def evaluate(
        self,
        x: IntervalLike,
        *,
        coeff_bound: float,
        ratio: float,
        psi_bound: float,
        normalized: bool = False,
    ) -> ComplexInterval:
        r"""``evaluate_kept(x)`` symmetrically inflated by :meth:`tail_bound`.

        A rigorous enclosure of the full (infinite) sum ``sum_{n>=0} c_n
        psi_n(x)`` under the same hypothesis :meth:`tail_bound` documents.
        """
        kept = self.evaluate_kept(x, normalized=normalized)
        b = self.tail_bound(coeff_bound=coeff_bound, ratio=ratio, psi_bound=psi_bound).hi
        pad = Interval(-b, b)
        return ComplexInterval(kept.re + pad, kept.im + pad)

    def fourier_transform_exact(self) -> HermiteExpansion:
        r"""Exact Fourier transform: ``c_n -> (-i)^n c_n`` (bit-for-bit, not rounded).

        Because the Hermite functions are exact Fourier eigenfunctions
        (``F[psi_n] = (-i)^n psi_n``, see the module docstring), transforming a
        finite expansion is *exactly* this diagonal relabelling of its own
        coefficients -- mirroring how :meth:`omnibias.core.verified.fourier.ValidatedFourierSeries.riesz`
        / ``.hilbert`` / ``.leray`` act coefficient-wise on that series' kept
        block. ``|(-i)^n| = 1`` is an isometry, so any :meth:`tail_bound` computed
        with the same ``(coeff_bound, ratio, psi_bound)`` hypothesis is
        identical before and after the transform (the ``order`` -- the only
        thing :meth:`tail_bound` depends on besides the hypothesis -- is
        unchanged).
        """
        return HermiteExpansion([_apply_neg_i_power(c, n) for n, c in enumerate(self.coeffs)])

    def __repr__(self) -> str:
        return f"HermiteExpansion(order={self.order}, coeffs={self.coeffs!r})"


__all__ = [
    "CRAMER_UNIFORM_BOUND",
    "HermiteExpansion",
    "gaussian_weight",
    "hermite_function",
    "hermite_function_normalized",
    "hermite_poly_coeffs_exact",
    "neg_i_power",
]
