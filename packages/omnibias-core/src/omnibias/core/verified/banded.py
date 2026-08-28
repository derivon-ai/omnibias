# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified tail-inverse-norm bound for a **banded** (non-diagonal) infinite operator.

Motivation -- the diagonal-only gap in :mod:`omnibias.core.verified.radii_spectral`
--------------------------------------------------------------------------------
:mod:`~omnibias.core.verified.radii_spectral` closes a Newton-Kantorovich /
radii-polynomial existence proof for ``F(a) = ell*a + Q(a,a) - f = 0`` by splitting
the approximate inverse ``A`` of ``DF(a_bar)`` into a numerically-inverted finite
block plus an *exact* tail piece.  That tail piece only works because its module
docstring's linear part ``ell`` is a **diagonal** Fourier multiplier: the tail
inverse is then just ``1/ell(k)`` pointwise, bounded by the scalar
``mu = sup_{|k|>N} |ell(k)|^{-1}``.

A **banded** linear operator -- one whose matrix in the same index basis has, for
every row, a bounded diagonal entry plus a *finite* (or uniformly summable) set of
nonzero off-diagonal entries coupling nearby indices -- has no such elementary
tail inverse.  The canonical example is the self-similar scaling generator
``x . grad`` of a finite-time-singularity ansatz: in a Fourier / Chebyshev /
Hermite basis it couples each mode to a handful of its neighbours, so it is
genuinely non-diagonal but still *banded* (a fixed, small number of nonzero
off-diagonal entries per row, or a uniformly bounded row sum for a decaying
envelope). This module supplies the missing tail-inverse-norm bound for that
banded case, so that :mod:`omnibias.core.verified.radii_spectral` can accept a
non-diagonal but banded linear part via :class:`~omnibias.core.verified.radii_spectral.BandedLinearPart`
and :func:`~omnibias.core.verified.radii_spectral.tail_inverse_bound_from_banded`.
This module still only supplies the tail-inverse bound itself.

The bound: Gershgorin diagonal dominance + a Neumann series
------------------------------------------------------------
Let ``M`` be the linear operator restricted to the tail index set (``|k| > N`` or
``n > N``, in whatever integer-indexed basis the caller uses), written as
``M = D - R`` where ``D = diag(M)`` and ``R`` is the negative off-diagonal part
(``R_ii = 0``). Suppose, uniformly over every tail row ``i``:

.. math::

    |M_{ii}| \ge d_{\min} > 0,
    \qquad
    \sum_{j \ne i} |M_{ij}| \le s < d_{\min}.

(``s`` need not come from literally finite support -- a uniformly bounded row sum
of a decaying envelope, e.g. ``|M_{i,i+d}| \le c\,\rho^{|d|}`` with ``\rho < 1``,
satisfies the same hypothesis; see :func:`geometric_band_row_sum_bound`.)

Then ``D^{-1}R`` has operator norm (induced by the sup norm, i.e. the matrix
*infinity* norm -- max absolute row sum, exactly the norm
:func:`omnibias.core.verified.linalg.inf_norm_matrix` computes for a finite
matrix) bounded by

.. math::

    \|D^{-1}R\|_\infty
        \le \sup_i \frac{1}{|M_{ii}|}\sum_{j\ne i}|M_{ij}|
        \le \frac{s}{d_{\min}} < 1,

so by the classical Neumann-series lemma (the same lemma
:func:`omnibias.core.verified.linalg.neumann_inverse_norm_bound` uses for a
*finite* matrix, except here the bound is a closed form in ``(d_min, s)`` and
needs no floating-point matrix or its numerical inverse at all) ``I - D^{-1}R``
is invertible with ``\|(I-D^{-1}R)^{-1}\|_\infty \le d_{\min}/(d_{\min}-s)``.
Since ``M = D(I - D^{-1}R)``,

.. math::

    \|M^{-1}\|_\infty
      = \|(I-D^{-1}R)^{-1} D^{-1}\|_\infty
      \le \|(I-D^{-1}R)^{-1}\|_\infty \, \|D^{-1}\|_\infty
      \le \frac{d_{\min}}{d_{\min}-s}\cdot\frac{1}{d_{\min}}
      = \frac{1}{d_{\min}-s}.

This is exactly the classical *Levy-Desplanques* / Gershgorin diagonal-dominance
invertibility theorem, generalised verbatim to an infinite (tail) operator: the
argument never uses finiteness, only that each row sum is finite and uniformly
bounded by ``s``. Setting ``s = 0`` (a purely diagonal tail, as in
``radii_spectral``) recovers exactly ``radii_spectral``'s
``mu = sup |ell(k)|^{-1} = 1/d_min``.

Which norm this certifies -- and which it does **not**
--------------------------------------------------------
:func:`banded_tail_inverse_bound` is honest about scope: it certifies
``\|M^{-1}\|`` for the operator norm **induced by the (unweighted) sup norm**
``ell^infty`` on the tail sequence space -- the same "matrix infinity norm / max
absolute row sum" convention already used throughout
:mod:`omnibias.core.verified.linalg` (``inf_norm_matrix``,
``neumann_inverse_norm_bound``). The identical row-sum-vs-diagonal algebra also
bounds the induced norm of a **weighted** sup space ``ell^infty_w`` (weight
``w_i > 0``) *if* the caller recomputes ``s`` as the corresponding weighted row
sum ``sup_i (1/w_i) sum_{j!=i} w_j |M_ij|`` -- the derivation above goes through
unchanged with ``D^{-1}R`` replaced by its weighted-similarity transform, since
diagonal conjugation by ``diag(w)`` does not change a diagonal entry's
magnitude. This module does not attempt that transform for the caller; it only
certifies the bound once ``d_min`` / ``s`` are supplied in a consistent norm.

It does **not** certify a bound for the **weighted ell^1_nu** operator norm that
``radii_spectral`` and ``sequence_space`` actually use for the Banach-algebra
tail (that norm's induced operator bound is a *column*-sum criterion, i.e. the
transpose picture, and mixing row- and column-sum hypotheses on a banded --
rather than diagonal -- operator is exactly the extra bookkeeping a diagonal
multiplier avoids for free). A caller wiring this into
``radii_spectral`` / ``kantorovich`` therefore has two honest options, and must
pick one explicitly rather than silently reusing this bound as if it were an
``ell^1_nu`` bound:

#. reformulate the relevant tail estimate (``Z1``/``Z2`` in
   :func:`omnibias.core.verified.kantorovich.radii_polynomial_certificate`) in the
   sup-norm topology throughout, so :func:`banded_tail_inverse_bound`'s ``ell^infty``
   bound is exactly the tail operator norm the radii polynomial needs; or
#. recompute ``off_diagonal_row_sum_upper`` as a **column**-sum (or
   ``nu``-weighted row-sum) quantity consistent with the ``ell^1_nu`` norm before
   calling this function -- the arithmetic identity ``1/(d_min - s)`` is unchanged,
   only the meaning of ``d_min``/``s`` changes.

Either way, the plug-in point is exactly where ``radii_spectral.SpectralProblem``
today takes a scalar ``tail_inverse_bound: float`` (``mu``): a banded-capable
successor would replace that field with the ``.hi`` of this function's return
value, once the caller has resolved which norm convention its ``Z1``/``Z2``
estimates are stated in. Implementing that successor is explicitly **out of
scope** for this module (Phase 2 work).

Soundness discipline
---------------------
:func:`banded_tail_inverse_bound` raises :class:`ValueError` -- rather than
returning a bound -- whenever the diagonal-dominance hypothesis
``off_diagonal_row_sum_upper < diag_lower`` is not satisfied (including the
degenerate outward-rounded margin), matching the "raise, don't degrade" doctrine
used throughout :mod:`omnibias.core.verified` (e.g.
``interval_ldlt_inertia``, ``interval_solve``). Every arithmetic step is outward-
rounded :class:`~omnibias.core.verified.interval.Interval` algebra; the module has
no NumPy or floating-point-matrix dependency, matching the pure-scalar style of
:mod:`omnibias.core.verified.radii_spectral`'s
``laplacian_tail_inverse_bound``.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.verified.interval import Interval, sum_intervals


def banded_tail_inverse_bound(
    diag_lower: float,
    off_diagonal_row_sum_upper: float,
    *,
    bandwidth: int | None = None,
) -> Interval:
    r"""Certified ``||M^{-1}||_infty`` bound for a row-wise diagonally-dominant tail.

    Hypothesis (the caller's proof obligation, uniform over every tail row ``i``
    of the banded operator ``M`` -- e.g. ``|k| > N`` or ``n > N``):

    .. math::

        |M_{ii}| \ge \texttt{diag\_lower} > 0,
        \qquad
        \sum_{j \ne i} |M_{ij}| \le \texttt{off\_diagonal\_row\_sum\_upper}.

    Returns a rigorous enclosure of ``1/(diag_lower - off_diagonal_row_sum_upper)``;
    its ``.hi`` is the certified upper bound on the operator norm ``||M^{-1}||``
    induced by the (unweighted) sup norm ``ell^infty`` on the tail -- see the
    module docstring for the Gershgorin / Neumann-series derivation, the precise
    norm this does and does not cover, and how a caller would plug it into a
    ``radii_spectral``-style ``Z1``/``Z2`` estimate.

    ``bandwidth`` is optional provenance only (e.g. the number of nonzero
    off-diagonal entries per row a caller used to build
    ``off_diagonal_row_sum_upper`` via :func:`finite_band_row_sum_bound`); it is
    validated but never enters the arithmetic, because the bound holds for any
    row sum -- finite band or summable decaying envelope alike.

    Raises :class:`ValueError` -- rather than fabricating a bound -- when
    ``diag_lower`` is not strictly positive, when
    ``off_diagonal_row_sum_upper`` is negative, or when the row-wise
    diagonal-dominance margin ``diag_lower - off_diagonal_row_sum_upper`` is not
    certifiably positive after outward rounding.
    """
    diag_lower = float(diag_lower)
    off_diagonal_row_sum_upper = float(off_diagonal_row_sum_upper)
    if diag_lower <= 0.0:
        raise ValueError(
            f"diag_lower must be strictly positive (a coercivity hypothesis), got {diag_lower!r}"
        )
    if off_diagonal_row_sum_upper < 0.0:
        raise ValueError(
            "off_diagonal_row_sum_upper must be non-negative (it bounds a sum of "
            f"magnitudes), got {off_diagonal_row_sum_upper!r}"
        )
    if bandwidth is not None and bandwidth < 0:
        raise ValueError(f"bandwidth must be non-negative when supplied, got {bandwidth!r}")
    if off_diagonal_row_sum_upper >= diag_lower:
        raise ValueError(
            "row-wise diagonal dominance requires off_diagonal_row_sum_upper < "
            f"diag_lower; got off_diagonal_row_sum_upper={off_diagonal_row_sum_upper!r} "
            f">= diag_lower={diag_lower!r}"
        )
    margin = Interval.point(diag_lower) - Interval.point(off_diagonal_row_sum_upper)
    if margin.lo <= 0.0:
        raise ValueError(
            "row-wise diagonal-dominance margin is not certifiably positive after "
            "outward rounding; widen the gap between diag_lower and "
            "off_diagonal_row_sum_upper"
        )
    return margin.reciprocal()


def finite_band_row_sum_bound(band_magnitude_bounds: Sequence[float]) -> float:
    r"""Outward-rounded row-sum bound ``s`` from explicit per-offset band magnitudes.

    ``band_magnitude_bounds`` lists a rigorous upper bound on ``|M_{i, i+d}|`` for
    each of the finitely many nonzero offsets ``d`` a banded row couples to
    (excluding the diagonal ``d = 0`` itself) -- e.g. the two neighbour couplings
    ``d = -1, +1`` a nearest-neighbour scaling generator produces in a Fourier /
    Chebyshev basis. Returns ``sum(band_magnitude_bounds)`` (outward rounded), the
    ``off_diagonal_row_sum_upper`` that :func:`banded_tail_inverse_bound` consumes.

    Raises :class:`ValueError` if any bound is negative.
    """
    bounds = [float(b) for b in band_magnitude_bounds]
    if any(b < 0.0 for b in bounds):
        raise ValueError("band magnitude bounds must be non-negative")
    if not bounds:
        return 0.0
    return sum_intervals([Interval.point(b) for b in bounds]).hi


def geometric_band_row_sum_bound(coeff: float, ratio: float) -> float:
    r"""Outward-rounded row-sum bound for a two-sided geometric-decay band envelope.

    Models the per-row hypothesis ``|M_{i, i+d}| <= coeff * ratio^{|d|}`` for
    every nonzero offset ``d`` (a "decaying envelope" that need not have finite
    support). Requires ``coeff >= 0`` and ``0 <= ratio < 1``; the row sum over all
    ``d != 0`` is the two-sided geometric series

    .. math::

        \sum_{d \ne 0} \texttt{coeff}\cdot\texttt{ratio}^{|d|}
            = \frac{2\,\texttt{coeff}\cdot\texttt{ratio}}{1-\texttt{ratio}}

    (outward rounded), the ``off_diagonal_row_sum_upper`` that
    :func:`banded_tail_inverse_bound` consumes.
    """
    coeff = float(coeff)
    ratio = float(ratio)
    if coeff < 0.0:
        raise ValueError(f"coeff must be non-negative, got {coeff!r}")
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"ratio must be in [0, 1) for the row sum to converge, got {ratio!r}")
    c = Interval.point(coeff)
    r = Interval.point(ratio)
    one_minus = Interval.point(1.0) - r
    return (Interval.point(2.0) * c * r / one_minus).hi


__all__ = [
    "banded_tail_inverse_bound",
    "finite_band_row_sum_bound",
    "geometric_band_row_sum_bound",
]
