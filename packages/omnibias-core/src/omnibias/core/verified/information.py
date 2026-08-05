# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified information theory: rigorous entropy / divergence enclosures.

The differentiable information operators in ``omnibias.{torch,jax}.information``
get their *proof-carrying* counterparts here. Every functional of a discrete
distribution is returned as a guaranteed
:class:`~omnibias.core.verified.interval.Interval` enclosure, built from the
monotone ``ln`` enclosure in :mod:`omnibias.core.verified.transcend` and the
outward-rounded interval algebra.

Inputs are sequences of probabilities, each an
:data:`~omnibias.core.verified.interval.IntervalLike` -- a plain float, or an
:class:`Interval` (e.g. a band-mass enclosure from
:func:`omnibias.core.verified.probability.band_mass_enclosure`). The bridge
helper :func:`binned_distribution_enclosure` turns a location-scale model CDF
into exactly such a vector of certified bin masses, so

    ``entropy_enclosure(binned_distribution_enclosure("sigmoid", edges))``

is a rigorous enclosure of the binned model entropy.

Conventions (all in nats):

* ``0 ln 0 := 0`` -- an exact-zero probability contributes nothing.
* Entropy / cross-entropy / KL / JS are clamped to their proven sign
  (``H >= 0``, ``D >= 0``) since the true value cannot be negative.
* ``ln`` needs a strictly positive argument: a probability that merely
  *straddles* zero (``lo == 0 < hi``) is rejected with a clear error -- supply a
  positive lower bound (or an exact ``0``) instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from omnibias.core.verified.interval import Interval, IntervalLike, sum_intervals
from omnibias.core.verified.probability import band_mass_enclosure
from omnibias.core.verified.transcend import ln_iv

#: Outward (upper) bound on ``1/e`` -- the maximum of ``g(x) = -x ln x``.
_E_INV_HI = math.nextafter(math.exp(-1.0), math.inf)
_E_INV_LO = math.nextafter(math.exp(-1.0), -math.inf)


def _check_prob(p: Interval) -> None:
    if p.lo < 0.0 or p.hi > 1.0:
        raise ValueError(f"probability must lie in [0, 1], got [{p.lo}, {p.hi}]")


def _neg_x_ln_x(p: Interval) -> Interval:
    r"""Tight rigorous enclosure of ``g(x) = -x ln x`` over ``p`` (``g(0) := 0``).

    ``g`` rises on ``(0, 1/e)`` to its maximum ``1/e`` and falls on ``(1/e, 1)``,
    so over ``[a, b]`` its minimum is at an endpoint (or ``0`` if ``a == 0``) and
    its maximum is at an endpoint unless ``1/e`` lies inside, where it is ``1/e``.
    """
    _check_prob(p)
    a, b = p.lo, p.hi

    def g(x: float) -> Interval:
        if x == 0.0:
            return Interval.point(0.0)  # 0 ln 0 := 0
        xi = Interval.point(x)
        return -(xi * ln_iv(xi))

    ga, gb = g(a), g(b)
    lo = max(0.0, min(ga.lo, gb.lo))  # g >= 0 on [0, 1]
    hi = max(ga.hi, gb.hi)
    if a <= _E_INV_HI and _E_INV_LO <= b:  # 1/e possibly inside -> include the peak
        hi = max(hi, _E_INV_HI)
    return Interval(lo, hi)


def entropy_enclosure(probs: Sequence[IntervalLike]) -> Interval:
    r"""Rigorous enclosure of the Shannon entropy ``H = -sum_i p_i ln p_i`` (nats).

    Does not assume ``sum_i p_i == 1`` (it evaluates the functional as given); the
    result is clamped to ``H >= 0``.
    """
    terms = [_neg_x_ln_x(Interval.from_value(p)) for p in probs]
    if not terms:
        raise ValueError("entropy_enclosure needs at least one probability")
    return sum_intervals(terms)


def cross_entropy_enclosure(
    p: Sequence[IntervalLike], q: Sequence[IntervalLike]
) -> Interval:
    r"""Rigorous enclosure of the cross-entropy ``H(p, q) = -sum_i p_i ln q_i``.

    Requires ``q_i`` strictly positive wherever ``p_i`` can be positive; clamped
    to ``H(p, q) >= 0``.
    """
    if len(p) != len(q):
        raise ValueError(f"length mismatch: len(p)={len(p)} != len(q)={len(q)}")
    terms: list[Interval] = []
    for pi_raw, qi_raw in zip(p, q, strict=True):
        pi = Interval.from_value(pi_raw)
        qi = Interval.from_value(qi_raw)
        _check_prob(pi)
        _check_prob(qi)
        if pi.hi == 0.0:
            terms.append(Interval.point(0.0))
            continue
        if qi.lo <= 0.0:
            raise ValueError("cross_entropy_enclosure needs q_i > 0 where p_i > 0")
        terms.append(-(pi * ln_iv(qi)))
    s = sum_intervals(terms)
    return Interval(max(s.lo, 0.0), s.hi)


def kl_divergence_enclosure(
    p: Sequence[IntervalLike], q: Sequence[IntervalLike]
) -> Interval:
    r"""Rigorous enclosure of ``D(p || q) = sum_i p_i ln(p_i / q_i)`` (nats).

    Each ``p_i`` must have a positive lower bound or be exactly ``0`` (then its
    term is ``0``); each ``q_i`` with ``p_i > 0`` must be strictly positive. The
    result is clamped to ``D >= 0`` (Gibbs' inequality).
    """
    if len(p) != len(q):
        raise ValueError(f"length mismatch: len(p)={len(p)} != len(q)={len(q)}")
    terms: list[Interval] = []
    for pi_raw, qi_raw in zip(p, q, strict=True):
        pi = Interval.from_value(pi_raw)
        qi = Interval.from_value(qi_raw)
        _check_prob(pi)
        _check_prob(qi)
        if pi.hi == 0.0:
            terms.append(Interval.point(0.0))
            continue
        if pi.lo <= 0.0:
            raise ValueError(
                "kl_divergence_enclosure needs p_i with a positive lower bound "
                "(or exactly 0)"
            )
        if qi.lo <= 0.0:
            raise ValueError("kl_divergence_enclosure needs q_i > 0 where p_i > 0")
        terms.append(pi * (ln_iv(pi) - ln_iv(qi)))
    s = sum_intervals(terms)
    return Interval(max(s.lo, 0.0), s.hi)


def js_divergence_enclosure(
    p: Sequence[IntervalLike], q: Sequence[IntervalLike]
) -> Interval:
    r"""Rigorous enclosure of the Jensen-Shannon divergence (nats).

    ``JS = 1/2 D(p || m) + 1/2 D(q || m)`` with ``m = (p + q)/2``. Symmetric and
    bounded in ``[0, ln 2]``; the enclosure is clamped to that range.
    """
    if len(p) != len(q):
        raise ValueError(f"length mismatch: len(p)={len(p)} != len(q)={len(q)}")
    half = Interval.point(0.5)
    m = [half * (Interval.from_value(pi) + Interval.from_value(qi)) for pi, qi in zip(p, q, strict=True)]
    js = half * kl_divergence_enclosure(p, m) + half * kl_divergence_enclosure(q, m)
    ln2_hi = math.nextafter(math.log(2.0), math.inf)
    return Interval(max(js.lo, 0.0), min(js.hi, ln2_hi))


def mutual_information_enclosure(
    joint: Sequence[Sequence[IntervalLike]],
) -> Interval:
    r"""Rigorous enclosure of the mutual information ``I(X; Y)`` of a joint table.

    ``joint[i][j]`` is the certified probability ``P(X=i, Y=j)`` -- any
    :data:`IntervalLike` (float, or e.g. a band-mass :class:`Interval`) -- of a
    discrete joint distribution that sums to one over the whole table. The
    identity ``I(X; Y) = D(P || p_X (x) p_Y)`` (KL of the joint against the
    product of its marginals) is used directly, so this delegates to
    :func:`kl_divergence_enclosure` against the rigorous outer product of the
    interval marginals. Consequences:

    * ``I >= 0`` is guaranteed (the KL enclosure is clamped to its proven sign);
    * the ``0 ln 0 := 0`` convention applies per cell (an exact-``0`` cell adds
      nothing), while a cell that merely *straddles* zero is rejected with the
      same clear error as :func:`kl_divergence_enclosure` -- supply a positive
      lower bound (or exact ``0``).

    Symmetric in ``X`` and ``Y`` (transposing ``joint`` gives the same value).
    """
    rows = [[Interval.from_value(v) for v in row] for row in joint]
    if not rows or not rows[0]:
        raise ValueError("mutual_information_enclosure needs a non-empty 2-D table")
    n_cols = len(rows[0])
    if any(len(row) != n_cols for row in rows):
        raise ValueError("mutual_information_enclosure needs a rectangular table")
    px = [sum_intervals(row) for row in rows]  # marginal over Y, one per X row
    py = [sum_intervals([rows[i][j] for i in range(len(rows))]) for j in range(n_cols)]
    p_flat: list[Interval] = []
    prod_flat: list[Interval] = []
    for i, row in enumerate(rows):
        for j in range(n_cols):
            p_flat.append(row[j])
            prod_flat.append(px[i] * py[j])
    return kl_divergence_enclosure(p_flat, prod_flat)


def total_variation_enclosure(
    p: Sequence[IntervalLike], q: Sequence[IntervalLike]
) -> Interval:
    r"""Rigorous enclosure of the total-variation distance ``TV = 1/2 sum_i |p_i - q_i|``.

    Symmetric, metric, and bounded in ``[0, 1]`` for probability vectors; the
    enclosure is clamped to that range. The proof-carrying twin of
    :func:`omnibias.{torch,jax}.information.total_variation_distance`.
    """
    if len(p) != len(q):
        raise ValueError(f"length mismatch: len(p)={len(p)} != len(q)={len(q)}")
    if len(p) == 0:
        raise ValueError("total_variation_enclosure needs at least one probability")
    terms: list[Interval] = []
    for pi_raw, qi_raw in zip(p, q, strict=True):
        pi = Interval.from_value(pi_raw)
        qi = Interval.from_value(qi_raw)
        _check_prob(pi)
        _check_prob(qi)
        terms.append((pi - qi).abs())
    s = Interval.point(0.5) * sum_intervals(terms)
    return Interval(max(s.lo, 0.0), min(s.hi, 1.0))


def hellinger_enclosure(
    p: Sequence[IntervalLike], q: Sequence[IntervalLike]
) -> Interval:
    r"""Rigorous enclosure of the Hellinger distance ``H(p, q)`` in ``[0, 1]``.

    ``H^2 = 1/2 sum_i (sqrt(p_i) - sqrt(q_i))^2`` via the rigorous interval
    :meth:`~omnibias.core.verified.interval.Interval.sqrt`; the squared distance is
    clamped to ``[0, 1]`` before the outer square root. Symmetric and metric -- the
    proof-carrying twin of
    :func:`omnibias.{torch,jax}.information.hellinger_distance`.
    """
    if len(p) != len(q):
        raise ValueError(f"length mismatch: len(p)={len(p)} != len(q)={len(q)}")
    if len(p) == 0:
        raise ValueError("hellinger_enclosure needs at least one probability")
    terms: list[Interval] = []
    for pi_raw, qi_raw in zip(p, q, strict=True):
        pi = Interval.from_value(pi_raw)
        qi = Interval.from_value(qi_raw)
        _check_prob(pi)
        _check_prob(qi)
        terms.append((pi.sqrt() - qi.sqrt()).pow_int(2))
    h2 = Interval.point(0.5) * sum_intervals(terms)
    return Interval(max(h2.lo, 0.0), min(h2.hi, 1.0)).sqrt()


def chi_squared_enclosure(
    p: Sequence[IntervalLike], q: Sequence[IntervalLike]
) -> Interval:
    r"""Rigorous enclosure of Pearson's ``chi^2(p || q) = sum_i (p_i - q_i)^2 / q_i``.

    Each ``q_i`` must be strictly positive (positive lower bound). Clamped to
    ``chi^2 >= 0``. The proof-carrying twin of
    :func:`omnibias.{torch,jax}.information.chi_squared_divergence`.
    """
    if len(p) != len(q):
        raise ValueError(f"length mismatch: len(p)={len(p)} != len(q)={len(q)}")
    if len(p) == 0:
        raise ValueError("chi_squared_enclosure needs at least one probability")
    terms: list[Interval] = []
    for pi_raw, qi_raw in zip(p, q, strict=True):
        pi = Interval.from_value(pi_raw)
        qi = Interval.from_value(qi_raw)
        _check_prob(pi)
        _check_prob(qi)
        if qi.lo <= 0.0:
            raise ValueError("chi_squared_enclosure needs q_i > 0")
        terms.append((pi - qi).pow_int(2) * qi.reciprocal())
    s = sum_intervals(terms)
    return Interval(max(s.lo, 0.0), s.hi)


def binned_distribution_enclosure(
    name: str,
    edges: Sequence[float],
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> list[Interval]:
    r"""Certified per-bin masses of a location-scale model CDF over ``edges``.

    Bin ``i`` gets the rigorous mass enclosure ``F(edges[i+1]) - F(edges[i])``
    from :func:`omnibias.core.verified.probability.band_mass_enclosure`. Feed the
    result straight into :func:`entropy_enclosure` / :func:`kl_divergence_enclosure`
    for a proof-carrying entropy / divergence of the binned model distribution.
    """
    if len(edges) < 2:
        raise ValueError("binned_distribution_enclosure needs at least two edges")
    return [
        band_mass_enclosure(name, edges[i], edges[i + 1], loc=loc, scale=scale)
        for i in range(len(edges) - 1)
    ]


__all__ = [
    "binned_distribution_enclosure",
    "chi_squared_enclosure",
    "cross_entropy_enclosure",
    "entropy_enclosure",
    "hellinger_enclosure",
    "js_divergence_enclosure",
    "kl_divergence_enclosure",
    "mutual_information_enclosure",
    "total_variation_enclosure",
]
