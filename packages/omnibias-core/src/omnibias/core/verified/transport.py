# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified optimal transport: rigorous 1-D Wasserstein-1 enclosure.

The Wasserstein-1 distance between a continuous model CDF ``F`` and the empirical
CDF ``F_n`` of samples is the pure CDF functional

.. math::

    W_1(F, F_n) = \int_{-\infty}^{\infty} \lvert F(x) - F_n(x) \rvert \, dx,

so it is certifiable directly from the CDF -- here via its *exact antiderivative*
rather than quadrature, which makes the enclosure tight. For a logistic
(``sigmoid``) CDF ``F(x) = sigmoid((x - loc)/s)`` the antiderivative is
``Phi(x) = s * softplus((x - loc)/s)`` with ``Phi(-inf) = 0``; the ``tanh`` CDF
is the same logistic with half the scale (``tanh CDF == sigmoid CDF`` at scale
``s/2``).

The integral splits at the order statistics ``x_(1) <= ... <= x_(n)``: two tails
in closed form, and on each interior panel ``[x_(i), x_(i+1)]`` (where ``F_n``
holds the constant level ``c = i/n``) the exact ``int |F - c|`` via the crossing
point ``x* = F^{-1}(c)`` clamped to the panel -- every quantity an outward-rounded
:class:`~omnibias.core.verified.interval.Interval`.

``arctan`` (Cauchy) is intentionally **unsupported**: it has no first moment, so
``W_1`` is infinite.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import ln_iv, softplus_iv

#: Bases with a finite-mean (hence finite ``W_1``) location-scale CDF.
SUPPORTED_W1_CDFS: tuple[str, ...] = ("sigmoid", "tanh")


def _effective_scale(name: str, scale: float) -> float:
    """Logistic scale ``s`` such that ``F == sigmoid((x - loc)/s)``."""
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    key = name.lower()
    if key == "sigmoid":
        return scale
    if key == "tanh":
        return scale * 0.5  # tanh CDF (t+1)/2 == sigmoid at half the scale
    raise NotImplementedError(
        f"certified W1 supports {SUPPORTED_W1_CDFS} (finite first moment); "
        f"{name!r} has no finite mean so W1 is infinite"
    )


def _phi(x: Interval, loc: float, s: float) -> Interval:
    """Logistic CDF antiderivative ``Phi(x) = s * softplus((x - loc)/s)``."""
    arg = (x - Interval.point(loc)) * Interval.point(s).reciprocal()
    return Interval.point(s) * softplus_iv(arg)


def _panel_integral(a: float, b: float, c: Fraction, loc: float, s: float) -> Interval:
    r"""Rigorous enclosure of ``int_a^b |F(x) - c| dx`` for the logistic ``F``.

    Uses the identity ``int_a^b |F - c| = c(2t - a - b) + Phi(a) + Phi(b) - 2 Phi(t)``
    with ``t = clamp(F^{-1}(c), a, b)`` and ``F^{-1}(c) = loc + s*(ln c - ln(1-c))``.
    The formula is exact for ``t = F^{-1}(c)`` inside ``[a, b]`` and continues to
    the correct monotone-branch value at the clamped endpoints, so clamping the
    rigorous ``x*`` enclosure keeps the result a guaranteed enclosure.
    """
    c_iv = Interval.from_rational(c)
    phi_a = _phi(Interval.point(a), loc, s)
    phi_b = _phi(Interval.point(b), loc, s)
    one = Interval.point(1.0)
    xstar = Interval.point(loc) + Interval.point(s) * (ln_iv(c_iv) - ln_iv(one - c_iv))
    t_lo = min(max(xstar.lo, a), b)
    t_hi = min(max(xstar.hi, a), b)
    t = Interval(t_lo, t_hi)
    phi_t = _phi(t, loc, s)
    integral = c_iv * (t * 2 - (Interval.point(a) + Interval.point(b))) + phi_a + phi_b - phi_t * 2
    return Interval(max(integral.lo, 0.0), integral.hi)  # |.| integral >= 0


def certified_wasserstein1(
    name: str,
    samples: Sequence[float],
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Interval:
    r"""Rigorous enclosure of ``W_1(F, F_n)`` for a location-scale model CDF.

    ``F`` is the ``sigmoid`` / ``tanh`` CDF with the given ``loc`` / ``scale`` and
    ``F_n`` is the empirical CDF of ``samples``. Returns a guaranteed
    :class:`Interval` enclosing the true Wasserstein-1 distance (in the same units
    as ``x``). Raises :class:`NotImplementedError` for bases without a finite mean
    (see :data:`SUPPORTED_W1_CDFS`).
    """
    s = _effective_scale(name, scale)
    xs = sorted(float(v) for v in samples)
    n = len(xs)
    if n == 0:
        raise ValueError("certified_wasserstein1 needs at least one sample")
    recip_s = Interval.point(s).reciprocal()
    # Left tail (-inf, x_(1)]: int F dx = Phi(x_(1)); right tail [x_(n), inf):
    # int (1 - F) dx = s * softplus(-(x_(n) - loc)/s).
    total = _phi(Interval.point(xs[0]), loc, s)
    arg_right = -((Interval.point(xs[-1]) - Interval.point(loc)) * recip_s)
    total = total + Interval.point(s) * softplus_iv(arg_right)
    for i in range(1, n):
        a, b = xs[i - 1], xs[i]
        if b <= a:
            continue  # tie -> zero-width panel
        total = total + _panel_integral(a, b, Fraction(i, n), loc, s)
    return total


def certified_wasserstein1_samples(
    u: Sequence[float],
    v: Sequence[float],
) -> Interval:
    r"""Rigorous enclosure of the two-sample 1-D ``W_1`` between equal-size samples.

    For two samples of equal length ``n`` the Wasserstein-1 distance between their
    empirical distributions is the mean absolute difference of the order
    statistics,

    .. math::

        W_1(F_u, F_v) = \frac1n \sum_{i=1}^{n} \lvert u_{(i)} - v_{(i)} \rvert,

    which is exact in one dimension. The samples are sorted exactly and the mean
    is accumulated in outward-rounded interval arithmetic, so the result is a
    guaranteed enclosure of the true value -- here it bounds only the
    floating-point rounding of the closed-form distance. This is the
    proof-carrying twin of the differentiable two-sample
    ``omnibias.{torch,jax}.information.wasserstein1`` (the model-vs-empirical
    distance is :func:`certified_wasserstein1`).
    """
    if len(u) != len(v):
        raise ValueError(
            f"certified_wasserstein1_samples needs equal-length samples, got "
            f"{len(u)} and {len(v)}"
        )
    if len(u) == 0:
        raise ValueError("certified_wasserstein1_samples needs at least one sample")
    us = sorted(float(a) for a in u)
    vs = sorted(float(b) for b in v)
    n = len(us)
    acc = Interval.point(0.0)
    for a, b in zip(us, vs, strict=True):
        acc = acc + (Interval.point(a) - Interval.point(b)).abs()
    return acc * Interval.from_rational(Fraction(1, n))


def certified_wasserstein2_samples(
    u: Sequence[float],
    v: Sequence[float],
) -> Interval:
    r"""Rigorous enclosure of the two-sample 1-D ``W_2`` between equal-size samples.

    For two samples of equal length ``n`` the Wasserstein-2 distance between their
    empirical distributions is the root-mean-square of the sorted differences,

    .. math::

        W_2(F_u, F_v) = \Bigl(\frac1n \sum_{i=1}^{n} (u_{(i)} - v_{(i)})^2\Bigr)^{1/2},

    exact in one dimension. The samples are sorted exactly and the mean square is
    accumulated in outward-rounded interval arithmetic before the rigorous
    :meth:`~omnibias.core.verified.interval.Interval.sqrt`, so the result is a
    guaranteed enclosure -- the proof-carrying twin of the differentiable
    ``omnibias.{torch,jax}.information.wassersteinp`` at ``p = 2``.
    """
    if len(u) != len(v):
        raise ValueError(
            f"certified_wasserstein2_samples needs equal-length samples, got "
            f"{len(u)} and {len(v)}"
        )
    if len(u) == 0:
        raise ValueError("certified_wasserstein2_samples needs at least one sample")
    us = sorted(float(a) for a in u)
    vs = sorted(float(b) for b in v)
    n = len(us)
    acc = Interval.point(0.0)
    for a, b in zip(us, vs, strict=True):
        acc = acc + (Interval.point(a) - Interval.point(b)).pow_int(2)
    mean_sq = acc * Interval.from_rational(Fraction(1, n))
    return Interval(max(mean_sq.lo, 0.0), mean_sq.hi).sqrt()  # radicand >= 0


def certified_wasserstein2_gaussian(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
) -> Interval:
    r"""Rigorous enclosure of the 1-D Gaussian ``W_2`` distance.

    For two univariate Gaussians the Wasserstein-2 distance has the closed form
    ``W_2 = sqrt((mu1 - mu2)^2 + (sigma1 - sigma2)^2)`` (standard deviations,
    ``sigma >= 0``). Every operation is outward-rounded, so the returned
    :class:`Interval` is a guaranteed enclosure -- the proof-carrying twin of
    :func:`omnibias.{torch,jax}.information.wasserstein2_gaussian`.
    """
    if sigma1 < 0.0 or sigma2 < 0.0:
        raise ValueError(
            f"standard deviations must be >= 0, got {sigma1} and {sigma2}"
        )
    dmu = (Interval.point(mu1) - Interval.point(mu2)).pow_int(2)
    dsigma = (Interval.point(sigma1) - Interval.point(sigma2)).pow_int(2)
    radicand = dmu + dsigma
    return Interval(max(radicand.lo, 0.0), radicand.hi).sqrt()  # radicand >= 0


__all__ = [
    "SUPPORTED_W1_CDFS",
    "certified_wasserstein1",
    "certified_wasserstein1_samples",
    "certified_wasserstein2_gaussian",
    "certified_wasserstein2_samples",
]
