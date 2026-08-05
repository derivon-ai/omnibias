# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified probabilities: rigorous CDF / band-mass enclosures + DKW goodness-of-fit.

Combines two *rigorous* ingredients into a proof-carrying statistical test:

* **model side** -- a guaranteed enclosure ``[F_lo, F_hi]`` of a location-scale
  CDF on a ``sigmoid`` / ``tanh`` / ``arctan`` base, built from the monotone
  transcendental enclosures in :mod:`omnibias.core.verified.transcend` (clamped
  to the true range ``[0, 1]``);
* **data side** -- the distribution-free DKW envelope ``F_n +/- eps_n(alpha)``
  around the empirical CDF (:func:`omnibias.core.probability.dkw_epsilon`).

:func:`certified_gof` reports a *guaranteed lower bound* on the Kolmogorov-Smirnov
statistic ``sup_x |F_n(x) - F(x)|`` using the model enclosure; if that lower bound
exceeds ``eps_n(alpha)`` the model's true CDF cannot be the data-generating CDF at
level ``alpha`` -- a rigorous rejection (the DKW bad event has probability
``<= alpha``). Every quantity is outward-rounded, matching the certified-evidence contract.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.probability import dkw_epsilon
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import atan_iv, backend_name, sigmoid_iv, tanh_iv

#: Rigorous enclosure of pi (``math.pi`` is the nearest double below the truth).
_PI = Interval(math.pi, math.nextafter(math.pi, math.inf))

#: Activation bases with a certified location-scale CDF enclosure.
SUPPORTED_CDFS: tuple[str, ...] = ("sigmoid", "tanh", "arctan")


def _standardize(x: float, loc: float, scale: float) -> Interval:
    """Rigorous enclosure of the standardised argument ``(x - loc)/scale``."""
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    num = Interval.point(float(x)) - Interval.point(float(loc))
    return num * Interval.point(float(scale)).reciprocal()


def cdf_enclosure(
    name: str, x: float, *, loc: float = 0.0, scale: float = 1.0
) -> Interval:
    r"""Rigorous enclosure of the location-scale CDF ``F(x)``, clamped to ``[0, 1]``.

    ``F(x) = G((x - loc)/scale)`` with ``G`` the base activation rescaled to a CDF:
    ``sigmoid`` (identity), ``tanh`` (``(t + 1)/2``), ``arctan`` (``t/pi + 1/2``).
    Unsupported bases raise :class:`NotImplementedError` (see :data:`SUPPORTED_CDFS`).
    """
    arg = _standardize(x, loc, scale)
    key = name.lower()
    if key == "sigmoid":
        f = sigmoid_iv(arg)
    elif key == "tanh":
        f = tanh_iv(arg) * 0.5 + 0.5
    elif key in ("arctan", "atan"):
        f = atan_iv(arg) * _PI.reciprocal() + 0.5
    else:
        raise NotImplementedError(
            f"no certified CDF enclosure for activation {name!r}; "
            f"supported bases: {SUPPORTED_CDFS}"
        )
    return Interval(max(f.lo, 0.0), min(f.hi, 1.0))


def band_mass_enclosure(
    name: str, a: float, b: float, *, loc: float = 0.0, scale: float = 1.0
) -> Interval:
    """Rigorous enclosure of the band probability ``F(b) - F(a)`` in ``[0, 1]``.

    Requires ``a <= b`` (a band mass is non-negative); reversed limits raise
    :class:`ValueError`.
    """
    if a > b:
        raise ValueError(f"require a <= b for a band mass, got a={a}, b={b}")
    fa = cdf_enclosure(name, a, loc=loc, scale=scale)
    fb = cdf_enclosure(name, b, loc=loc, scale=scale)
    mass = fb - fa
    return Interval(max(mass.lo, 0.0), min(mass.hi, 1.0))


def empirical_cdf(samples: Sequence[float], x: float) -> float:
    """Right-continuous empirical CDF ``F_n(x) = (#{s <= x}) / n``."""
    n = len(samples)
    if n == 0:
        raise ValueError("empirical_cdf needs at least one sample")
    return sum(1 for s in samples if s <= x) / n


@dataclass(frozen=True)
class CertifiedGoFResult:
    """Outcome of a certified DKW goodness-of-fit test.

    Attributes
    ----------
    rejected
        ``True`` iff :attr:`certified_ks_lower_bound` exceeds :attr:`epsilon`
        (a sound level-``alpha`` rejection).
    alpha
        Significance level used for the DKW envelope.
    epsilon
        The (outward-rounded) DKW deviation ``eps_n(alpha)``.
    certified_ks_lower_bound
        A guaranteed lower bound on ``sup_x |F_n(x) - F(x)|`` from the model
        CDF enclosure.
    n
        Number of samples.
    backend
        Transcendental enclosure backend (``"mpmath"`` or ``"libm_fallback"``).
    worst_x
        Sample point attaining the certified lower bound.
    """

    rejected: bool
    alpha: float
    epsilon: float
    certified_ks_lower_bound: float
    n: int
    backend: str
    worst_x: float


def certified_gof(
    samples: Sequence[float],
    name: str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
    alpha: float = 0.05,
) -> CertifiedGoFResult:
    r"""Certified DKW goodness-of-fit test for a location-scale model CDF.

    The Kolmogorov-Smirnov statistic ``sup_x |F_n(x) - F(x)|`` is bounded *below*
    by the worst distance from an empirical value ``F_n(x)`` (and its left limit)
    to the rigorous model enclosure ``[F_lo(x), F_hi(x)]`` at the sample points.
    Because the enclosure is outward-rounded and ``eps`` is rounded up,
    ``rejected = (lower_bound > eps)`` is a *sound* level-``alpha`` test: a
    rejection proves the model's true CDF is inconsistent with the data (the DKW
    bad event has probability ``<= alpha``). Non-rejection is *not* a proof of
    fit -- DKW is one-sided.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("certified_gof needs at least one sample")
    eps = dkw_epsilon(n, alpha)
    xs = sorted(float(s) for s in samples)
    ks_lb = 0.0
    worst = xs[0]
    for x in xs:
        f = cdf_enclosure(name, x, loc=loc, scale=scale)
        f_right = bisect.bisect_right(xs, x) / n  # F_n(x)
        f_left = bisect.bisect_left(xs, x) / n  # F_n(x^-)
        for femp in (f_left, f_right):
            # dist(femp, [f.lo, f.hi]) -- a rigorous lower bound on |femp - F(x)|.
            gap = max(0.0, f.lo - femp, femp - f.hi)
            if gap > ks_lb:
                ks_lb = gap
                worst = x
    return CertifiedGoFResult(
        rejected=ks_lb > eps,
        alpha=alpha,
        epsilon=eps,
        certified_ks_lower_bound=ks_lb,
        n=n,
        backend=backend_name(),
        worst_x=worst,
    )


__all__ = [
    "CertifiedGoFResult",
    "SUPPORTED_CDFS",
    "band_mass_enclosure",
    "cdf_enclosure",
    "certified_gof",
    "empirical_cdf",
]
