# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Probability / measure metadata for activations: CDF detection + DKW bound.

An activation ``sigma`` with finite, *increasing* saturations is -- after an
affine rescale -- a cumulative distribution function: ``F = scale*sigma + shift``
maps ``R -> [0, 1]`` monotonically, so the two-bias *band* difference
``F(z + b_hi) - F(z + b_lo)`` is the probability mass of the slab between the two
parallel hyperplanes: a geometric (measure) probability that *self-normalises*
because ``F(+inf) - F(-inf) = 1``.

This module is the pure-Python metadata that turns the ``band`` operator into a
probability:

* :func:`cdf_normalization` / :func:`is_cdf_activation` -- which activations
  qualify and the affine map that rescales them to a CDF;
* :func:`dkw_epsilon` -- the Dvoretzky-Kiefer-Wolfowitz-Massart uniform deviation
  used to wrap an empirical CDF in a distribution-free confidence band.

No tensor backend is imported, matching the ``omnibias.core`` zero-dependency
contract; the differentiable operators live in ``omnibias.{torch,jax}.probability``
and the rigorous (interval) enclosures in ``omnibias.core.verified.probability``.
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.core.spec import ActivationSpec


def cdf_normalization(spec: ActivationSpec[Any]) -> tuple[float, float] | None:
    r"""Return ``(scale, shift)`` so ``scale*sigma + shift`` is a CDF, else ``None``.

    Requires both saturation limits to be finite and strictly *increasing*
    (``limit_neg_inf < limit_pos_inf``); then

    .. math::  scale = \frac{1}{L_+ - L_-}, \qquad shift = \frac{-L_-}{L_+ - L_-}

    rescale ``sigma`` to map ``R -> [0, 1]`` monotonically. Examples:
    ``sigmoid`` (``0 -> 1``) gives ``(1, 0)``; ``tanh`` (``-1 -> 1``) gives
    ``(0.5, 0.5)``; ``arctan`` (``-pi/2 -> pi/2``) gives ``(1/pi, 0.5)``.

    Returns ``None`` for activations with no finite limit (``exp``), a vanishing
    swing (``gaussian``: ``0 -> 0``, a *density* not a CDF), or decreasing
    saturations -- those are not affine-reducible to an increasing CDF.
    """
    lo = spec.limit_neg_inf
    hi = spec.limit_pos_inf
    if lo is None or hi is None:
        return None
    span = hi - lo
    if span <= 0.0:
        return None
    scale = 1.0 / span
    shift = -lo / span
    return (scale, shift)


def is_cdf_activation(spec: ActivationSpec[Any]) -> bool:
    """``True`` when ``spec`` is affine-reducible to a CDF.

    Thin predicate over :func:`cdf_normalization` (``None`` -> not a CDF).
    """
    return cdf_normalization(spec) is not None


def dkw_epsilon(n: int, alpha: float = 0.05) -> float:
    r"""Dvoretzky-Kiefer-Wolfowitz-Massart uniform deviation ``eps_n(alpha)``.

    With probability at least ``1 - alpha`` the empirical CDF ``F_n`` of ``n``
    i.i.d. draws from a distribution with CDF ``F`` satisfies

    .. math::  \sup_x |F_n(x) - F(x)| \le eps, \qquad
               eps = \sqrt{\frac{\ln(2/alpha)}{2 n}}

    (the Massart tight constant). The value is rounded *outward* (one ulp up) so
    a test of the form "reject when the certified gap exceeds ``eps``" remains a
    sound level-``alpha`` test despite floating-point rounding -- a slightly
    larger threshold only makes the test more conservative, never less.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    eps = math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    return math.nextafter(eps, math.inf)


__all__ = [
    "cdf_normalization",
    "dkw_epsilon",
    "is_cdf_activation",
]
