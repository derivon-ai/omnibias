# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Vectorized soft-binning band / integral embedding (theory 05-03).

Batched lift of the :func:`omnibias.core.ftc.ftc_block` /
``OperatorBlock(op="band"|"integral")`` closed-form window over a *grid* of
``J`` interior thresholds ``t_1 < ... < t_J``, padded with ``t_0 = -inf`` and
``t_{J+1} = +inf``, giving ``J + 1`` soft bins for one numeric feature:

.. math::

    \mathrm{band}(x, t)_j = \sigma(\beta (x - t_j)) - \sigma(\beta (x - t_{j+1})), \quad j = 0..J

Each pairwise difference is exactly the ``OperatorBlock(op="band")`` window
``sigma(z + b_hi) - sigma(z + b_lo)`` with ``z = beta * x``, ``b_hi = -beta
t_j``, ``b_lo = -beta t_{j+1}`` -- this module reuses that formula rather
than forking it, batched over the threshold grid instead of a single learned
window (spec section 4(e)). The two open bins are finite, closed-form limits
of that same substitution (``sigma(+-inf) = 1, 0``), not literal infinities
in the implementation.

Two identities hold **exactly** (not just as ``beta -> inf``), verified as
unit tests:

- **Telescoping sum**: ``sum_j band_embed(x, t)_j == 1`` for every finite
  ``beta > 0`` -- a genuine soft partition of unity, not merely a large-beta
  approximation.
- **FTC**: ``d/dx integral_embed(x, t) == beta * band_embed(x, t)``
  elementwise, since :func:`integral_embed` is `band_embed`'s closed-form
  antiderivative (``S' = sigma``, ``S`` = softplus), with an integration
  constant per open tail chosen so every bin stays finite for finite ``x``
  (see the module-level derivation in the theory spec, section 4(e)/5(iii)).

``beta -> inf`` hardens each soft bin into a crisp 0/1 indicator of
``t_j <= x < t_{j+1}`` -- **temperature collapse** (feasibility sense), not
the founding ``delta -> 0`` bias collapse. No founding collapse limit
appears in this module.
"""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _softplus(z: FloatArray) -> FloatArray:
    """Numerically stable ``log(1 + exp(z))``, vectorized (``ftc.softplus`` twin)."""
    return cast(FloatArray, np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))))


def _sigmoid(z: FloatArray) -> FloatArray:
    """Numerically stable logistic sigmoid, vectorized (``ftc.sigmoid`` twin)."""
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _prepare(x: ArrayLike, t: ArrayLike) -> tuple[FloatArray, FloatArray]:
    xv = np.asarray(x, dtype=np.float64)
    tv = np.asarray(t, dtype=np.float64)
    if tv.ndim < 1 or tv.shape[-1] < 1:
        raise ValueError(f"t needs at least 1 interior threshold along its last axis, got shape {tv.shape}")
    if not np.all(np.diff(tv, axis=-1) > 0.0):
        raise ValueError("t must be strictly increasing along its last axis (t_1 < ... < t_J)")
    return xv, tv


def band_embed(x: ArrayLike, t: ArrayLike, *, beta: float = 1.0) -> FloatArray:
    r"""Open-tail soft-histogram band embedding over a threshold grid.

    ``band_embed(x, t)[..., j] = sigma(beta*(x - t_j)) - sigma(beta*(x - t_{j+1}))``,
    ``j = 0..J``, with ``t_0 := -inf`` and ``t_{J+1} := +inf`` folded in as the
    exact limits ``sigma(+inf) = 1`` / ``sigma(-inf) = 0`` (never computed as
    literal infinities).

    Parameters
    ----------
    x : array_like
        Feature values, any shape.
    t : array_like
        Sorted interior thresholds, shape ``(..., J)`` with ``J >= 1`` and
        strictly increasing along the last axis (checked). A shared
        ``(J,)`` grid broadcasts against every entry of ``x``; a
        ``(n_features, J)`` grid broadcasts against ``x`` of shape
        ``(n, n_features)`` for a per-feature grid.
    beta : float, default 1.0
        Sharpness. ``beta -> inf`` hardens each bin into a crisp indicator of
        ``t_j <= x < t_{j+1}`` (temperature collapse, not founding bias
        collapse).

    Returns
    -------
    FloatArray
        Shape ``x.shape + (J+1,)`` (broadcast). Each entry lies in ``(0, 1)``
        and **every row sums to exactly 1** (telescoping; see module
        docstring), for any ``beta > 0`` -- a genuine soft partition of
        unity, unlike a finite-window-only band.
    """
    xv, tv = _prepare(x, t)
    b = float(beta)
    z = b * (xv[..., np.newaxis] - tv)
    sig = _sigmoid(z)
    lead = np.broadcast_shapes(xv.shape, tv.shape[:-1])
    ones = np.ones(lead + (1,), dtype=np.float64)
    zeros = np.zeros(lead + (1,), dtype=np.float64)
    sig = np.broadcast_to(sig, lead + (tv.shape[-1],))
    padded = np.concatenate([ones, sig, zeros], axis=-1)
    return cast(FloatArray, padded[..., :-1] - padded[..., 1:])


def integral_embed(x: ArrayLike, t: ArrayLike, *, beta: float = 1.0) -> FloatArray:
    r"""Closed-form antiderivative twin of :func:`band_embed`.

    ``d/dx integral_embed(x, t) == beta * band_embed(x, t)`` exactly
    (a unit test, not an approximation): each of the ``J-1`` interior bins is
    the usual finite-window antiderivative ``S(beta*(x-t_j)) -
    S(beta*(x-t_{j+1}))`` (``S`` = softplus, ``S' = sigma``); the two open
    bins use the integration constant that keeps them finite for finite
    ``x`` -- ``beta*(x-t_1) - S(beta*(x-t_1))`` on the left (matching
    ``S(-z) = S(z) - z``) and plain ``S(beta*(x-t_J))`` on the right, which
    is exactly the ``t_0 -> -inf`` / ``t_{J+1} -> +inf`` limit of the finite
    window formula, taken without ever evaluating ``S`` at infinity.

    A second exact identity (telescoping, like :func:`band_embed`'s):
    ``sum_j integral_embed(x, t)[..., j] == beta * (x - t_1)``.

    Parameters and shapes match :func:`band_embed`.
    """
    xv, tv = _prepare(x, t)
    b = float(beta)
    z = b * (xv[..., np.newaxis] - tv)
    sp = _softplus(z)
    lead = np.broadcast_shapes(xv.shape, tv.shape[:-1])
    sp = np.broadcast_to(sp, lead + (tv.shape[-1],))
    t0 = np.broadcast_to(tv[..., 0], lead)
    bin0 = (b * (xv - t0) - sp[..., 0])[..., np.newaxis]
    zeros = np.zeros(lead + (1,), dtype=np.float64)
    padded = np.concatenate([sp, zeros], axis=-1)
    tail = padded[..., :-1] - padded[..., 1:]
    return cast(FloatArray, np.concatenate([bin0, tail], axis=-1))


def quantile_thresholds(x_ref: ArrayLike, n_bins: int) -> FloatArray:
    """Quantile-spaced interior thresholds from a reference sample.

    Returns ``n_bins - 1`` thresholds, the empirical quantiles of ``x_ref``
    at ``1/n_bins, 2/n_bins, ..., (n_bins-1)/n_bins``, for the "quantile
    init" convention used by ``BandFeatureEmbedder``
    (:mod:`omnibias.tab.torch.embed`) -- feeding these into :func:`band_embed`
    gives ``n_bins`` open-tail soft bins each holding (at ``beta -> inf``)
    about ``1/n_bins`` of the reference mass. Degenerate quantile ties (e.g.
    from a constant or heavily-duplicated reference column) are nudged
    strictly increasing so every bin stays non-empty.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2 (>= 1 interior threshold), got {n_bins}")
    xv = np.asarray(x_ref, dtype=np.float64).reshape(-1)
    if xv.size < 2:
        raise ValueError("quantile_thresholds needs at least 2 reference points")
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(xv, qs)
    eps = 1e-6 * (float(np.max(np.abs(xv))) + 1.0)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return cast(FloatArray, edges)


__all__ = [
    "FloatArray",
    "band_embed",
    "integral_embed",
    "quantile_thresholds",
]
