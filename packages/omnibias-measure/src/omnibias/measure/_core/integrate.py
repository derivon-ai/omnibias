# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Measure-integral primitives (pure Python + numpy reference).

Every function here has a bit-identical differentiable twin in
``omnibias.measure.torch`` / ``omnibias.measure.jax``; this module is the numpy
reference the parity tests check against.

* :func:`lebesgue_integral` -- the measure integral ``int f dmu`` as the
  weight-contraction ``sum_i w_i f(x_i)``. (For a probability measure this is an
  expectation; for the Lebesgue box measure it is ``int_box f dx``.)
* :func:`importance_expectation` -- (self-normalized) importance-sampling
  expectation ``E_p[f]`` from proposal samples carried by the measure.
* :func:`superlevel_measure` -- the soft superlevel-set measure
  ``mu({f > t}) ~= sum_i w_i sigmoid(beta (f(x_i) - t))``, the shared building
  block of the layer-cake and simple-function primitives.
* :func:`layer_cake_integral` -- ``int f dmu = int_0^inf mu({f>t}) dt`` (the
  distribution-function / layer-cake formula), with a soft superlevel indicator
  so the estimate is differentiable through both ``f`` and the measure weights.
* :func:`simple_function_approx` -- the monotone from-below simple-function
  approximation ``int f dmu ~= sum_k level_k * mu(band_k)`` (the classical
  measure-theoretic construction), with soft band membership.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from omnibias.measure._core.measure import Measure

# ``numpy.trapz`` was renamed to ``numpy.trapezoid`` in numpy 2.0 (and dropped in
# 2.x). Resolve by name so it is not a static attribute access (keeps the module
# type-clean across numpy versions without a version-flaky type: ignore).
_TRAPEZOID_NAME = "trapezoid" if hasattr(np, "trapezoid") else "trapz"
_TRAPEZOID: Callable[..., Any] = getattr(np, _TRAPEZOID_NAME)

#: A numpy integrand ``nodes (n, d) -> (n,)`` (scalar) or ``-> (n, k)`` (vector).
IntegrandFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]

ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True)
class SimpleFunctionApprox(Generic[ArrayT]):
    """Result of :func:`simple_function_approx`.

    Attributes
    ----------
    integral
        The simple-function integral estimate ``sum_k level_k * mu(band_k)``.
    levels
        The (sorted, ascending) threshold levels defining the bands.
    superlevel_measures
        ``G_k = mu({f > level_k})`` (soft), one per level.
    band_masses
        ``mu(band_k) = G_k - G_{k+1}`` (with ``G_m := 0`` for the top band).
    """

    integral: ArrayT
    levels: ArrayT
    superlevel_measures: ArrayT
    band_masses: ArrayT


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable logistic sigmoid on numpy float64."""
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _trapezoid(y: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Trapezoidal rule (``numpy.trapezoid`` on numpy>=2, ``numpy.trapz`` before)."""
    return cast("NDArray[np.float64]", np.asarray(_TRAPEZOID(y, x), dtype=np.float64))


def _values(f: IntegrandFn, measure: Measure) -> NDArray[np.float64]:
    return np.asarray(f(measure.nodes), dtype=np.float64)


def lebesgue_integral(f: IntegrandFn, measure: Measure) -> NDArray[np.float64]:
    r"""Measure integral ``int f dmu = sum_i weights_i * f(nodes_i)``.

    ``f`` maps nodes ``(n, d)`` to values ``(n,)`` (scalar integrand) or
    ``(n, k)`` (vector integrand); the return is a numpy scalar or a length-``k``
    array respectively.

    This is the third sense of "integral" in omnibias (see the operator-surface
    matrix): the measure integral, distinct from the closed-form activation
    antiderivative window and from field domain quadrature.
    """
    vals = _values(f, measure)
    if vals.shape[0] != measure.n_nodes:
        raise ValueError(
            f"f must return one value per node ({measure.n_nodes}); "
            f"got leading dim {vals.shape[0]}"
        )
    return np.asarray(np.tensordot(measure.weights, vals, axes=([0], [0])), dtype=np.float64)


def importance_expectation(
    f: IntegrandFn,
    measure: Measure,
    log_weight: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    self_normalized: bool = True,
) -> NDArray[np.float64]:
    r"""Importance-sampling expectation of ``f`` under a reweighted target.

    The ``measure`` carries the proposal ``q`` (its nodes are proposal
    "samples", its weights the proposal masses). ``log_weight`` returns the log
    importance ratio ``log(p/q)`` at the nodes (needed only up to an additive
    constant when ``self_normalized=True``). The importance weights are
    ``iw_i = weights_i * exp(log_weight_i)`` and

    * self-normalized (SNIS): ``E_p[f] ~= (sum_i iw_i f_i) / (sum_i iw_i)`` --
      consistent estimator valid for an unnormalized target;
    * plain IS: ``E_p[f] ~= sum_i iw_i f_i`` -- unbiased when ``p`` and ``q`` are
      properly normalized.
    """
    x = measure.nodes
    vals = np.asarray(f(x), dtype=np.float64)
    lw = np.asarray(log_weight(x), dtype=np.float64)
    if lw.shape != (measure.n_nodes,):
        raise ValueError(
            f"log_weight must return shape ({measure.n_nodes},), got {lw.shape}"
        )
    if self_normalized:
        shift = float(np.max(lw))
        iw = measure.weights * np.exp(lw - shift)
        num = np.asarray(np.tensordot(iw, vals, axes=([0], [0])), dtype=np.float64)
        den = float(np.sum(iw))
        return num / den
    iw = measure.weights * np.exp(lw)
    return np.asarray(np.tensordot(iw, vals, axes=([0], [0])), dtype=np.float64)


def superlevel_measure(
    f: IntegrandFn,
    measure: Measure,
    levels: ArrayLike,
    *,
    beta: float = 50.0,
) -> NDArray[np.float64]:
    r"""Soft superlevel-set measure ``G_k = mu({f > level_k})``.

    Approximates the hard indicator ``1[f(x) > t]`` by the sigmoid
    ``sigmoid(beta (f(x) - t))`` (whose derivative tower is exactly the omnibias
    sigmoid tower), then contracts with the measure weights:
    ``G_k = sum_i weights_i * sigmoid(beta (f(x_i) - level_k))``. As
    ``beta -> inf`` this converges to the true superlevel measure.
    """
    fx = _values(f, measure)
    t = np.asarray(levels, dtype=np.float64).reshape(-1)
    diff = beta * (fx[None, :] - t[:, None])  # (m, n)
    g = np.sum(measure.weights[None, :] * _sigmoid(diff), axis=1)
    return cast("NDArray[np.float64]", np.asarray(g, dtype=np.float64))


def layer_cake_integral(
    f: IntegrandFn,
    measure: Measure,
    *,
    t_grid: ArrayLike | None = None,
    t_max: float | None = None,
    num_t: int = 256,
    beta: float = 50.0,
    signed: bool = True,
) -> NDArray[np.float64]:
    r"""Layer-cake (distribution-function) measure integral of a scalar ``f``.

    Uses the identity, for the positive part,
    :math:`\int f_+ \, d\mu = \int_0^\infty \mu(\{f > t\})\, dt`, and for a
    signed integrand
    :math:`\int f \, d\mu = \int_0^\infty [\mu(\{f>t\}) - \mu(\{f<-t\})]\, dt`.
    The superlevel measures are the soft :func:`superlevel_measure`, so the whole
    estimate is differentiable through ``f`` and the measure weights. The outer
    ``dt`` integral is a trapezoid over ``t_grid`` (a uniform grid on
    ``[0, t_max]`` when not given; ``t_max`` defaults to just above
    ``max |f|``).

    This is the route to differentiate through a non-smooth / thresholded
    integrand where the derivative-remainder view is awkward: it moves the
    non-smoothness into the (soft) level set.
    """
    fx = _values(f, measure)
    if fx.ndim != 1:
        raise ValueError("layer_cake_integral requires a scalar integrand f -> (n,)")
    if t_grid is None:
        if t_max is None:
            t_max = float(np.max(np.abs(fx))) * 1.05 + 1e-6
        tg = np.linspace(0.0, float(t_max), int(num_t))
    else:
        tg = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    diff_pos = beta * (fx[None, :] - tg[:, None])
    s_pos = np.sum(measure.weights[None, :] * _sigmoid(diff_pos), axis=1)
    integrand = s_pos
    if signed:
        diff_neg = beta * (-fx[None, :] - tg[:, None])
        s_neg = np.sum(measure.weights[None, :] * _sigmoid(diff_neg), axis=1)
        integrand = s_pos - s_neg
    return _trapezoid(integrand, tg)


def simple_function_approx(
    f: IntegrandFn,
    measure: Measure,
    *,
    levels: ArrayLike,
    beta: float = 50.0,
) -> SimpleFunctionApprox[NDArray[np.float64]]:
    r"""Monotone from-below simple-function approximation of ``int f dmu``.

    Realizes the textbook construction ``s = sum_k level_k * 1[band_k]`` with
    bands ``band_k = {level_k <= f < level_{k+1}}`` (soft membership from the
    sigmoid superlevel indicator) and integrates the simple function exactly:
    ``int s dmu = sum_k level_k * mu(band_k)``. Equivalently, by Abel summation,
    ``= sum_k (level_k - level_{k-1}) * mu({f > level_k})`` -- the discrete
    layer-cake. Intended for non-negative ``f`` with ``levels`` spanning
    ``[0, max f]``; for signed integrands use :func:`layer_cake_integral`.
    """
    t = np.sort(np.asarray(levels, dtype=np.float64).reshape(-1))
    if t.shape[0] < 1:
        raise ValueError("levels must be non-empty")
    g = superlevel_measure(f, measure, t, beta=beta)  # (m,) G_k
    g_next = np.concatenate([g[1:], np.zeros((1,), dtype=np.float64)])
    band_masses = g - g_next  # mu(band_k) = G_k - G_{k+1}
    prev = np.concatenate([np.zeros((1,), dtype=np.float64), t[:-1]])
    integral = np.asarray(np.sum((t - prev) * g), dtype=np.float64)
    return SimpleFunctionApprox(
        integral=integral,
        levels=t,
        superlevel_measures=g,
        band_masses=band_masses,
    )


__all__ = [
    "IntegrandFn",
    "SimpleFunctionApprox",
    "importance_expectation",
    "layer_cake_integral",
    "lebesgue_integral",
    "simple_function_approx",
    "superlevel_measure",
]
