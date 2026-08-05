# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable probability / measure operators (JAX).

Bit-identical twin of :mod:`omnibias.torch.probability`. Two evaluations of the
same band functional ``[low, high]``:

* :func:`model_band_mass` -- the closed-form CDF difference ``F(high) - F(low)``
  of a location-scale model whose base activation is a CDF
  (:func:`omnibias.core.probability.cdf_normalization`);
* :func:`empirical_band_mass` -- the data frequency in the band, hard count or a
  differentiable difference-of-sigmoids soft count.

Comparison utilities: :func:`binned_calibration_error`, :func:`ks_statistic`,
:func:`soft_histogram`. The rigorous (interval) counterpart lives in
:mod:`omnibias.core.verified.probability`.
"""

from __future__ import annotations

from omnibias.core.probability import cdf_normalization
from omnibias.jax.activations import JaxActivationSpec, get_activation

import jax.numpy as jnp
from jax import Array
from jax import nn as jnn


def _as_array(x: Array | float, ref: Array | None = None) -> Array:
    """Promote a Python scalar to an array (matching ``ref``'s dtype)."""
    if ref is not None:
        return jnp.asarray(x, dtype=ref.dtype)
    return jnp.asarray(x)


def _resolve_cdf(
    base: str | JaxActivationSpec,
) -> tuple[JaxActivationSpec, float, float]:
    """Resolve ``base`` to ``(spec, scale, shift)`` with ``scale*sigma+shift`` a CDF."""
    spec = get_activation(base)
    norm = cdf_normalization(spec)
    if norm is None:
        raise ValueError(
            f"activation {spec.name!r} is not a CDF (saturations "
            f"{spec.limit_neg_inf} -> {spec.limit_pos_inf}); a probability model "
            "needs a finite, increasing activation (e.g. sigmoid, tanh, arctan)"
        )
    scale_n, shift_n = norm
    return spec, scale_n, shift_n


def cdf(
    x: Array | float,
    *,
    base: str | JaxActivationSpec = "sigmoid",
    loc: float | Array = 0.0,
    scale: float | Array = 1.0,
) -> Array:
    """Location-scale CDF ``F(x) = scale_n * sigma((x - loc)/scale) + shift_n``."""
    spec, scale_n, shift_n = _resolve_cdf(base)
    z = (_as_array(x) - loc) / scale
    out: Array = scale_n * spec.forward(z) + shift_n
    return out


def model_band_mass(
    low: Array | float,
    high: Array | float,
    *,
    base: str | JaxActivationSpec = "sigmoid",
    loc: float | Array = 0.0,
    scale: float | Array = 1.0,
) -> Array:
    """Closed-form band probability ``F(high) - F(low)`` (the model measure)."""
    lo = cdf(low, base=base, loc=loc, scale=scale)
    hi = cdf(high, base=base, loc=loc, scale=scale)
    out: Array = hi - lo
    return out


def _soft_ecdf(samples: Array, e: Array, temperature: float) -> Array:
    r"""Smooth empirical CDF ``mean_i sigmoid((e - x_i)/tau)`` (``-> F_n`` as ``tau->0``)."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    diff = (jnp.expand_dims(e, -1) - samples) / temperature
    out: Array = jnn.sigmoid(diff).mean(axis=-1)
    return out


def empirical_band_mass(
    samples: Array,
    low: Array | float,
    high: Array | float,
    *,
    soft: bool = True,
    temperature: float = 0.1,
) -> Array:
    """Fraction of ``samples`` in ``[low, high]`` (soft = differentiable)."""
    lo = _as_array(low, ref=samples)
    hi = _as_array(high, ref=samples)
    if soft:
        out: Array = _soft_ecdf(samples, hi, temperature) - _soft_ecdf(
            samples, lo, temperature
        )
        return out
    mask = (samples >= lo) & (samples <= hi)
    hard: Array = mask.astype(samples.dtype).mean()
    return hard


def _hard_ecdf(samples: Array, edges: Array) -> Array:
    """Right-continuous empirical CDF ``F_n(edge_k)`` evaluated at each edge."""
    counts: Array = (jnp.expand_dims(samples, -1) <= edges).astype(samples.dtype).mean(
        axis=0
    )
    return counts


def binned_calibration_error(
    samples: Array,
    edges: Array,
    *,
    base: str | JaxActivationSpec = "sigmoid",
    loc: float | Array = 0.0,
    scale: float | Array = 1.0,
    soft: bool = True,
    temperature: float = 0.1,
) -> Array:
    r"""L1 distance ``sum_k |p_model(B_k) - p_emp(B_k)|`` over the bins from ``edges``."""
    edges = _as_array(edges, ref=samples)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a 1-D array of length >= 2")
    model_cdf = cdf(edges, base=base, loc=loc, scale=scale)
    model_bins = model_cdf[1:] - model_cdf[:-1]
    emp_cdf = (
        _soft_ecdf(samples, edges, temperature)
        if soft
        else _hard_ecdf(samples, edges)
    )
    emp_bins = emp_cdf[1:] - emp_cdf[:-1]
    out: Array = jnp.abs(model_bins - emp_bins).sum()
    return out


def ks_statistic(
    samples: Array,
    *,
    base: str | JaxActivationSpec = "sigmoid",
    loc: float | Array = 0.0,
    scale: float | Array = 1.0,
) -> Array:
    """Kolmogorov-Smirnov statistic ``sup_x |F_n(x) - F_model(x)|`` at the samples."""
    n = samples.size
    if n == 0:
        raise ValueError("ks_statistic needs at least one sample")
    xs = jnp.sort(samples)
    fm = cdf(xs, base=base, loc=loc, scale=scale)
    idx = jnp.arange(1, n + 1, dtype=fm.dtype)
    f_right = idx / n
    f_left = (idx - 1.0) / n
    gaps = jnp.maximum(jnp.abs(f_right - fm), jnp.abs(f_left - fm))
    out: Array = gaps.max()
    return out


def soft_histogram(
    samples: Array,
    edges: Array,
    *,
    temperature: float = 0.1,
    normalize: bool = True,
) -> Array:
    """Differentiable per-bin masses (a bank of soft bands) over ``edges``."""
    edges = _as_array(edges, ref=samples)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a 1-D array of length >= 2")
    emp_cdf = _soft_ecdf(samples, edges, temperature)
    bins = emp_cdf[1:] - emp_cdf[:-1]
    if normalize:
        bins = bins / bins.sum()
    return bins


__all__ = [
    "binned_calibration_error",
    "cdf",
    "empirical_band_mass",
    "ks_statistic",
    "model_band_mass",
    "soft_histogram",
]
