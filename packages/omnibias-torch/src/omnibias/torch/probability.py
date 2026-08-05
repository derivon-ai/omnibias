# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable probability / measure operators (PyTorch).

Two evaluations of the *same* band functional ``[low, high]``:

* :func:`model_band_mass` -- the closed-form CDF difference ``F(high) - F(low)``
  of a location-scale model whose base activation is a CDF
  (:func:`omnibias.core.probability.cdf_normalization`); this is the analytic
  (geometric) probability, exact and differentiable;
* :func:`empirical_band_mass` -- the data frequency in the band, either a hard
  count or a differentiable difference-of-sigmoids *soft* count.

Comparing the two gives :func:`binned_calibration_error` (model vs empirical
histogram), :func:`ks_statistic` (empirical vs model CDF), and
:func:`soft_histogram` (a differentiable bank-of-bands density). Glivenko-Cantelli
ties them together: the empirical band ratio converges to the model band mass as
``N -> inf``; the certified counterpart lives in
:mod:`omnibias.core.verified.probability`.
"""

from __future__ import annotations

from omnibias.core.probability import cdf_normalization
from omnibias.torch.activations.registry import ActivationSpec, get_activation

import torch
from torch import Tensor


def _as_tensor(x: Tensor | float, ref: Tensor | None = None) -> Tensor:
    """Promote a Python scalar to a tensor (matching ``ref``'s dtype/device)."""
    if isinstance(x, Tensor):
        return x
    if ref is not None:
        return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(x, dtype=torch.get_default_dtype())


def _resolve_cdf(
    base: str | ActivationSpec[Tensor],
) -> tuple[ActivationSpec[Tensor], float, float]:
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
    x: Tensor | float,
    *,
    base: str | ActivationSpec[Tensor] = "sigmoid",
    loc: float | Tensor = 0.0,
    scale: float | Tensor = 1.0,
) -> Tensor:
    """Location-scale CDF ``F(x) = scale_n * sigma((x - loc)/scale) + shift_n``."""
    spec, scale_n, shift_n = _resolve_cdf(base)
    z = (_as_tensor(x) - loc) / scale
    out: Tensor = scale_n * spec.forward(z) + shift_n
    return out


def model_band_mass(
    low: Tensor | float,
    high: Tensor | float,
    *,
    base: str | ActivationSpec[Tensor] = "sigmoid",
    loc: float | Tensor = 0.0,
    scale: float | Tensor = 1.0,
) -> Tensor:
    """Closed-form band probability ``F(high) - F(low)`` (the model measure)."""
    lo = cdf(low, base=base, loc=loc, scale=scale)
    hi = cdf(high, base=base, loc=loc, scale=scale)
    out: Tensor = hi - lo
    return out


def _soft_ecdf(samples: Tensor, e: Tensor, temperature: float) -> Tensor:
    r"""Smooth empirical CDF ``mean_i sigmoid((e - x_i)/tau)`` (``-> F_n`` as ``tau->0``)."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    diff = (e.unsqueeze(-1) - samples) / temperature
    out: Tensor = torch.sigmoid(diff).mean(dim=-1)
    return out


def empirical_band_mass(
    samples: Tensor,
    low: Tensor | float,
    high: Tensor | float,
    *,
    soft: bool = True,
    temperature: float = 0.1,
) -> Tensor:
    """Fraction of ``samples`` in ``[low, high]``.

    ``soft=True`` returns the differentiable difference-of-sigmoids count (the
    same band primitive applied to the data); ``soft=False`` returns the exact
    (non-differentiable) count.
    """
    lo = _as_tensor(low, ref=samples)
    hi = _as_tensor(high, ref=samples)
    if soft:
        out: Tensor = _soft_ecdf(samples, hi, temperature) - _soft_ecdf(
            samples, lo, temperature
        )
        return out
    mask = (samples >= lo) & (samples <= hi)
    hard: Tensor = mask.to(samples.dtype).mean()
    return hard


def _hard_ecdf(samples: Tensor, edges: Tensor) -> Tensor:
    """Right-continuous empirical CDF ``F_n(edge_k)`` evaluated at each edge."""
    counts: Tensor = (samples.unsqueeze(-1) <= edges).to(samples.dtype).mean(dim=0)
    return counts


def binned_calibration_error(
    samples: Tensor,
    edges: Tensor,
    *,
    base: str | ActivationSpec[Tensor] = "sigmoid",
    loc: float | Tensor = 0.0,
    scale: float | Tensor = 1.0,
    soft: bool = True,
    temperature: float = 0.1,
) -> Tensor:
    r"""L1 distance ``sum_k |p_model(B_k) - p_emp(B_k)|`` over the bins from ``edges``.

    A distributional calibration error: how far the model's bin probabilities are
    from the data's bin frequencies. Differentiable when ``soft=True`` (so it can
    be used as a calibration-aware training penalty).
    """
    edges = _as_tensor(edges, ref=samples)
    if edges.ndim != 1 or edges.numel() < 2:
        raise ValueError("edges must be a 1-D tensor of length >= 2")
    model_cdf = cdf(edges, base=base, loc=loc, scale=scale)
    model_bins = model_cdf[1:] - model_cdf[:-1]
    emp_cdf = _soft_ecdf(samples, edges, temperature) if soft else _hard_ecdf(
        samples, edges
    )
    emp_bins = emp_cdf[1:] - emp_cdf[:-1]
    out: Tensor = (model_bins - emp_bins).abs().sum()
    return out


def ks_statistic(
    samples: Tensor,
    *,
    base: str | ActivationSpec[Tensor] = "sigmoid",
    loc: float | Tensor = 0.0,
    scale: float | Tensor = 1.0,
) -> Tensor:
    """Kolmogorov-Smirnov statistic ``sup_x |F_n(x) - F_model(x)|`` at the samples."""
    n = samples.numel()
    if n == 0:
        raise ValueError("ks_statistic needs at least one sample")
    xs, _ = torch.sort(samples)
    fm = cdf(xs, base=base, loc=loc, scale=scale)
    idx = torch.arange(1, n + 1, dtype=fm.dtype, device=fm.device)
    f_right = idx / n
    f_left = (idx - 1.0) / n
    gaps = torch.maximum((f_right - fm).abs(), (f_left - fm).abs())
    out: Tensor = gaps.max()
    return out


def soft_histogram(
    samples: Tensor,
    edges: Tensor,
    *,
    temperature: float = 0.1,
    normalize: bool = True,
) -> Tensor:
    """Differentiable per-bin masses (a bank of soft bands) over ``edges``.

    With ``normalize=True`` the masses sum to 1 (a density over the binned axis).
    """
    edges = _as_tensor(edges, ref=samples)
    if edges.ndim != 1 or edges.numel() < 2:
        raise ValueError("edges must be a 1-D tensor of length >= 2")
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
