# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward-pass helpers used by :class:`OperatorMultiBiasUnit` and
:class:`OperatorBlock`.

Two kernels:

- :func:`multibias_literal_forward`: ``sum_k s_k * sigma(z + b_k)``,
  the literal multi-bias activation.
- :func:`multibias_collapsed_forward`: closed-form
  ``sigma^(order)(z + bias_mean)`` via the activation spec's fast-path
  kernel; one base-activation evaluation regardless of K.
- :func:`multibias_integral_forward`: closed-form sign-weighted
  antiderivative sum, whose K=2 alternating-sign form is a definite
  integral over a bias interval.

Both are plain functions (no ``nn.Module`` overhead) so they can be
called from inside training loops without parameter bookkeeping.

**Why no custom autograd Function?** The bias-gradient scrambling that
the predecessor ``multibias`` v0.2.1 release patched was a consequence
of computing ``(sigma(z + b + delta) - sigma(z + b)) / delta`` in
float32 and then asking autograd to differentiate that quotient: the
forward already lost ``log10(1/delta)`` digits, and the backward lost
the rest. The closed-form fast paths in :mod:`omnibias.fastpath` avoid
the quotient entirely -- ``sigma^(n)(z + bar b)`` is one ``sigmoid``
plus a Horner polynomial in the result -- and autograd through Horner
is bit-stable. There is therefore nothing for a custom backward to fix
in the collapsed branch; we let autograd do its job.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from omnibias.torch.activations.registry import ActivationSpec


def multibias_literal_forward(
    z: Tensor,
    biases: Tensor,
    signs: Tensor,
    sigma: Callable[[Tensor], Tensor],
) -> Tensor:
    """Literal multi-bias activation.

    Computes ``out[..., c] = sum_k signs[c, k] * sigma(z[..., c] + biases[c, k])``.

    Parameters
    ----------
    z : Tensor of shape ``(..., C)``
        Input pre-activations, channel-last.
    biases : Tensor of shape ``(C, K)``
        Per-channel bias terms.
    signs : Tensor of shape ``(C, K)`` or ``(K,)``
        Per-channel signs (or shared across channels).
    sigma : Callable
        Base activation, applied elementwise.

    Returns
    -------
    Tensor of shape ``(..., C)``.
    """
    if z.shape[-1] != biases.shape[0]:
        raise ValueError(
            f"Input last-dim ({z.shape[-1]}) must equal num_channels ({biases.shape[0]})."
        )
    if signs.dim() == 1:
        signs = signs.unsqueeze(0)  # (1, K) for broadcasting over channels
    elif signs.dim() == 2:
        if signs.shape != biases.shape:
            raise ValueError(
                f"signs shape {tuple(signs.shape)} does not match biases shape "
                f"{tuple(biases.shape)}."
            )
    else:
        raise ValueError(f"signs must be 1D or 2D, got shape {tuple(signs.shape)}.")

    # z: (..., C) -> (..., C, 1); biases: (C, K). Broadcast adds to (..., C, K).
    z_expand = z.unsqueeze(-1)
    activated = sigma(z_expand + biases)
    out = (activated * signs).sum(dim=-1)
    return out


def multibias_collapsed_forward(
    z: Tensor,
    biases: Tensor,
    spec: ActivationSpec[Tensor],
    order: int,
) -> Tensor:
    """Closed-form ``sigma^(order)(z + bias_mean)``.

    Parameters
    ----------
    z : Tensor of shape ``(..., C)``
    biases : Tensor of shape ``(C, K)``
        Per-channel biases; only their per-channel mean is used.
    spec : :class:`ActivationSpec`
        Must have a non-None ``fastpath`` kernel.
    order : int
        Derivative order ``n``; must be ``>= 0``.

    Returns
    -------
    Tensor of shape ``(..., C)``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}.")
    if spec.fastpath is None:
        raise NotImplementedError(f"Activation {spec.name!r} has no closed-form derivative kernel.")
    b_mean = biases.mean(dim=-1)  # (C,)
    out: Tensor = spec.fastpath(z + b_mean, order)
    return out


def multibias_integral_forward(
    z: Tensor,
    biases: Tensor,
    signs: Tensor,
    spec: ActivationSpec[Tensor],
) -> Tensor:
    """Closed-form sign-weighted antiderivative forward.

    Computes ``sum_k signs[c, k] * S(z[..., c] + biases[c, k])`` where
    ``S' = sigma``. For K=2 with signs ``(+1, -1)`` this is the oriented
    integral ``S(z + b_1) - S(z + b_2)`` across the bias window.
    """
    if spec.integral is None:
        raise NotImplementedError(
            f"Activation {spec.name!r} has no closed-form integral kernel; "
            "use literal_forward() instead, or pick a base with an antiderivative."
        )
    return multibias_literal_forward(z, biases, signs, spec.integral)


def multibias_integral_window_forward(
    z: Tensor,
    center: Tensor,
    width: Tensor,
    signs: Tensor,
    spec: ActivationSpec[Tensor],
    *,
    normalize: bool = False,
    small_width_threshold: float = 1e-4,
    use_small_width_taylor: bool = True,
) -> Tensor:
    """Closed-form ordered bias-window integral.

    ``center`` and positive ``width`` define endpoints
    ``b_lo = center - width / 2`` and ``b_hi = center + width / 2``. With
    fixed signs ``(-1, +1)`` this computes
    ``S(z + b_hi) - S(z + b_lo)``, i.e. the low-to-high definite integral
    of ``sigma`` across the bias interval. ``normalize=True`` returns the
    window average instead of total area.
    """
    if width.shape != center.shape:
        raise ValueError(f"width shape {tuple(width.shape)} must match center shape {tuple(center.shape)}.")
    if signs.dim() == 1:
        signs = signs.unsqueeze(0)
    if signs.shape != (center.shape[0], 2):
        raise ValueError(f"signs shape {tuple(signs.shape)} must be ({center.shape[0]}, 2).")

    half_width = 0.5 * width
    endpoints = torch.stack((center - half_width, center + half_width), dim=-1)
    exact = multibias_integral_forward(z, endpoints, signs, spec)
    if normalize:
        exact = exact / width.clamp_min(width.new_tensor(1e-12))

    if not use_small_width_taylor or small_width_threshold <= 0.0:
        return exact

    midpoint = z + center
    approx = spec.forward(midpoint)
    if not normalize:
        approx = approx * width
    mask = width < small_width_threshold
    return torch.where(mask, approx, exact)


def bias_spread(biases: Tensor) -> Tensor:
    """Per-channel ``max(b_k) - min(b_k)``; near zero in the collapse regime."""
    return biases.max(dim=-1).values - biases.min(dim=-1).values


def is_collapsed(biases: Tensor, threshold: float) -> bool:
    """True iff every channel's bias spread is strictly below ``threshold``."""
    if threshold <= 0:
        return bool((bias_spread(biases) == 0).all().item())
    return bool((bias_spread(biases) < threshold).all().item())


# Convenience re-export name for older call sites that imported the
# combined-dispatch entry point.
def multibias_forward(
    z: Tensor,
    biases: Tensor,
    signs: Tensor,
    spec: ActivationSpec[Tensor],
    *,
    collapse_threshold: float = 0.0,
    use_collapsed: bool = False,
    order: int | None = None,
) -> Tensor:
    """Combined dispatcher.

    If ``use_collapsed`` is True (and the spec has a fast-path kernel),
    computes :func:`multibias_collapsed_forward` at the requested order
    (default ``biases.shape[-1] - 1``). Otherwise computes the literal
    forward.

    .. deprecated::
        ``collapse_threshold`` never affected dispatch and is deprecated. It
        is kept only for call-site compatibility; passing a non-zero value
        now emits a :class:`DeprecationWarning`. To select the collapsed
        fast-path based on the bias spread, test :func:`is_collapsed`
        yourself and pass ``use_collapsed=True`` explicitly.
    """
    if collapse_threshold != 0.0:
        warnings.warn(
            "multibias_forward(collapse_threshold=...) is a no-op and is "
            "deprecated: it has never influenced which forward is selected. "
            "Use is_collapsed(biases, threshold) to test the collapse regime "
            "and pass use_collapsed=True to choose the collapsed fast-path.",
            DeprecationWarning,
            stacklevel=2,
        )
    if use_collapsed and spec.fastpath is not None:
        K = biases.shape[-1]
        return multibias_collapsed_forward(z, biases, spec, K - 1 if order is None else order)
    return multibias_literal_forward(z, biases, signs, spec.forward)


__all__ = [
    "bias_spread",
    "is_collapsed",
    "multibias_collapsed_forward",
    "multibias_forward",
    "multibias_integral_forward",
    "multibias_integral_window_forward",
    "multibias_literal_forward",
]
