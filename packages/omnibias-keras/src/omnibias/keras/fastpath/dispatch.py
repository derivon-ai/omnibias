# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward-pass helpers shared by the OMBU and OperatorBlock layers,
written against ``keras.ops`` so they run on any Keras backend.

- :func:`multibias_literal_forward`: ``sum_k s_k * sigma(z + b_k)``.
- :func:`multibias_collapsed_forward`: closed-form ``sigma^(order)(z + bias_mean)``.
- :func:`multibias_integral_forward`: sign-weighted antiderivative sum.
- :func:`multibias_integral_window_forward`: ordered bias-window integral.

Closed-form fast paths avoid the finite-difference quotient entirely, so
autograd through them (on whichever Keras backend is active) is stable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from keras import ops

if TYPE_CHECKING:
    from omnibias.core.spec import ActivationSpec


def _as_2d_signs(signs: Any, num_channels: int, K: int) -> Any:
    """Broadcast ``(K,)`` signs up to ``(num_channels, K)``."""
    shape = tuple(signs.shape)
    if len(shape) == 1:
        return ops.broadcast_to(ops.reshape(signs, (1, K)), (num_channels, K))
    return signs


def multibias_literal_forward(
    z: Any,
    biases: Any,
    signs: Any,
    sigma: Callable[[Any], Any],
) -> Any:
    """Literal multi-bias activation.

    ``out[..., c] = sum_k signs[c, k] * sigma(z[..., c] + biases[c, k])``.

    Parameters
    ----------
    z : tensor of shape ``(..., C)``
    biases : tensor of shape ``(C, K)``
    signs : tensor of shape ``(C, K)`` or ``(K,)``
    sigma : callable applied elementwise.
    """
    num_channels = int(biases.shape[0])
    K = int(biases.shape[1])
    signs2d = _as_2d_signs(signs, num_channels, K)
    z_expand = ops.expand_dims(z, -1)  # (..., C, 1)
    activated = sigma(z_expand + biases)  # (..., C, K)
    return ops.sum(activated * signs2d, axis=-1)


def multibias_collapsed_forward(
    z: Any,
    biases: Any,
    spec: ActivationSpec[Any],
    order: int,
) -> Any:
    """Closed-form ``sigma^(order)(z + bias_mean)``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}.")
    if spec.fastpath is None:
        raise NotImplementedError(
            f"Activation {spec.name!r} has no closed-form derivative kernel."
        )
    b_mean = ops.mean(biases, axis=-1)  # (C,)
    return spec.fastpath(z + b_mean, order)


def multibias_integral_forward(
    z: Any,
    biases: Any,
    signs: Any,
    spec: ActivationSpec[Any],
) -> Any:
    """Closed-form sign-weighted antiderivative forward."""
    if spec.integral is None:
        raise NotImplementedError(
            f"Activation {spec.name!r} has no closed-form integral kernel."
        )
    return multibias_literal_forward(z, biases, signs, spec.integral)


def multibias_integral_window_forward(
    z: Any,
    center: Any,
    width: Any,
    signs: Any,
    spec: ActivationSpec[Any],
    *,
    normalize: bool = False,
    small_width_threshold: float = 1e-4,
    use_small_width_taylor: bool = True,
) -> Any:
    """Closed-form ordered bias-window integral ``S(z + b_hi) - S(z + b_lo)``."""
    half_width = 0.5 * width
    endpoints = ops.stack((center - half_width, center + half_width), axis=-1)
    exact = multibias_integral_forward(z, endpoints, signs, spec)
    if normalize:
        exact = exact / ops.maximum(width, 1e-12)

    if not use_small_width_taylor or small_width_threshold <= 0.0:
        return exact

    midpoint = z + center
    approx = spec.forward(midpoint)
    if not normalize:
        approx = approx * width
    mask = width < small_width_threshold
    return ops.where(mask, approx, exact)


__all__ = [
    "multibias_collapsed_forward",
    "multibias_integral_forward",
    "multibias_integral_window_forward",
    "multibias_literal_forward",
]
