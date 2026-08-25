# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causal transverse filter (JAX twin; theory 05-02 G5).

Functional twin of :mod:`omnibias.torch.sequence`. ``order`` and
``width`` stay Python ints so the path is ``jit``-safe. Founding tower
taps, not temperature collapse; not :mod:`omnibias.struct`.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.sequence import leaky_integrator_init
from omnibias.jax.activations import get_activation

import jax.numpy as jnp
from jax import Array
from jax.lax import conv_general_dilated


def causal_transverse_taps(
    coeff: Array,
    alpha: Array,
    tau: Array,
    *,
    order: int,
    width: int,
    base: str = "sigmoid",
) -> Array:
    """Designed causal taps; ``order`` / ``width`` are static Python ints."""
    if int(order) < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if int(width) < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    spec = get_activation(base)
    if spec.fastpath is None:
        raise RuntimeError(f"{base!r} fastpath is required")
    lags = jnp.arange(int(width), dtype=coeff.dtype)
    if int(order) == 0:
        argument = tau - alpha * lags
    else:
        argument = alpha * (lags - tau)
    return coeff * spec.fastpath(argument, int(order))


def apply_causal_transverse_filter(x: Array, taps: Array, bias: Array) -> Array:
    """Causal FIR; ``taps[0]`` multiplies ``x_t``."""
    xx = x if x.ndim == 2 else x[None, :]
    if xx.ndim != 2:
        raise ValueError(f"x must be (batch, time) or (time,), got {xx.shape}")
    width = int(taps.shape[0])
    pred = conv_general_dilated(
        xx[:, None, :],
        taps[::-1][None, None, :],
        window_strides=(1,),
        padding=((width - 1, width - 1),),
        dimension_numbers=("NCH", "OIH", "NCH"),
    )[:, 0, : xx.shape[1]]
    pred = pred + bias
    return pred if x.ndim == 2 else pred[0]


def init_causal_transverse(
    *,
    order: int,
    alpha: float,
    width: int,
    coeff: float = 1.0,
    tau: float = 0.0,
    bias: float = 0.0,
    base: str = "sigmoid",
) -> dict[str, Any]:
    """Parameter dict matching :class:`~omnibias.torch.sequence.CausalTransverseFilter`."""
    if float(alpha) <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    return {
        "order": int(order),
        "width": int(width),
        "base": str(base),
        "coeff": jnp.asarray(coeff),
        "alpha": jnp.asarray(alpha),
        "tau": jnp.asarray(tau),
        "bias": jnp.asarray(bias),
    }


def init_leaky_integrator(rho: float, *, width: int) -> dict[str, Any]:
    """Four-parameter init near the AR(1) leaky integrator."""
    coeff, alpha, tau = leaky_integrator_init(rho)
    return init_causal_transverse(
        order=0,
        alpha=alpha,
        width=width,
        coeff=coeff,
        tau=tau,
        bias=0.0,
    )


def causal_transverse_filter(x: Array, params: dict[str, Any]) -> Array:
    taps = causal_transverse_taps(
        params["coeff"],
        params["alpha"],
        params["tau"],
        order=int(params["order"]),
        width=int(params["width"]),
        base=str(params["base"]),
    )
    return apply_causal_transverse_filter(x, taps, params["bias"])
