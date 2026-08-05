# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch LIF / IF neuron primitives with closed-form surrogate gradients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from omnibias.core.spec import ActivationSpec
from omnibias.spiking._surrogates import (
    fast_sigmoid_derivative,
    gaussian_derivative,
    normalize_surrogate_kind,
)
from torch import Tensor
from torch.autograd import Function

SurrogateArg = str | ActivationSpec[Tensor]


def _torch_sigmoid(z: Tensor) -> Tensor:
    return torch.sigmoid(z)


def _torch_exp(z: Tensor) -> Tensor:
    return torch.exp(z)


def _dispatch_surrogate_derivative(u: Tensor, kind: str) -> Tensor:
    key = normalize_surrogate_kind(kind)
    if key == "fast_sigmoid":
        return fast_sigmoid_derivative(u, sigmoid=_torch_sigmoid)
    return gaussian_derivative(u, exp=_torch_exp)


def _resolve_surrogate_fn(
    surrogate: SurrogateArg,
) -> Callable[[Tensor], Tensor]:
    if isinstance(surrogate, str):
        key = normalize_surrogate_kind(surrogate)
        return lambda u: _dispatch_surrogate_derivative(u, key)
    if surrogate.derivative is not None:
        return surrogate.derivative
    if surrogate.fastpath is not None:
        return lambda u: surrogate.fastpath(u, 1)
    msg = f"ActivationSpec {surrogate.name!r} has no derivative or fastpath"
    raise ValueError(msg)


def surrogate_derivative(u: Tensor, kind: str = "fast_sigmoid") -> Tensor:
    """Closed-form surrogate derivative ``d sigma / du`` for kind ``kind``."""
    return _dispatch_surrogate_derivative(u, kind)


class _HeavisideSpikeFn(Function):
    @staticmethod
    def forward(
        ctx: Any,
        v: Tensor,
        threshold: float,
        surrogate: SurrogateArg,
        surrogate_scale: float,
    ) -> Tensor:
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.surrogate_fn = _resolve_surrogate_fn(surrogate)
        ctx.surrogate_scale = surrogate_scale
        return (v >= threshold).to(dtype=v.dtype)

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        (v,) = ctx.saved_tensors
        u = ctx.surrogate_scale * (v - ctx.threshold)
        ds_du = ctx.surrogate_fn(u)
        grad_v = grad_out * ds_du * ctx.surrogate_scale
        return grad_v, None, None, None


def heaviside_spike(
    v: Tensor,
    threshold: float = 1.0,
    *,
    surrogate: SurrogateArg = "fast_sigmoid",
    surrogate_scale: float = 4.0,
) -> Tensor:
    """Hard spike forward with a closed-form surrogate backward pass."""
    return _HeavisideSpikeFn.apply(v, threshold, surrogate, surrogate_scale)


def lif_step(
    v: Tensor,
    x: Tensor,
    *,
    decay: float = 0.9,
    threshold: float = 1.0,
    surrogate: SurrogateArg = "fast_sigmoid",
    surrogate_scale: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """One leaky integrate-and-fire step with soft reset."""
    v_pre = decay * v + x
    s = heaviside_spike(
        v_pre,
        threshold,
        surrogate=surrogate,
        surrogate_scale=surrogate_scale,
    )
    v_out = v_pre - s * threshold
    return s, v_out


def if_step(
    v: Tensor,
    x: Tensor,
    *,
    threshold: float = 1.0,
    surrogate: SurrogateArg = "fast_sigmoid",
    surrogate_scale: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """One integrate-and-fire step (``decay=1`` LIF) with soft reset."""
    return lif_step(
        v,
        x,
        decay=1.0,
        threshold=threshold,
        surrogate=surrogate,
        surrogate_scale=surrogate_scale,
    )


__all__ = [
    "heaviside_spike",
    "if_step",
    "lif_step",
    "surrogate_derivative",
]
