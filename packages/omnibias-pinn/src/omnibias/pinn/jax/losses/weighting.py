# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Loss-term statistics and pointwise self-adaptive weights (jax twin).

Twin of :mod:`omnibias.pinn.torch.losses.weighting`; see that module for the
exposition. The EMA / cadence state machine itself is shared pure Python
(:mod:`omnibias.pinn._core.weighting`), so only the measurement and the tensor
primitive are written twice.

As everywhere in the JAX backend the measurement helpers take a
``loss_fn(params) -> scalar`` callable plus a pytree of parameters rather than a
list of tensors, matching
:func:`~omnibias.pinn.jax.losses.estimate_ntk_trace`, and the trainable weights
are a frozen dataclass pytree rather than an ``nn.Module``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.pinn._core.weighting import GradStats


def grad_stats(
    loss_fns: Mapping[str, Callable[[Any], Array]],
    params: Any,
) -> dict[str, GradStats]:
    r"""Per-term ``max`` and ``mean`` of ``|dL_k / dtheta|``.

    The measurement half of
    :class:`~omnibias.pinn._core.weighting.GradNormWeighter`. Both statistics
    run over every leaf of ``params``; JAX returns an explicit zero gradient for
    a parameter a term does not touch, so the "unused parameters count as zero"
    rule the torch twin has to arrange by hand holds here automatically.
    """
    if not loss_fns:
        raise ValueError("grad_stats: empty loss_fns mapping")
    out: dict[str, GradStats] = {}
    for name, fn in loss_fns.items():
        grads = jax.tree_util.tree_leaves(jax.grad(fn)(params))
        if not grads:
            raise ValueError(f"grad_stats: no parameter leaves for term {name!r}")
        total = 0.0
        count = 0
        peak = 0.0
        for g in grads:
            a = jnp.abs(g)
            count += int(a.size)
            total += float(jnp.sum(a))
            peak = max(peak, float(jnp.max(a)))
        out[name] = GradStats(max_abs=peak, mean_abs=total / count)
    return out


def ntk_trace_stats(
    loss_fns: Mapping[str, Callable[[Any], Array]],
    params: Any,
) -> dict[str, float]:
    """Per-term NTK trace proxy ``sum_theta (dL_k / dtheta)^2``, as floats.

    The measurement half of :class:`~omnibias.pinn._core.weighting.NTKWeighter`;
    the same proxy as :func:`~omnibias.pinn.jax.losses.estimate_ntk_trace`,
    computed for every term in one sweep.
    """
    if not loss_fns:
        raise ValueError("ntk_trace_stats: empty loss_fns mapping")
    out: dict[str, float] = {}
    for name, fn in loss_fns.items():
        grads = jax.tree_util.tree_leaves(jax.grad(fn)(params))
        total = 0.0
        for g in grads:
            total += float(jnp.sum(g * g))
        out[name] = total
    return out


def reverse_gradient(x: Array) -> Array:
    """Identity forward, negated gradient backward.

    ``2 x_stopped - x`` is exactly ``x`` in IEEE-754 and differentiates to
    ``-1``, so an ascent objective can be driven by an ordinary descent
    optimiser. See the torch twin for why this is exact rather than approximate.
    """
    return 2.0 * jax.lax.stop_gradient(x) - x


_MASK_SHORTCUTS = {"identity", "square"}


def _mask_value(lam: Array, mask: str | JaxActivationSpec) -> Array:
    """Apply a soft-attention mask ``m`` to the raw weights."""
    if isinstance(mask, str):
        if mask == "identity":
            return lam
        if mask == "square":
            return lam * lam
        try:
            spec = get_activation(mask)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"unknown mask {mask!r}: expected an ActivationSpec, one of "
                f"{sorted(_MASK_SHORTCUTS)}, or an omnibias activation name"
            ) from exc
    else:
        spec = mask
    return spec.forward(lam)


def self_adaptive_loss(
    residual: Array,
    lambdas: Array,
    *,
    mask: str | JaxActivationSpec = "sigmoid",
    ascent: bool = True,
) -> Array:
    r"""Soft-attention self-adaptive residual loss (McClenny & Braga-Neto 2020).

    ``L = mean_k m(lambda_k) r_k^2``, minimised over the network parameters and
    maximised over ``lambdas``; with ``ascent=True`` the mask's gradient is
    reversed so one ordinary descent update does both. See the torch twin for
    the full discussion.
    """
    try:
        broadcast = jnp.broadcast_shapes(residual.shape, lambdas.shape)
    except ValueError:
        broadcast = None
    if broadcast != residual.shape:
        raise ValueError(
            f"lambdas of shape {lambdas.shape} do not broadcast to "
            f"residual of shape {residual.shape}"
        )
    w = _mask_value(lambdas, mask)
    if ascent:
        w = reverse_gradient(w)
    return jnp.mean(w * residual**2)


@dataclass(frozen=True)
class SelfAdaptiveWeights:
    """Trainable pointwise weights for :func:`self_adaptive_loss` (jax pytree).

    Twin of :class:`omnibias.pinn.torch.losses.SelfAdaptiveWeights`. ``raw`` is
    the only leaf, so the weights ride along in the parameter pytree and
    ``jax.grad`` reaches them with no extra wiring. Build one with
    :func:`make_self_adaptive_weights`.
    """

    raw: Array
    mask: str | JaxActivationSpec = "sigmoid"
    ascent: bool = True

    def attention(self) -> Array:
        """The current masked weights ``m(lambda)``, for diagnostics."""
        return jax.lax.stop_gradient(_mask_value(self.raw, self.mask))

    def loss(self, residual: Array) -> Array:
        """The self-adaptive loss of a residual with matching leading length."""
        if residual.shape[0] != self.raw.shape[0]:
            raise ValueError(
                f"residual has {residual.shape[0]} points but this object holds "
                f"{self.raw.shape[0]} weights"
            )
        lam = self.raw.reshape((-1,) + (1,) * (residual.ndim - 1))
        return self_adaptive_loss(residual, lam, mask=self.mask, ascent=self.ascent)

    def __repr__(self) -> str:
        return (
            f"SelfAdaptiveWeights(n_points={self.raw.shape[0]}, "
            f"mask={self.mask!r}, ascent={self.ascent})"
        )


def _saw_flatten(
    w: SelfAdaptiveWeights,
) -> tuple[tuple[Array], tuple[str | JaxActivationSpec, bool]]:
    return (w.raw,), (w.mask, w.ascent)


def _saw_unflatten(
    aux: tuple[str | JaxActivationSpec, bool], leaves: tuple[Array]
) -> SelfAdaptiveWeights:
    obj = SelfAdaptiveWeights.__new__(SelfAdaptiveWeights)
    object.__setattr__(obj, "raw", leaves[0])
    object.__setattr__(obj, "mask", aux[0])
    object.__setattr__(obj, "ascent", aux[1])
    return obj


jax.tree_util.register_pytree_node(
    SelfAdaptiveWeights, _saw_flatten, _saw_unflatten
)


def make_self_adaptive_weights(
    n_points: int,
    *,
    mask: str | JaxActivationSpec = "sigmoid",
    init: float = 0.0,
    ascent: bool = True,
    dtype: Any = jnp.float64,
) -> SelfAdaptiveWeights:
    """Initialise uniform :class:`SelfAdaptiveWeights` over ``n_points`` points."""
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    raw = jnp.full((n_points,), float(init), dtype=dtype)
    return SelfAdaptiveWeights(raw=raw, mask=mask, ascent=ascent)


__all__ = [
    "SelfAdaptiveWeights",
    "grad_stats",
    "make_self_adaptive_weights",
    "ntk_trace_stats",
    "reverse_gradient",
    "self_adaptive_loss",
]
