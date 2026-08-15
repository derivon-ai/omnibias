# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-KAN JAX twin (theory 02-03). Functional, bit-identical to torch.

No GrowableOMBU on JAX: refinement is preallocated zero-weight packs.
Kolmogorov-Arnold does not justify the architecture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omnibias.core.spectral_design import BandPlan
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.jet import compose_jet
from omnibias.jax.jet_mv import compose_jet_mv, identity_jet

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class JetKANConfig:
    widths: tuple[int, ...]
    packs_per_edge: int = 3
    extra_packs: int = 2
    orders: tuple[int, ...] = (0, 1, 2)
    base: str = "tanh"

    def __post_init__(self) -> None:
        if len(self.widths) < 2:
            raise ValueError("widths must include at least input and output")
        if int(self.packs_per_edge) < 1:
            raise ValueError("packs_per_edge must be >= 1")
        if int(self.extra_packs) < 0:
            raise ValueError("extra_packs must be >= 0")


def jetkan_from_band_plan(
    plan: BandPlan,
    widths: tuple[int, ...],
    *,
    base: str = "tanh",
) -> JetKANConfig:
    return JetKANConfig(
        widths=tuple(int(w) for w in widths),
        packs_per_edge=int(plan.n_channels),
        extra_packs=2,
        orders=tuple(int(n) for n in plan.orders),
        base=base,
    )


@dataclass(frozen=True)
class JetKANParams:
    """Per-layer ``(weights, means, log_scales)`` with trailing pack axis."""

    weights: tuple[Array, ...]
    means: tuple[Array, ...]
    log_scales: tuple[Array, ...]
    active_g: tuple[int, ...]


def _capacity(config: JetKANConfig) -> int:
    return int(config.packs_per_edge) + int(config.extra_packs)


def init_jet_kan(config: JetKANConfig) -> tuple[JaxActivationSpec, JetKANParams]:
    """Zero-init extra packs; live packs get uniform weight ``1/G``."""
    act = get_activation(config.base)
    g = _capacity(config)
    live = int(config.packs_per_edge)
    weights: list[Array] = []
    means: list[Array] = []
    log_scales: list[Array] = []
    active: list[int] = []
    for n_in, n_out in zip(config.widths[:-1], config.widths[1:], strict=True):
        w = jnp.zeros((int(n_out), int(n_in), g), dtype=jnp.float64)
        w = w.at[..., :live].set(1.0 / float(max(live, 1)))
        weights.append(w)
        means.append(jnp.zeros((int(n_out), int(n_in), g), dtype=jnp.float64))
        log_scales.append(jnp.zeros((int(n_out), int(n_in), g), dtype=jnp.float64))
        active.append(live)
    params = JetKANParams(
        weights=tuple(weights),
        means=tuple(means),
        log_scales=tuple(log_scales),
        active_g=tuple(active),
    )
    return act, params


def _order_at(orders: tuple[int, ...], g: int) -> int:
    return int(orders[g]) if g < len(orders) else int(orders[-1])


def _layer_apply(
    x: Array,
    weights: Array,
    means: Array,
    log_scales: Array,
    *,
    orders: tuple[int, ...],
    active_g: int,
    act: JaxActivationSpec,
) -> Array:
    if act.fastpath is None:
        raise NotImplementedError(
            f"Activation {act.name!r} has no closed-form derivative kernel"
        )
    fp = act.fastpath
    acc: Array | None = None
    u = x[..., None, :]
    for g in range(int(active_g)):
        n = _order_at(orders, g)
        alpha = jnp.exp(log_scales[..., g])
        z = alpha * u + means[..., g]
        term = weights[..., g] * fp(z, n)
        acc = term if acc is None else acc + term
    assert acc is not None
    return acc.sum(axis=-1)


def jet_kan_apply(
    params: JetKANParams,
    x: Array,
    *,
    config: JetKANConfig,
    base: str | JaxActivationSpec | None = None,
) -> Array:
    act = get_activation(config.base) if base is None else (
        get_activation(base) if isinstance(base, str) else base
    )
    h = x
    for i in range(len(params.weights)):
        h = _layer_apply(
            h,
            params.weights[i],
            params.means[i],
            params.log_scales[i],
            orders=config.orders,
            active_g=params.active_g[i],
            act=act,
        )
    return h


def _edge_tower(
    u0: Array,
    weights: Array,
    means: Array,
    log_scales: Array,
    orders: tuple[int, ...],
    act: JaxActivationSpec,
    order: int,
) -> Array:
    if act.fastpath is None:
        raise NotImplementedError(
            f"Activation {act.name!r} has no closed-form derivative kernel"
        )
    fp = act.fastpath
    rows: list[Array] = []
    g_live = int(weights.shape[-1])
    for k in range(order + 1):
        acc: Array | None = None
        for g in range(g_live):
            n = _order_at(orders, g)
            alpha = jnp.exp(log_scales[g])
            term = weights[g] * (alpha ** k) * fp(alpha * u0 + means[g], n + k)
            acc = term if acc is None else acc + term
        assert acc is not None
        rows.append(acc)
    return jnp.stack(rows, axis=0)


def _layer_directional_jet(
    in_jet: Array,
    weights: Array,
    means: Array,
    log_scales: Array,
    *,
    orders: tuple[int, ...],
    active_g: int,
    act: JaxActivationSpec,
    order: int,
) -> Array:
    n_out = int(weights.shape[0])
    n_in = int(weights.shape[1])
    outs: list[Array] = []
    for q in range(n_out):
        acc: Array | None = None
        for p in range(n_in):
            u_jet = in_jet[:, p]
            tower = _edge_tower(
                u_jet[0],
                weights[q, p, :active_g],
                means[q, p, :active_g],
                log_scales[q, p, :active_g],
                orders,
                act,
                order,
            )
            part = compose_jet(u_jet, tower)
            acc = part if acc is None else acc + part
        assert acc is not None
        outs.append(acc)
    return jnp.stack(outs, axis=-1)


def jet_kan_jet(
    params: JetKANParams,
    x: Array,
    order: int,
    direction: Array,
    *,
    config: JetKANConfig,
) -> Array:
    act = get_activation(config.base)
    x0 = jnp.asarray(x)
    v = jnp.asarray(direction)
    rows = [x0]
    if order >= 1:
        rows.append(v)
    rows.extend(jnp.zeros_like(x0) for _ in range(max(order - 1, 0)))
    jet = jnp.stack(rows[: order + 1], axis=0)
    for i in range(len(params.weights)):
        jet = _layer_directional_jet(
            jet,
            params.weights[i],
            params.means[i],
            params.log_scales[i],
            orders=config.orders,
            active_g=params.active_g[i],
            act=act,
            order=order,
        )
    return jet


def _layer_mixed_jet(
    in_jets: Array,
    weights: Array,
    means: Array,
    log_scales: Array,
    *,
    orders: tuple[int, ...],
    active_g: int,
    act: JaxActivationSpec,
    dim: int,
    order: int,
) -> Array:
    n_out = int(weights.shape[0])
    n_in = int(weights.shape[1])
    outs: list[Array] = []
    for q in range(n_out):
        acc: Array | None = None
        for p in range(n_in):
            u_jet = in_jets[:, p]
            tower = _edge_tower(
                u_jet[0],
                weights[q, p, :active_g],
                means[q, p, :active_g],
                log_scales[q, p, :active_g],
                orders,
                act,
                order,
            )
            part = compose_jet_mv(u_jet, tower, dim, order)
            acc = part if acc is None else acc + part
        assert acc is not None
        outs.append(acc)
    return jnp.stack(outs, axis=-1)


def jet_kan_jet_mv(
    params: JetKANParams,
    x: Array,
    total_order: int,
    *,
    config: JetKANConfig,
) -> Array:
    act = get_activation(config.base)
    x0 = jnp.asarray(x)
    dim = int(x0.shape[0])
    jet = identity_jet(x0, total_order)
    for i in range(len(params.weights)):
        jet = _layer_mixed_jet(
            jet,
            params.weights[i],
            params.means[i],
            params.log_scales[i],
            orders=config.orders,
            active_g=params.active_g[i],
            act=act,
            dim=dim,
            order=total_order,
        )
    return jet


def refine_pack(params: JetKANParams, config: JetKANConfig) -> JetKANParams:
    """Activate the next zero-weight pack on every layer (forward unchanged)."""
    cap = _capacity(config)
    active = []
    for a in params.active_g:
        if int(a) >= cap:
            raise RuntimeError("no unused pack slots remain")
        active.append(int(a) + 1)
    return JetKANParams(params.weights, params.means, params.log_scales, tuple(active))


def jet_kan_from_torch_state(
    config: JetKANConfig,
    state: Sequence[tuple[Any, Any, Any, int]],
) -> JetKANParams:
    """``state`` is ``[(weights, means, log_scales, active_g), ...]``."""
    del config
    weights = tuple(jnp.asarray(w, dtype=jnp.float64) for w, _m, _s, _a in state)
    means = tuple(jnp.asarray(m, dtype=jnp.float64) for _w, m, _s, _a in state)
    log_scales = tuple(jnp.asarray(s, dtype=jnp.float64) for _w, _m, s, _a in state)
    active = tuple(int(a) for _w, _m, _s, a in state)
    return JetKANParams(weights, means, log_scales, active)


__all__ = [
    "JetKANConfig",
    "JetKANParams",
    "init_jet_kan",
    "jet_kan_apply",
    "jet_kan_from_torch_state",
    "jet_kan_jet",
    "jet_kan_jet_mv",
    "jetkan_from_band_plan",
    "refine_pack",
]
