# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Depth-causal local jet (theory 08-03), JAX twin.

Each layer takes a damped Gauss-Newton step on a named local residual
and pushes a compressed ``k``-direction :func:`~omnibias.jax.jet.layer_jet`
to the next layer. Later parameters update first so an earlier layer
sees a corrected downstream map. ``compose_jet`` is the chain rule.

Do not wrap the driver in ``jax.jit``. The API raises
:class:`LocalJetForbidden` when ``n_directions >= n_params`` unless
``allow_full=True``. Local GN is greedy, not a global min, and not CCF
stretch. Bias collapse (``delta -> 0``) supplies the tower.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple, cast

from omnibias.core.local_jet import (
    LocalJetConfig,
    LocalJetForbidden,
    LocalJetReport,
    invert_sigma,
    mlp_param_count,
    reject_local_jet_flood,
    require_invertible_sigma,
)
from omnibias.jax.activations import get_activation
from omnibias.jax.jet import affine_jet, layer_jet
from omnibias.jax.optim import gauss_newton_direction

import jax
import jax.numpy as jnp
from jax import Array

Layer = tuple[Array, Array | None, str | None]
LocalResidualFn = Callable[["LocalLayerState"], Array]


class LocalLayerState(NamedTuple):
    """Named residual arguments at one layer during a reverse sweep."""

    layer_index: int
    h: Array
    u: Array
    h_prev: Array
    jet: Array
    output: Array
    output_jet: Array


def _as_layer_list(layers: Sequence[Layer]) -> list[Layer]:
    if not layers:
        raise ValueError("layers must be non-empty")
    out: list[Layer] = []
    for W, b, spec in layers:
        out.append((jnp.asarray(W), None if b is None else jnp.asarray(b), spec))
    return out


def _param_sizes(layers: Sequence[Layer]) -> list[tuple[int, int | None]]:
    sizes: list[tuple[int, int | None]] = []
    for W, b, _spec in layers:
        sizes.append((int(W.size), None if b is None else int(b.size)))
    return sizes


def _pack(W: Array, b: Array | None) -> Array:
    flat_w = jnp.reshape(W, (-1,))
    if b is None:
        return flat_w
    return jnp.concatenate([flat_w, jnp.reshape(b, (-1,))], axis=0)


def _unpack(theta: Array, W: Array, b: Array | None) -> tuple[Array, Array | None]:
    n_w = int(W.size)
    new_w = jnp.reshape(theta[:n_w], W.shape)
    if b is None:
        return new_w, None
    return new_w, jnp.reshape(theta[n_w:], b.shape)


def _affine(h: Array, W: Array, b: Array | None) -> Array:
    u: Array = jnp.tensordot(h, W, axes=([-1], [-1]))
    if b is not None:
        u = u + b
    return u


def _sigma(u: Array, spec: str | None) -> Array:
    if spec is None:
        return u
    return get_activation(spec).forward(u)


def _prepare_x(x: Array, d_in: int) -> Array:
    x_t = jnp.asarray(x)
    if x_t.ndim == 0:
        x_t = jnp.reshape(x_t, (1,))
    if x_t.ndim == 1 and int(x_t.shape[0]) != d_in:
        if d_in == 1:
            return x_t[:, None]
        raise ValueError(
            f"x shape {tuple(x_t.shape)} is not a single sample of dim {d_in} "
            f"or a batch of scalars"
        )
    if x_t.ndim >= 2 and int(x_t.shape[-1]) != d_in:
        raise ValueError(
            f"x trailing dim {int(x_t.shape[-1])} must equal d_in={d_in}"
        )
    return x_t


def _default_directions(d_in: int, k: int, *, dtype: jnp.dtype) -> Array:
    eye = jnp.eye(d_in, dtype=dtype)
    return jnp.stack([eye[i % d_in] for i in range(k)], axis=0)


def make_input_jet(x: Array, directions: Array, order: int) -> Array:
    """Taylor jet of ``x + t v_i`` for each input direction ``v_i``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    x_t = jnp.asarray(x)
    dirs = jnp.asarray(directions, dtype=x_t.dtype)
    if dirs.ndim != 2 or int(dirs.shape[-1]) != int(x_t.shape[-1]):
        raise ValueError(
            f"directions must have shape (k, d_in={int(x_t.shape[-1])}), "
            f"got {tuple(dirs.shape)}"
        )
    k = int(dirs.shape[0])
    if x_t.ndim == 1:
        value = jnp.broadcast_to(x_t[None, :], (k, int(x_t.shape[0])))
        dir_rows = dirs
        zero = jnp.zeros((k, int(x_t.shape[0])), dtype=x_t.dtype)
    else:
        n = int(x_t.shape[0])
        d = int(x_t.shape[-1])
        value = jnp.broadcast_to(x_t[None, :, :], (k, n, d))
        dir_rows = jnp.broadcast_to(dirs[:, None, :], (k, n, d))
        zero = jnp.zeros((k, n, d), dtype=x_t.dtype)
    rows = [value]
    if order >= 1:
        rows.append(dir_rows)
    rows.extend(zero for _ in range(max(order - 1, 0)))
    return jnp.stack(rows[: order + 1], axis=0)


def _push_layer(z_jet: Array, layer: Layer) -> Array:
    W, b, spec = layer
    if spec is None:
        return affine_jet(z_jet, W, b)
    return layer_jet(z_jet, W, b, spec)


def _push_tail(z_jet: Array, layers: Sequence[Layer]) -> Array:
    jet = z_jet
    for layer in layers:
        jet = _push_layer(jet, layer)
    return jet


def _value_from_jet(jet: Array) -> Array:
    row = jet[0]
    if row.ndim >= 2 and int(row.shape[0]) >= 1:
        return row[0]
    return row


def _flatten_residual(res: Array) -> Array:
    flat = jnp.reshape(jnp.asarray(res), (-1,))
    if int(flat.size) == 0:
        raise ValueError("local residual must be non-empty")
    return flat


def _spec_name(spec: str | None) -> str | None:
    if spec is None:
        return None
    return get_activation(spec).name


def _invert_tensor(name: str, target: Array) -> Array:
    require_invertible_sigma(name)
    y = jnp.asarray(target)
    key = name.lower()
    if key == "tanh":
        if bool(jnp.any(jnp.abs(y) >= 1.0)):
            raise ValueError("artanh requires |y| < 1 for every entry")
        return jnp.arctanh(y)
    if bool(jnp.any((y <= 0.0) | (y >= 1.0))):
        raise ValueError("logit requires y in (0, 1) for every entry")
    return jnp.log(y / (1.0 - y))


def _readout_residual(output: Array, target: Array) -> Array:
    out = _flatten_residual(output)
    tgt = jnp.reshape(jnp.asarray(target, dtype=out.dtype), (-1,))
    if int(tgt.size) != int(out.size):
        raise ValueError(
            f"readout target has {int(tgt.size)} entries, "
            f"network output has {int(out.size)}"
        )
    return out - tgt


def _named_residual(
    *,
    variant: str,
    spec: str | None,
    state: LocalLayerState,
    target: Array | None,
    z_star: Array | None,
) -> Array | None:
    if variant == "readout":
        if target is None:
            raise ValueError("readout variant requires target")
        return _readout_residual(state.output, target)
    if variant == "invert":
        name = _spec_name(spec)
        if name is None or z_star is None:
            return None
        return _flatten_residual(state.u - z_star)
    if target is None:
        raise ValueError("predcode variant requires target")
    t = jnp.asarray(target)
    h = state.h
    if t.shape != h.shape:
        if int(t.size) == int(h.size):
            t = jnp.reshape(t, h.shape)
        else:
            return None
    return _flatten_residual(h - jax.lax.stop_gradient(t))


def _gn_update(
    layer: Layer,
    h_prev: Array,
    incoming_jet: Array,
    tail: Sequence[Layer],
    *,
    layer_index: int,
    config: LocalJetConfig,
    target: Array | None,
    local_residual_fn: LocalResidualFn | None,
) -> tuple[Layer, float]:
    W0, b0, spec = layer
    z_star: Array | None = None
    if local_residual_fn is None and config.variant == "invert":
        name = _spec_name(spec)
        if name is not None:
            if target is None:
                raise ValueError("invert variant requires target")
            z_star = _invert_tensor(name, target)

    def residual_of(theta: Array) -> Array:
        W, b = _unpack(theta, W0, b0)
        trial: Layer = (W, b, spec)
        u = _affine(h_prev, W, b)
        h = _sigma(u, spec)
        jet = _push_layer(incoming_jet, trial)
        out_jet = _push_tail(jet, tail)
        output = _value_from_jet(out_jet)
        state = LocalLayerState(
            layer_index=layer_index,
            h=h,
            u=u,
            h_prev=h_prev,
            jet=jet,
            output=output,
            output_jet=out_jet,
        )
        if local_residual_fn is not None:
            return _flatten_residual(local_residual_fn(state))
        named = _named_residual(
            variant=config.variant,
            spec=spec,
            state=state,
            target=target,
            z_star=z_star,
        )
        if named is None:
            return jnp.zeros((0,), dtype=theta.dtype)
        return named

    theta0 = _pack(W0, b0)
    probe = residual_of(theta0)
    if int(probe.size) == 0:
        return layer, 0.0
    jac = cast(Array, jax.jacrev(residual_of)(theta0))
    if jac.ndim == 1:
        jac = jac[None, :]
    delta = gauss_newton_direction(jac, probe, config.damping)
    new_w, new_b = _unpack(theta0 + delta, W0, b0)
    new_layer: Layer = (new_w, new_b, spec)
    new_res = residual_of(theta0 + delta)
    norm = float(jnp.linalg.norm(new_res)) if int(new_res.size) else 0.0
    return new_layer, norm


def local_jet_step(
    layers: Sequence[Layer],
    x: Array,
    *,
    config: LocalJetConfig | None = None,
    target: Array | None = None,
    directions: Array | None = None,
    local_residual_fn: LocalResidualFn | None = None,
    allow_full: bool | None = None,
) -> tuple[list[Layer], LocalJetReport]:
    """One reverse-depth local GN sweep, then a forward ``layer_jet`` push.

    ``local_residual_fn`` must be a pure function of the
    :class:`LocalLayerState` (activations and jets). Later layers are
    frozen while an earlier layer steps, so the tail is a corrected
    downstream map.
    """
    cfg = config if config is not None else LocalJetConfig()
    packed = _as_layer_list(layers)
    n_params = mlp_param_count(_param_sizes(packed))
    full = cfg.allow_full if allow_full is None else bool(allow_full)
    reject_local_jet_flood(cfg.n_directions, n_params, full)
    d_in = int(packed[0][0].shape[-1])
    x_t = _prepare_x(x, d_in)
    tgt = None if target is None else jnp.asarray(target, dtype=x_t.dtype)
    if directions is None:
        dirs = _default_directions(d_in, cfg.n_directions, dtype=x_t.dtype)
    else:
        dirs = jnp.asarray(directions, dtype=x_t.dtype)
        if int(dirs.shape[0]) != cfg.n_directions:
            raise ValueError(
                f"directions rows {int(dirs.shape[0])} must equal "
                f"n_directions={cfg.n_directions}"
            )
    incoming = make_input_jet(x_t, dirs, cfg.jet_order)
    in_jets: list[Array] = [incoming]
    h_prev_list: list[Array] = [x_t]
    h = x_t
    jet = incoming
    for layer in packed:
        h = _sigma(_affine(h, layer[0], layer[1]), layer[2])
        jet = _push_layer(jet, layer)
        h_prev_list.append(h)
        in_jets.append(jet)

    def _refresh() -> tuple[list[Array], list[Array]]:
        hs = [x_t]
        jets: list[Array] = [incoming]
        h_cur = x_t
        jet_cur = incoming
        for lyr in updated:
            h_cur = _sigma(_affine(h_cur, lyr[0], lyr[1]), lyr[2])
            jet_cur = _push_layer(jet_cur, lyr)
            hs.append(h_cur)
            jets.append(jet_cur)
        return hs, jets

    updated = list(packed)
    norms: list[float] = [0.0] * len(updated)
    for ell in range(len(updated) - 1, -1, -1):
        new_layer, norm = _gn_update(
            updated[ell],
            h_prev_list[ell],
            in_jets[ell],
            updated[ell + 1 :],
            layer_index=ell,
            config=cfg,
            target=tgt,
            local_residual_fn=local_residual_fn,
        )
        updated[ell] = new_layer
        norms[ell] = norm
    if cfg.refit_last:
        h_prev_list, in_jets = _refresh()
        last = len(updated) - 1
        updated[last], norms[last] = _gn_update(
            updated[last],
            h_prev_list[last],
            in_jets[last],
            updated[last + 1 :],
            layer_index=last,
            config=cfg,
            target=tgt,
            local_residual_fn=local_residual_fn,
        )

    final_jet = incoming
    for layer in updated:
        final_jet = _push_layer(final_jet, layer)
    report = LocalJetReport(
        residual_norms=tuple(norms),
        n_directions=cfg.n_directions,
        n_params=n_params,
        variant=cfg.variant,
        jet_shape=tuple(int(s) for s in final_jet.shape),
        greedy_only_claimed_optimal=False,
    )
    return updated, report


__all__ = [
    "Layer",
    "LocalJetConfig",
    "LocalJetForbidden",
    "LocalJetReport",
    "LocalLayerState",
    "LocalResidualFn",
    "invert_sigma",
    "local_jet_step",
    "make_input_jet",
]
