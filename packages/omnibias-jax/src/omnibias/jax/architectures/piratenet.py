# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Reusable PirateNet α-skip (jaxpi Wang–Li–Chen–Perdikaris).

Identity-init residual blocks: ``α=0`` is the embedding. Optional Fourier
features compose via :class:`~omnibias.jax.architectures.pinn.FourierFeatureMLP`
— this module does not fork an embed. Not ImageNet / ViT, not CCF stretch,
not a Wave-3 gated invention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class PirateNetConfig:
    """PirateNet width / depth. ``α`` starts at 0."""

    in_dim: int = 1
    hidden: int = 24
    n_layers: int = 2
    out_dim: int = 1
    seed: int = 0


def _glorot(key: Array, shape: tuple[int, ...]) -> Array:
    fan_in = int(shape[-1]) if len(shape) > 1 else 1
    scale = jnp.sqrt(2.0 / jnp.float64(fan_in))
    return scale * jax.random.normal(key, shape, dtype=jnp.float64)


def init_pirate_params(
    cfg: PirateNetConfig,
    *,
    key: Array | None = None,
) -> dict[str, Any]:
    """Identity-skip PirateNet (``alpha=0``) plus a zero readout."""
    hid = int(cfg.hidden)
    n_layers = int(cfg.n_layers)
    in_dim = int(cfg.in_dim)
    rng = jax.random.PRNGKey(int(cfg.seed)) if key is None else key
    keys = jax.random.split(rng, 6 + 3 * n_layers)
    if int(cfg.out_dim) == 1:
        wout: Array = jnp.zeros((hid,), dtype=jnp.float64)
        bout: Array = jnp.zeros((), dtype=jnp.float64)
    else:
        wout = jnp.zeros((hid, int(cfg.out_dim)), dtype=jnp.float64)
        bout = jnp.zeros((int(cfg.out_dim),), dtype=jnp.float64)
    params: dict[str, Any] = {
        "We": _glorot(keys[0], (hid, in_dim)),
        "be": jnp.zeros((hid,), dtype=jnp.float64),
        "Wu": _glorot(keys[1], (hid, hid)),
        "bu": jnp.zeros((hid,), dtype=jnp.float64),
        "Wv": _glorot(keys[2], (hid, hid)),
        "bv": jnp.zeros((hid,), dtype=jnp.float64),
        "Wout": wout,
        "bout": bout,
        "alpha": jnp.zeros((n_layers,), dtype=jnp.float64),
        "blocks": [],
    }
    for i in range(n_layers):
        k1, k2, k3 = keys[6 + 3 * i : 6 + 3 * i + 3]
        params["blocks"].append(
            {
                "W1": _glorot(k1, (hid, hid)),
                "b1": jnp.zeros((hid,), dtype=jnp.float64),
                "W2": _glorot(k2, (hid, hid)),
                "b2": jnp.zeros((hid,), dtype=jnp.float64),
                "W3": _glorot(k3, (hid, hid)),
                "b3": jnp.zeros((hid,), dtype=jnp.float64),
            }
        )
    params["blocks"] = tuple(params["blocks"])
    return params


def pirate_features(params: dict[str, Any], x: Array) -> Array:
    """Penultimate features, shape ``(..., hidden)``. ``alpha=0`` is identity."""
    coords = jnp.asarray(x, dtype=jnp.float64)
    if coords.ndim == 1:
        coords = coords[None, :]
        squeeze = True
    else:
        squeeze = False
    h = jnp.tanh(coords @ params["We"].T + params["be"])
    u = jnp.tanh(h @ params["Wu"].T + params["bu"])
    v = jnp.tanh(h @ params["Wv"].T + params["bv"])
    for i, block in enumerate(params["blocks"]):
        identity = h
        z = jnp.tanh(h @ block["W1"].T + block["b1"])
        z = z * u + (1.0 - z) * v
        z = jnp.tanh(z @ block["W2"].T + block["b2"])
        z = z * u + (1.0 - z) * v
        z = jnp.tanh(z @ block["W3"].T + block["b3"])
        alpha = params["alpha"][i]
        h = alpha * z + (1.0 - alpha) * identity
    if squeeze:
        return jnp.asarray(h[0])
    return jnp.asarray(h)


def pirate_apply(params: dict[str, Any], x: Array) -> Array:
    """Linear readout of :func:`pirate_features`."""
    feat = pirate_features(params, x)
    return jnp.asarray(feat @ params["Wout"] + params["bout"])


__all__ = [
    "PirateNetConfig",
    "init_pirate_params",
    "pirate_apply",
    "pirate_features",
]
