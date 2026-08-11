# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Heterogeneous multi-pack Birkhoff unit (JAX twin of torch; theory 01-01).

Closed-form evaluation of

    F(z) = sum_g c_g * sigma^(n_g)(z + mu_g)

Orders and ``mean_index`` are static Python tuples so the path stays
``jit``-safe. Founding bias collapse ``delta -> 0`` only -- no temperature
collapse.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.multipack import MultiPackSpec, PackSpec
from omnibias.jax.activations import JaxActivationSpec, get_activation

import jax.numpy as jnp
from jax import Array


def multipack_response(
    z: Array,
    means: Array,
    weights: Array,
    orders: tuple[int, ...],
    spec: JaxActivationSpec,
    *,
    mean_index: tuple[int, ...] | None = None,
) -> Array:
    """Evaluate ``sum_g c_g sigma^(n_g)(z + mu_g)`` with a closed-form tower.

    ``orders`` / ``mean_index`` must be Python tuples of static ints.
    """
    if spec.fastpath is None:
        raise NotImplementedError(
            f"Activation {spec.name!r} has no closed-form derivative kernel"
        )
    means_a = jnp.asarray(means)
    weights_a = jnp.asarray(weights)
    G = len(orders)
    if int(weights_a.shape[-1]) != G:
        raise ValueError("weights length must equal number of packs")
    if mean_index is None:
        if int(means_a.shape[-1]) != G:
            raise ValueError("means length must equal number of packs")
        index = tuple(range(G))
    else:
        if len(mean_index) != G:
            raise ValueError("mean_index length must equal number of packs")
        index = mean_index
    for n in orders:
        if n < 0:
            raise ValueError(f"order must be >= 0, got {n}")

    fp = spec.fastpath
    slots = sorted(set(index))
    slot_u = {s: z + means_a[s] for s in slots}
    out: Array | None = None
    for g, n in enumerate(orders):
        term = weights_a[g] * fp(slot_u[index[g]], n)
        out = term if out is None else out + term
    assert out is not None
    return out


def init_multipack(
    num_channels: int,
    spec: MultiPackSpec,
    *,
    base: str | JaxActivationSpec = "sigmoid",
    share_means: bool = True,
) -> tuple[JaxActivationSpec, Array, Array, tuple[int, ...], tuple[int, ...]]:
    """Return ``(act, means, weights, orders, mean_index)``."""
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    act = get_activation(base) if isinstance(base, str) else base
    if act.fastpath is None:
        raise NotImplementedError(
            f"Activation {act.name!r} has no closed-form derivative kernel"
        )
    orders = tuple(int(p.order) for p in spec.packs)
    probe = jnp.zeros(())
    for n in orders:
        act.fastpath(probe, n)
    if share_means:
        distinct = spec.distinct_means
        mean_to_slot = {m: i for i, m in enumerate(distinct)}
        mean_index = tuple(mean_to_slot[p.mean] for p in spec.packs)
        means = jnp.asarray(list(distinct), dtype=jnp.float64)
    else:
        mean_index = tuple(range(len(spec.packs)))
        means = jnp.asarray([p.mean for p in spec.packs], dtype=jnp.float64)
    weights = jnp.asarray([p.weight for p in spec.packs], dtype=jnp.float64)
    return act, means, weights, orders, mean_index


def multipack_apply(
    z: Array,
    means: Array,
    weights: Array,
    orders: tuple[int, ...],
    base: str | JaxActivationSpec = "sigmoid",
    *,
    mean_index: tuple[int, ...] | None = None,
) -> Array:
    """Functional multi-pack forward (bit-identical twin of ``MultiPackUnit``)."""
    act = get_activation(base) if isinstance(base, str) else base
    return multipack_response(
        z, means, weights, orders, act, mean_index=mean_index
    )


def packs_from_arrays(
    orders: Sequence[int],
    means: Sequence[float],
    weights: Sequence[float] | None = None,
) -> MultiPackSpec:
    """Build a :class:`MultiPackSpec` from parallel sequences."""
    w = (1.0,) * len(orders) if weights is None else tuple(float(x) for x in weights)
    if not (len(orders) == len(means) == len(w)):
        raise ValueError("orders, means, weights length mismatch")
    return MultiPackSpec(
        tuple(
            PackSpec(order=int(n), mean=float(mu), weight=float(c))
            for n, mu, c in zip(orders, means, w, strict=True)
        )
    )


BirkhoffOMBU = multipack_apply

__all__ = [
    "BirkhoffOMBU",
    "MultiPackSpec",
    "PackSpec",
    "init_multipack",
    "multipack_apply",
    "multipack_response",
    "packs_from_arrays",
]
