# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""SU(2) lattice Langevin / stochastic-quantisation updater (JAX backend).

Parisi-Wu stochastic quantisation of the lattice gauge field: each link follows
overdamped Langevin dynamics on ``SU(2) = S^3`` whose stationary distribution is
the Wilson-action weight ``exp(-S)`` -- the same distribution the Kennedy-Pendleton
heat bath samples directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.geometry.gauge.lattice._core import kernels
from omnibias.geometry.gauge.lattice.jax.su2 import checkerboard_mask, staple_sum


def langevin_update_links(
    links: Array,
    mu: int,
    mask: Array,
    beta: float,
    eps: float,
    key: Array,
    *,
    noise: Array | None = None,
) -> Array:
    """Functional geodesic Langevin update of ``links[mu]`` where ``mask`` is True."""
    staple = staple_sum(links, mu)
    u_all = links[mu]
    staple_all = staple
    if noise is None:
        key, noise_key = jax.random.split(key)
        xi_all = jax.random.normal(noise_key, (*u_all.shape[:-1], 3), dtype=links.dtype)
    else:
        xi_all = noise
    stepped = kernels.langevin_link_step(jnp, u_all, staple_all, beta, eps, xi_all)
    new_mu = jnp.where(mask[..., None], stepped, links[mu])
    return links.at[mu].set(new_mu)


def langevin_sweep(
    links: Array,
    beta: float,
    eps: float,
    key: Array,
    *,
    n_sub: int = 1,
) -> Array:
    """One Langevin sweep: ``n_sub`` checkerboard passes over all directions."""
    lattice_shape = links.shape[1:-1]
    for _ in range(n_sub):
        for mu in range(4):
            for parity in (0, 1):
                key, subkey = jax.random.split(key)
                mask = checkerboard_mask(lattice_shape, parity, dtype=links.dtype)
                links = langevin_update_links(links, mu, mask, beta, eps, subkey)
    return links


__all__ = ["langevin_sweep", "langevin_update_links"]
