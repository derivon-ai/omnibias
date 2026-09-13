# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SU(2) lattice links as unit quaternions (JAX backend).

The deterministic quaternion / staple math is delegated to the backend-agnostic
:mod:`omnibias.geometry.gauge.lattice._core.kernels` (shared bit-identically with the torch
backend); this module owns the JAX-specific RNG (``jax.random.PRNGKey``) and the
functional checkerboard updates.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.geometry.gauge.lattice._core import kernels

if TYPE_CHECKING:
    from collections.abc import Sequence


def identity_links(
    lattice_shape: Sequence[int],
    *,
    dtype: jnp.dtype = jnp.float64,
) -> Array:
    """Return cold-start links: identity at every site, shape (4, *shape, 4)."""
    shape = tuple(int(s) for s in lattice_shape)
    links = jnp.zeros((4, *shape, 4), dtype=dtype)
    links = links.at[..., 0].set(1.0)
    return links


def random_links(
    lattice_shape: Sequence[int],
    key: Array,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> Array:
    """Uniformly random SU(2) links, shape (4, *shape, 4)."""
    shape = tuple(int(s) for s in lattice_shape)
    q = jax.random.normal(key, (4, *shape, 4), dtype=dtype)
    return normalize_quaternion(q)


def normalize_quaternion(q: Array) -> Array:
    """Project ``q`` onto the unit quaternion sphere (last dim = 4)."""
    return kernels.normalize_quaternion(jnp, q)


def quat_conj(q: Array) -> Array:
    """Quaternion conjugate / inverse for unit links."""
    return kernels.quat_conj(jnp, q)


def quat_mul(a: Array, b: Array) -> Array:
    """Group product matching ``U(q_a q_b) = U(q_a) @ U(q_b)`` (left path order)."""
    return kernels.quat_mul(jnp, a, b)


def quat_to_matrix(q: Array) -> Array:
    """Map unit quaternion ``(q0,q1,q2,q3)`` to 2x2 complex SU(2) matrix."""
    return kernels.quat_to_matrix(jnp, q.astype(jnp.float64))


def matrix_to_quat(u: Array) -> Array:
    """Project a 2x2 matrix onto the nearest unit quaternion (last two dims 2x2)."""
    return kernels.matrix_to_quat(jnp, u.astype(jnp.complex128))


def staple_sum(links: Array, mu: int) -> Array:
    """Sum of forward + backward staples for direction ``mu`` (quaternion-valued)."""
    return kernels.staple_sum(jnp, links, mu)


def staple_hat_and_magnitude(staple: Array) -> tuple[Array, Array]:
    """Return unit direction ``U_hat`` and scalar magnitude ``a = ||staple||``."""
    return kernels.staple_hat_and_magnitude(jnp, staple)


@partial(jax.jit, static_argnames=("max_iter",))
def _sample_q0_kp(
    a: Array,
    beta: float,
    key: Array,
    *,
    max_iter: int | None = 10_000,
) -> Array:
    r"""Rejection sample the Haar-weighted density ``sqrt(1-q0^2) exp(w q0)``.

    The inverse-CDF proposal and acceptance law match the torch backend.
    Floating-point inverse-CDF arithmetic and the uniform approximation for
    ``abs(w) <= 1e-10`` are numerical. A tracer-safe loop retries pending
    sites with a generous default budget. Exhaustion raises (also under jit),
    never returning rejected proposals. ``max_iter=None`` explicitly opts
    into unbounded continuation. Nonfinite weights fail without entering an unbounded loop.
    The compiled loop is cached by shape/dtype and retry budget; weights and
    the random key remain dynamic across calls.
    """
    if max_iter is not None and (type(max_iter) is not int or max_iter < 1):
        raise ValueError("max_iter must be a positive integer or None")
    w = beta * a
    valid_weights = jnp.all(jnp.isfinite(w))
    magnitude = jnp.abs(w)
    nonzero = magnitude > 1e-10
    w_safe = jnp.where(nonzero, magnitude, jnp.ones_like(w))

    def pending(state):
        iterations, _, _, accepted = state
        active = valid_weights & jnp.any(~accepted)
        if max_iter is not None:
            active = active & (iterations < max_iter)
        return active

    def propose(state):
        iterations, loop_key, q0, accepted = state
        loop_key, k_r, k_u, k_acc = jax.random.split(loop_key, 4)
        r = jax.random.uniform(k_r, a.shape, dtype=a.dtype)
        cand_exp = 1.0 + jnp.log(r + (1.0 - r) * jnp.exp(-2.0 * w_safe)) / w_safe
        cand_unif = 2.0 * jax.random.uniform(k_u, a.shape, dtype=a.dtype) - 1.0
        cand = jnp.where(nonzero, cand_exp, cand_unif)
        cand = jnp.where(w < 0, -cand, cand)
        u = jax.random.uniform(k_acc, a.shape, dtype=a.dtype)
        # Zero RNG output must not accept an endpoint or infinite proposal.
        accept = (u * u < jnp.maximum(1.0 - cand * cand, 0.0)) & (~accepted)
        q0 = jnp.where(accept, cand, q0)
        return iterations + 1, loop_key, q0, accepted | accept

    _, _, q0, accepted = jax.lax.while_loop(
        pending, propose,
        (jnp.asarray(0), key, jnp.zeros_like(a), jnp.zeros(a.shape, dtype=bool)),
    )

    def require_complete(valid, complete):
        # This callback executes on host values, never on traced tensors.
        if not bool(valid):
            raise ValueError("SU(2) heat-bath weights must be finite")
        if not bool(complete):
            raise RuntimeError("SU(2) heat-bath rejection budget exhausted; no sample returned")
        return 0

    complete = jnp.all(accepted)
    # An ordered io_callback is guaranteed to execute, unlike debug.callback.
    # This keeps failure observable even if jit callers discard the sample.
    # The sampler supports eager/jit use; ordered callbacks are not a vmap API.
    jax.lax.cond(
        valid_weights & complete,
        lambda _: jnp.asarray(0, dtype=jnp.int32),
        lambda _: jax.experimental.io_callback(
            require_complete, jax.ShapeDtypeStruct((), jnp.int32),
            valid_weights, complete, ordered=True,
        ),
        operand=None,
    )
    return q0


def _sample_unit_sphere(key: Array, shape: tuple[int, ...], *, dtype: jnp.dtype) -> Array:
    v = jax.random.normal(key, (*shape, 3), dtype=dtype)
    norm = jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-30)
    return v / norm


def heatbath_update_links(
    links: Array,
    mu: int,
    mask: Array,
    beta: float,
    key: Array,
) -> Array:
    """Functional Kennedy-Pendleton heat bath on ``links[mu]`` at sites where ``mask`` is True."""
    staple = staple_sum(links, mu)
    u_hat, a = staple_hat_and_magnitude(staple)
    key, k_q0, k_sphere = jax.random.split(key, 3)
    q0 = _sample_q0_kp(a, beta, k_q0)
    r = _sample_unit_sphere(k_sphere, a.shape, dtype=links.dtype)
    radial = jnp.sqrt(jnp.maximum(1.0 - q0 * q0, 0.0))
    v = jnp.concatenate((q0[..., None], r * radial[..., None]), axis=-1)
    u_new = quat_mul(v, u_hat)
    new_mu = jnp.where(mask[..., None], u_new, links[mu])
    return links.at[mu].set(new_mu)


def overrelax_update_links(links: Array, mu: int, mask: Array) -> Array:
    """Functional over-relaxation: ``U' = U_hat * U^{-1} * U_hat``."""
    staple = staple_sum(links, mu)
    u_hat, _ = staple_hat_and_magnitude(staple)
    u = links[mu]
    u_inv = quat_conj(u)
    u_new = quat_mul(quat_mul(u_hat, u_inv), u_hat)
    new_mu = jnp.where(mask[..., None], u_new, links[mu])
    return links.at[mu].set(new_mu)


def checkerboard_mask(
    lattice_shape: Sequence[int],
    parity: int,
    *,
    dtype: jnp.dtype = jnp.float64,
) -> Array:
    """Boolean mask for even (parity=0) or odd (parity=1) sites."""
    shape = tuple(int(s) for s in lattice_shape)
    grids = jnp.meshgrid(*(jnp.arange(s, dtype=dtype) for s in shape), indexing="ij")
    parity_sum = grids[0]
    for g in grids[1:]:
        parity_sum = parity_sum + g
    return (parity_sum % 2 == parity).astype(bool)


def sweep(
    links: Array,
    beta: float,
    key: Array,
    *,
    n_overrelax: int = 2,
) -> Array:
    """One MC sweep: heat bath (both colors, all directions) + over-relaxation hits."""
    lattice_shape = links.shape[1:-1]
    for mu in range(4):
        for p in (0, 1):
            key, subkey = jax.random.split(key)
            mask = checkerboard_mask(lattice_shape, p, dtype=links.dtype)
            links = heatbath_update_links(links, mu, mask, beta, subkey)
    for _ in range(n_overrelax):
        for mu in range(4):
            for p in (0, 1):
                mask = checkerboard_mask(lattice_shape, p, dtype=links.dtype)
                links = overrelax_update_links(links, mu, mask)
    return links
