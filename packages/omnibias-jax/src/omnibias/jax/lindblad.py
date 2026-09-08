# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable GKSL / Lindblad dynamics (JAX twin; theory 09-32).

Point-for-point mirror of :mod:`omnibias.torch.lindblad` and bit-identical
(given the same matrix exponential) twin of :mod:`omnibias.core.lindblad`.
The superoperator is assembled with ``jnp.kron``; the propagator is
``jax.scipy.linalg.expm``. This is the linear-semigroup analogue of the
sigma tower -- one propagator evaluation regardless of derivative order
-- and is labelled as such, not as the activation tower.

``order`` is a static Python ``int`` (branches only at trace time), so
:func:`derivative_tower` stays ``jit`` / ``vmap`` / ``grad`` safe.
Default dtype follows JAX's current default, never a hardcoded
``float32``.

Two collapse senses are named (mirroring the core module) and must not
be conflated: the founding bias collapse (``delta -> 0``) never appears
here; ``beta -> inf`` is the founding **temperature collapse**
(feasibility sense) and is not evaluated or requested by anything here
-- see :mod:`omnibias.core.occupancy` for the honest T=0 reference and
:mod:`omnibias.core.collapse.relaxation` for the new ``relaxation``
slot. Do not conflate the two.

The GKSL form is a caller input. Not a Born–Markov derivation, not a
general closed form, no thermodynamic limit.
"""

from __future__ import annotations

from typing import cast

from omnibias.jax.occupancy import occupancy

import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import Array


def liouvillian(hamiltonian: Array, jumps: Array, rates: Array) -> Array:
    """Complex superoperator ``L`` of shape ``(d**2, d**2)``, column-stacked."""
    ham = jnp.asarray(hamiltonian)
    dim = ham.shape[-1]
    eye = jnp.eye(dim, dtype=ham.dtype)
    gen = -1j * (jnp.kron(eye, ham) - jnp.kron(jnp.swapaxes(ham, -2, -1), eye))
    jump_ops = jnp.asarray(jumps, dtype=ham.dtype)
    rate_ops = jnp.asarray(rates)
    gen, _ = jax_scan_jumps(gen, jump_ops, rate_ops)
    return gen


def jax_scan_jumps(gen: Array, jump_ops: Array, rate_ops: Array) -> tuple[Array, None]:
    """Python loop over a static number of jumps (trace-time ``k``)."""
    for k in range(int(jump_ops.shape[0])):
        jump = jump_ops[k]
        rate = rate_ops[k]
        dag = jnp.conj(jnp.swapaxes(jump, -2, -1))
        left = dag @ jump
        dissipator = jnp.kron(jnp.conj(jump), jump) - 0.5 * (
            jnp.kron(jnp.eye(jump.shape[-1], dtype=jump.dtype), left)
            + jnp.kron(jnp.swapaxes(left, -2, -1), jnp.eye(jump.shape[-1], dtype=jump.dtype))
        )
        gen = gen + rate.astype(gen.dtype) * dissipator
    return gen, None


def _vec(rho: Array) -> Array:
    dim = rho.shape[-1]
    return jnp.swapaxes(rho, -2, -1).reshape(*rho.shape[:-2], dim * dim)


def _unvec(vector: Array, dim: int) -> Array:
    return jnp.swapaxes(vector.reshape(*vector.shape[:-1], dim, dim), -2, -1)


def propagator(hamiltonian: Array, jumps: Array, rates: Array, time: Array | float) -> Array:
    """``exp(t L)``."""
    gen = liouvillian(hamiltonian, jumps, rates)
    return cast(Array, jsp_linalg.expm(time * gen))


def evolve(
    rho0: Array,
    hamiltonian: Array,
    jumps: Array,
    rates: Array,
    time: Array | float,
) -> Array:
    """``rho(t) = exp(t L) rho(0)``."""
    dim = rho0.shape[-1]
    prop = propagator(hamiltonian, jumps, rates, time)
    vec = _vec(jnp.asarray(rho0, dtype=prop.dtype))
    return _unvec(prop @ vec, dim)


def apply_lindblad(rho: Array, hamiltonian: Array, jumps: Array, rates: Array) -> Array:
    """Evaluate ``L[rho]`` by the GKSL formula."""
    ham = jnp.asarray(hamiltonian)
    rho_c = jnp.asarray(rho, dtype=ham.dtype)
    out = -1j * (ham @ rho_c - rho_c @ ham)
    jump_ops = jnp.asarray(jumps, dtype=ham.dtype)
    rate_ops = jnp.asarray(rates)
    for k in range(int(jump_ops.shape[0])):
        jump = jump_ops[k]
        dag = jnp.conj(jnp.swapaxes(jump, -2, -1))
        lindblad = jump @ rho_c @ dag - 0.5 * (dag @ jump @ rho_c + rho_c @ dag @ jump)
        out = out + rate_ops[k].astype(ham.dtype) * lindblad
    return out


def derivative_tower(
    rho0: Array,
    hamiltonian: Array,
    jumps: Array,
    rates: Array,
    time: Array | float,
    *,
    order: int,
) -> Array:
    """Stack ``(rho, L rho, ..., L^order rho)`` at time ``t``. ``order`` is static."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    gen = liouvillian(hamiltonian, jumps, rates)
    dim = rho0.shape[-1]
    vec = propagator(hamiltonian, jumps, rates, time) @ _vec(
        jnp.asarray(rho0, dtype=gen.dtype)
    )
    rows: list[Array] = []
    for _ in range(order + 1):
        rows.append(_unvec(vec, dim))
        vec = gen @ vec
    return jnp.stack(rows, axis=0)


def steady_state(hamiltonian: Array, jumps: Array, rates: Array) -> Array:
    """Unique trace-1 kernel vector of ``L`` via a linear solve."""
    gen = liouvillian(hamiltonian, jumps, rates)
    dim = hamiltonian.shape[-1]
    dim2 = dim * dim
    idx = jnp.arange(dim)
    col = idx + idx * dim
    zero_row = jnp.zeros((dim2,), dtype=gen.dtype)
    ones = jnp.ones((dim,), dtype=gen.dtype)
    last = zero_row.at[col].set(ones)
    augmented = gen.at[-1, :].set(last)
    rhs = zero_row.at[-1].set(jnp.array(1, dtype=gen.dtype))
    vec = jnp.linalg.solve(augmented, rhs)
    return _unvec(vec, dim)


def thermal_population(omega: Array, beta: Array) -> Array:
    """Excited-state Fermi occupancy ``sigma(-beta * omega)``."""
    mu = jnp.zeros_like(omega)
    return occupancy(omega, beta, mu)


__all__ = [
    "apply_lindblad",
    "derivative_tower",
    "evolve",
    "liouvillian",
    "propagator",
    "steady_state",
    "thermal_population",
]
