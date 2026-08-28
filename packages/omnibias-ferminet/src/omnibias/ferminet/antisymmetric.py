# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sampler-driven antisymmetric Gaussian Slater determinant.

A two-electron (or ``n_e``-electron) determinant of Gaussian orbitals on the
existing :mod:`omnibias.ferminet.sampling` ``log_abs_psi_fn(params, position)``
contract, with a closed-form Laplacian of ``log|det M|`` via the standard
matrix identity

.. math::

    \nabla_r^2 \log|\det M|
        = \sum_j \Bigl(\mathrm{tr}(M^{-1} \partial_{r_j}^2 M)
                       - \mathrm{tr}((M^{-1} \partial_{r_j} M)^2)\Bigr).

Orbital values / first / second derivatives of
``phi_i(r_j) = exp(-alpha_i \|r_j - c_i\|^2)`` are elementary (not the
tanh FermiNet stack).  This is the antisymmetric ansatz the sampling
substrate was written to consume; it does not replace
:mod:`omnibias.ferminet.restricted`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

Params = dict[str, Array]
LogAbsPsiFn = Callable[[Params, Array], Array]


def gaussian_orbital_matrix(centers: Array, alphas: Array, positions: Array) -> Array:
    """Slater matrix ``M[i, j] = exp(-alpha_i ||r_j - c_i||^2)``.

    ``centers`` is ``(n_e, dim)``, ``alphas`` is ``(n_e,)``, ``positions`` is
    ``(n_e * dim,)`` flattened (the sampling contract).
    """
    n_e, dim = centers.shape
    r = positions.reshape((n_e, dim))
    delta = r[None, :, :] - centers[:, None, :]
    dist2 = jnp.sum(delta * delta, axis=-1)
    return jnp.exp(-alphas[:, None] * dist2)


def gaussian_slater_log_abs_psi(params: Params, position: Array) -> Array:
    """``log|det M|`` for the Gaussian Slater matrix (sampling contract)."""
    matrix = gaussian_orbital_matrix(params["centers"], params["alphas"], position)
    sign, logabs = jnp.linalg.slogdet(matrix)
    return logabs + 0.0 * sign  # keep ``sign`` in the graph; log|det| is the value


def gaussian_slater_closed_form_laplacian(params: Params, position: Array) -> Array:
    """Closed-form Laplacian of ``log|det M|`` for Gaussian orbitals.

    Each column ``j`` depends only on ``r_j``, so the Hessian of ``M`` is
    block-diagonal in the electrons -- the same collapse
    :mod:`omnibias.ferminet.restricted` uses for Tier-2-lite.
    """
    centers = params["centers"]
    alphas = params["alphas"]
    n_e, dim = centers.shape
    r = position.reshape((n_e, dim))
    matrix = gaussian_orbital_matrix(centers, alphas, position)
    inverse = jnp.linalg.inv(matrix)
    lap = jnp.zeros((), dtype=position.dtype)
    for j in range(n_e):
        d1 = jnp.zeros((n_e, n_e, dim), dtype=position.dtype)
        d2_trace = jnp.zeros((n_e, n_e), dtype=position.dtype)
        for i in range(n_e):
            delta = r[j] - centers[i]
            phi = matrix[i, j]
            grad_phi = -2.0 * alphas[i] * delta * phi
            # Laplacian of phi: d/dr·grad = -2 alpha dim phi + (2 alpha)^2 |delta|^2 phi
            # wait: grad_k = -2 a delta_k phi, d/dr_k grad_k = -2a phi + (-2a delta_k)^2 phi
            # sum_k = -2a dim phi + 4 a^2 |delta|^2 phi
            hess_trace = (-2.0 * alphas[i] * dim + 4.0 * alphas[i] ** 2 * jnp.sum(delta * delta)) * phi
            d1 = d1.at[i, j, :].set(grad_phi)
            d2_trace = d2_trace.at[i, j].set(hess_trace)
        for k in range(dim):
            gcol = d1[:, :, k]
            tmp = inverse @ gcol
            lap = lap - jnp.trace(tmp @ tmp)
        lap = lap + jnp.trace(inverse @ d2_trace)
    return lap


def make_gaussian_slater_log_abs_psi(centers: Array, alphas: Array) -> LogAbsPsiFn:
    """Bind orbital parameters into a ``log_abs_psi_fn`` for the sampler."""

    def _fn(params: Params, position: Array) -> Array:
        merged = {"centers": params.get("centers", centers), "alphas": params.get("alphas", alphas)}
        return gaussian_slater_log_abs_psi(merged, position)

    return _fn


__all__ = [
    "gaussian_orbital_matrix",
    "gaussian_slater_closed_form_laplacian",
    "gaussian_slater_log_abs_psi",
    "make_gaussian_slater_log_abs_psi",
]
