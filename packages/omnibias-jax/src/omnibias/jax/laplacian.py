# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form Laplacian for an omnibias-style one-layer scalar field.

The model we accelerate is the elementary multi-bias layer:

    f(x) = b + sum_h c_h * sigma(W_h . x + beta_h),       x in R^D

Its Laplacian on ``R^D`` is

    nabla_x^2 f(x) = sum_h c_h * sigma''(W_h . x + beta_h) * ||W_h||^2.

Computing this with standard JAX would call ``jax.hessian`` or
``jax.jacfwd(jax.jacrev(f))`` on the field, which builds a full D x D
Hessian matrix and traces it. That is ``O(B * D * H)`` per backward
pass and produces ``O(B * D^2)`` intermediate memory.

The omnibias kernel here is ``O(B * H)`` and ``O(1)`` in ``D`` for
the *Laplacian* (the ``||W_h||^2`` term is a per-row scalar reduction,
computed once per parameter update).

This module is the JAX-side workhorse a FermiNet/DeepQMC-style code
would call to assemble the kinetic-energy term of a neural
wavefunction without nesting two ``jax.jacrev`` calls.

The :func:`neural_field_value_and_laplacian` function additionally
returns the gradient, which most VMC loops need anyway.
"""

from __future__ import annotations

from omnibias.jax.activations import JaxActivationSpec, get_activation

import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Plain field evaluation
# ---------------------------------------------------------------------------


def neural_field_value(
    x: Array,  # (B, D) or (D,)
    W: Array,  # (H, D)
    beta: Array,  # (H,)
    c: Array,  # (H,)
    b: Array | float,  # scalar
    activation: str | JaxActivationSpec,
) -> Array:
    """``f(x) = b + sum_h c_h sigma(W_h . x + beta_h)``.

    Batched in ``x`` if it has shape ``(B, D)``; returns ``(B,)``. For
    a single point ``(D,)`` it returns a scalar.
    """
    spec = get_activation(activation)
    z = jnp.matmul(x, W.T) + beta  # (B, H) or (H,)
    return b + spec.forward(z) @ c  # (B,) or scalar


# ---------------------------------------------------------------------------
# Closed-form Laplacian
# ---------------------------------------------------------------------------


def neural_field_laplacian(
    x: Array,
    W: Array,
    beta: Array,
    c: Array,
    activation: str | JaxActivationSpec,
) -> Array:
    """``nabla_x^2 f(x)`` in closed form.

    ``b`` does not enter the Laplacian, so it is not a parameter here.
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no fast-path kernel, cannot use closed-form Laplacian"
        )
    z = jnp.matmul(x, W.T) + beta  # (B, H) or (H,)
    sigma_pp = spec.fastpath(z, 2)  # (B, H) or (H,)
    row_norm_sq = (W * W).sum(axis=-1)  # (H,)
    return (sigma_pp * (c * row_norm_sq)) @ jnp.ones_like(c)


def neural_field_value_and_laplacian(
    x: Array,
    W: Array,
    beta: Array,
    c: Array,
    b: Array | float,
    activation: str | JaxActivationSpec,
) -> tuple[Array, Array]:
    """Returns ``(f(x), nabla_x^2 f(x))`` in one forward pass.

    Reuses the pre-activation ``z`` between the value and the
    Laplacian, so this is strictly cheaper than calling
    :func:`neural_field_value` and :func:`neural_field_laplacian`
    separately.
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(f"activation {spec.name!r} has no fast-path kernel")
    z = jnp.matmul(x, W.T) + beta
    sigma_z = spec.forward(z)
    sigma_pp = spec.fastpath(z, 2)
    f = b + sigma_z @ c
    row_norm_sq = (W * W).sum(axis=-1)
    lap = sigma_pp @ (c * row_norm_sq)
    return f, lap


def neural_field_value_grad_laplacian(
    x: Array,
    W: Array,
    beta: Array,
    c: Array,
    b: Array | float,
    activation: str | JaxActivationSpec,
) -> tuple[Array, Array, Array]:
    """Returns ``(f(x), grad_x f(x), nabla_x^2 f(x))``.

    Closed form for all three. The gradient is needed by every VMC
    local-energy estimator (it shows up in the ``|grad log psi|^2``
    term).
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(f"activation {spec.name!r} has no fast-path kernel")
    z = jnp.matmul(x, W.T) + beta  # (..., H)
    sigma_z = spec.forward(z)  # (..., H)
    sigma_p = spec.fastpath(z, 1)  # (..., H)
    sigma_pp = spec.fastpath(z, 2)  # (..., H)

    f = b + sigma_z @ c  # (..., )
    grad = (sigma_p * c) @ W  # (..., D)
    row_norm_sq = (W * W).sum(axis=-1)  # (H,)
    lap = sigma_pp @ (c * row_norm_sq)  # (..., )
    return f, grad, lap


# ---------------------------------------------------------------------------
# Closed-form full Hessian (enables FermiNet Tier 2/3 integration)
#
# The Laplacian (trace of the Hessian) is enough for the standard
# Slater determinant identity ``nabla^2 det A / det A = trace(A^{-1} L)``
# when each orbital is a one-electron function. As soon as the orbital
# is composed through a coordinate transformation -- for example the
# backflow ``q(r) = r + delta(r)`` used in FermiNet-style ansatzes --
# the chain rule introduces a ``trace(J^T H J)`` term that requires
# the *full* Hessian matrix, not just its trace:
#
#     nabla_r^2 [phi(q(r))]
#         = trace(J^T H_q phi  J) + grad_q phi . nabla_r^2 q
#
# where J = dq/dr is the 3x3 Jacobian of the coordinate map.
#
# For the omnibias one-layer scalar field the Hessian has a clean
# rank-H closed form:
#
#     H_x f = W^T diag(sigma''(z) odot c) W,    z = W x + beta.
#
# That is O(H D^2) FLOPs -- and crucially O(1) calls to sigma'' rather
# than the O(D) calls a jax.hessian sweep would make. For the typical
# FermiNet-class per-electron input (D = 3) the cost is essentially
# the same as the Laplacian; for higher-D heads (D = 16-64 in the
# FermiNet equivariant block) it is the right thing to ship as the
# building block of Tier 2/3 integration.
# ---------------------------------------------------------------------------


def neural_field_hessian(
    x: Array,
    W: Array,
    beta: Array,
    c: Array,
    activation: str | JaxActivationSpec,
) -> Array:
    """``H_x f`` -- the full ``D x D`` Hessian of the one-layer field.

    Returns
    -------
    H : Array of shape ``(..., D, D)``
        The dense Hessian. For a batched ``x`` of shape ``(B, D)`` the
        output has shape ``(B, D, D)``.

    Notes
    -----
    ``b`` does not enter the Hessian, so it is not a parameter here.
    The Hessian is *symmetric* by construction (it is a sum of
    symmetric rank-1 outer products ``sigma''(z_h) * W_h W_h^T``).
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no fast-path kernel, cannot use closed-form Hessian"
        )
    z = jnp.matmul(x, W.T) + beta  # (..., H)
    sigma_pp = spec.fastpath(z, 2)  # (..., H)
    weights = sigma_pp * c  # (..., H)
    # H = W^T diag(weights) W  =  einsum over the hidden dimension.
    return jnp.einsum("...h,hi,hj->...ij", weights, W, W)


def neural_field_value_grad_hessian(
    x: Array,
    W: Array,
    beta: Array,
    c: Array,
    b: Array | float,
    activation: str | JaxActivationSpec,
) -> tuple[Array, Array, Array]:
    """Returns ``(f(x), grad_x f(x), H_x f(x))`` in one fused pass.

    Closed form for all three. Reuses the pre-activation ``z`` so the
    activation derivative tower is touched once per order rather than
    once per output. The Hessian is the building block for
    FermiNet Tier 2/3 integration -- it composes through the
    chain rule for backflow and equivariant-layer coordinate maps.

    Parameters
    ----------
    x : (..., D)        input point(s)
    W : (H, D)          hidden-layer weights
    beta : (H,)         hidden-layer biases
    c : (H,)            output-layer weights
    b : scalar          output-layer bias
    activation : str | JaxActivationSpec
        Must be one of the omnibias fast-path (Riccati-class)
        activations: ``tanh``, ``sigmoid``, ``softplus``, ``gaussian``,
        ``exp``.

    Returns
    -------
    f : (..., )         scalar field value
    grad : (..., D)     gradient
    hessian : (..., D, D)
        Symmetric Hessian matrix.

    See Also
    --------
    neural_field_value_grad_laplacian : same fused call but returning
        the *trace* of the Hessian (the Laplacian), for the single-
        electron Slater ``trace(A^{-1} L)`` path.
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no fast-path kernel, cannot use closed-form Hessian"
        )
    z = jnp.matmul(x, W.T) + beta  # (..., H)
    sigma_z = spec.forward(z)  # (..., H)
    sigma_p = spec.fastpath(z, 1)  # (..., H)
    sigma_pp = spec.fastpath(z, 2)  # (..., H)

    f = b + sigma_z @ c  # (..., )
    grad = (sigma_p * c) @ W  # (..., D)
    weights = sigma_pp * c  # (..., H)
    hessian = jnp.einsum("...h,hi,hj->...ij", weights, W, W)
    return f, grad, hessian


# ---------------------------------------------------------------------------
# Closed-form *polylaplacian* (relativistic-VMC primitive)
#
# For the one-layer scalar field
#
#     f(x) = b + sum_h c_h sigma(W_h . x + beta_h),       x in R^D,
#
# the k-th iterated Laplacian (the "polylaplacian"; also written
# Delta^k f = (nabla^2)^k f) has the closed form
#
#     Delta^k f(x) = sum_h c_h * sigma^{(2k)}(z_h) * ||W_h||^{2k}.
#
# Derivation: f's k-th mixed partial is
#     partial^k f / partial x_{i_1} ... partial x_{i_k}
#         = sum_h c_h sigma^{(k)}(z_h) W_{h, i_1} ... W_{h, i_k},
# so contracting any two indices (i_a, i_b) with delta_{i_a, i_b} gives
# a factor ||W_h||^2 and reduces the derivative order by 2.  Iterating
# k times (i.e., taking Delta^k = trace_{1,2} trace_{3,4} ... ) yields
# the formula above.
#
# Cost: O(B * H), *independent of k*.  This is the structural
# advantage over forward-Laplacian libraries like folx, which compute
# the trace at O(B * D * H) and require nesting for k >= 2 (giving
# O(B * D^{k-1} * H)).  For k=2 (the relativistic mass-velocity
# correction <p^4> = <Delta^2 psi>) on a D=30 (10-electron) molecule
# the omnibias / folx cost ratio is already ~30x; for k=3 it is ~900x.
#
# Pre-requisite: the activation's fastpath kernel must support order
# 2k.  All Riccati-class activations (sigmoid, tanh, softplus,
# gaussian, exp) support arbitrary order via their Eulerian / Legendre
# / Hermite / Stirling polynomial recursions in omnibias/fastpath/.
# ---------------------------------------------------------------------------


def neural_field_polylaplacian(
    x: Array,
    W: Array,
    beta: Array,
    c: Array,
    activation: str | JaxActivationSpec,
    k: int,
) -> Array:
    """``Delta^k f(x) = (nabla^2)^k f(x)`` in closed form.

    Computes the k-th iterated Laplacian of the one-layer field
    ``f(x) = b + sum_h c_h sigma(W_h . x + beta_h)``.  ``b`` cancels
    for k >= 1, so it is not a parameter here.

    Parameters
    ----------
    x : (..., D)
    W : (H, D)
    beta : (H,)
    c : (H,)
    activation : str | JaxActivationSpec
        Must be a Riccati-class activation with ``fastpath(z, 2k)``
        defined (i.e., ``sigmoid``, ``tanh``, ``softplus``, ``gaussian``,
        ``exp``).
    k : int >= 1
        Polylaplacian order.  k=1 reproduces ``neural_field_laplacian``;
        k=2 is the relativistic mass-velocity correction operator
        :math:`\\hat p^4 = (\\nabla^2)^2`; k=3 enters the third-order
        Foldy-Wouthuysen expansion.

    Returns
    -------
    Array of shape ``(...,)`` -- the k-th iterated Laplacian.

    Notes
    -----
    Memory cost: ``O(B * H)``, *independent of k*.  Time cost: one
    forward pass through ``sigma^{(2k)}`` plus one ``(B,H)`` reduction.
    Compare to folx-nested ``Delta^k`` which is ``O(B * D^{k-1} * H)``.
    """
    if k < 1:
        raise ValueError(f"polylaplacian order k must be >= 1, got {k}")
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {spec.name!r} has no fast-path kernel, "
            "cannot use closed-form polylaplacian"
        )
    z = jnp.matmul(x, W.T) + beta  # (..., H)
    sigma_2k = spec.fastpath(z, 2 * k)  # (..., H)
    row_norm_sq = (W * W).sum(axis=-1)  # (H,)
    row_norm_2k = row_norm_sq**k  # (H,)
    return sigma_2k @ (c * row_norm_2k)


def neural_field_value_and_polylaplacian(
    x: Array,
    W: Array,
    beta: Array,
    c: Array,
    b: Array | float,
    activation: str | JaxActivationSpec,
    k: int,
) -> tuple[Array, Array]:
    """Returns ``(f(x), Delta^k f(x))`` in one fused pass.

    Useful for VMC local-energy estimators that need both the
    wavefunction value (for ``log|psi|`` and importance sampling)
    and ``Delta^k psi / psi`` (for the local kinetic-energy term).
    """
    if k < 1:
        raise ValueError(f"polylaplacian order k must be >= 1, got {k}")
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(f"activation {spec.name!r} has no fast-path kernel")
    z = jnp.matmul(x, W.T) + beta
    sigma_z = spec.forward(z)
    sigma_2k = spec.fastpath(z, 2 * k)
    f = b + sigma_z @ c
    row_norm_sq = (W * W).sum(axis=-1)
    row_norm_2k = row_norm_sq**k
    poly_lap = sigma_2k @ (c * row_norm_2k)
    return f, poly_lap


def neural_field_local_p4_over_psi(
    x: Array,
    W: Array,
    beta: Array,
    c: Array,
    b: Array | float,
    activation: str | JaxActivationSpec,
) -> Array:
    """Relativistic mass-velocity local-energy operator p^4 psi / psi.

    For a wavefunction ``psi(x) = f(x)`` represented directly by the
    one-layer field (no log-magnitude transform), the mass-velocity
    local energy is

        L_rel(x) = p^4 psi(x) / psi(x) = Delta^2 f / f.

    This is the operator that enters the relativistic Foldy-Wouthuysen
    Hamiltonian as

        H_rel = - p^4 / (8 m^3 c^2)

    (in CGS-Gaussian / Hartree atomic units, where m = 1 and the
    speed of light c ~ 137.036).  The cost of this estimator on a
    K=H-neuron omnibias field is O(B * H), independent of D --
    compare to folx-nested which would be O(B * D * H).

    For wavefunctions parameterised as ``psi = exp(L)`` with L a
    neural log-magnitude (the standard FermiNet pattern), see
    ``neural_field_p4_chain_rule_log`` (roadmap to-do; the chain
    rule has six terms in 1D and twenty terms in 3D).
    """
    f, p4_psi = neural_field_value_and_polylaplacian(x, W, beta, c, b, activation, k=2)
    return p4_psi / f


__all__ = [
    "neural_field_hessian",
    "neural_field_laplacian",
    "neural_field_local_p4_over_psi",
    "neural_field_polylaplacian",
    "neural_field_value",
    "neural_field_value_and_laplacian",
    "neural_field_value_and_polylaplacian",
    "neural_field_value_grad_hessian",
    "neural_field_value_grad_laplacian",
]
