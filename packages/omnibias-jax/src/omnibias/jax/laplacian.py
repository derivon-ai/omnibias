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

from collections.abc import Sequence
from typing import TYPE_CHECKING

from omnibias.core.contraction import (
    DEFAULT_SUPPORT_BUDGET,
    Mode,
    PolylaplacianReport,
    polylaplacian_normalizer,
    polylaplacian_support_terms,
    select_mode,
)
from omnibias.core.multi_index import index_position, multi_index_factorial
from omnibias.core.verified.sampled import ConcentrationReport, hoeffding_enclosure
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.jet import jet_to_tower, mlp_jet
from omnibias.jax.jet_mv import mlp_jet_mv

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | str | None]

_PIECEWISE_AE_SECOND_ORDER_ZERO = frozenset(
    {
        "relu",
        "leaky_relu",
        "relu6",
        "hardtanh",
        "hardsigmoid",
        "hardshrink",
        "threshold",
        "abs",
        "sign",
        "step",
        "softshrink",
    }
)


def _check_piecewise_second_order(
    name: str, *, allow_almost_everywhere: bool
) -> None:
    if allow_almost_everywhere:
        return
    if name in _PIECEWISE_AE_SECOND_ORDER_ZERO:
        raise ValueError(
            f"activation {name!r} has an a.e.-zero second derivative; the "
            "deep-field Laplacian would return a confident zero. Pass "
            "allow_almost_everywhere=True to opt in explicitly."
        )

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


# ---------------------------------------------------------------------------
# Deep-network fast lane: no C(D+2k,D) ceiling at any depth or dimension
#
# Every kernel above is exact but structurally *single hidden layer*: the
# pre-activation z = W x + b is affine in x, so all higher x-derivatives of z
# vanish and Faa di Bruno collapses to one term.  A deep network has no such
# collapse; the naive fallback is the full multivariate jet
# (omnibias.jax.jet_mv.mlp_jet_mv), which materialises every mixed partial to
# total order N and is guarded by omnibias.core.multi_index's
# comb(dim + order, dim) budget -- the wrong algorithm for an operator that
# only ever wants one contraction (the trace of the Hessian, or its k-th
# iterate).
#
# The ``deep_field_*`` family below computes that contraction directly, at
# every input dimension D, in three tiers (see omnibias.core.contraction for
# the combinatorics and the full derivation):
#
#   Tier A (k=1, deep_field_laplacian / deep_field_value_grad_laplacian):
#     the classic forward-Laplacian recursion.  Per layer, with u = W a + b
#     then a' = sigma(u):
#
#         J_u = J_a @ W.T
#         L_u = L_a @ W.T
#         L_a' = sigma''(u) * (J_u**2).sum(axis=-2) + sigma'(u) * L_u
#         J_a' = sigma'(u)[:, None, :] * J_u
#
#     seeded with J = I, L = 0 (the first layer needs no explicit D x D
#     identity: J_u there is simply W1.T broadcast, so the identity multiply
#     is never materialised and the O(B*D^2) it would cost is never paid).
#     Every sigma', sigma'' comes from the closed-form fastpath tower -- no
#     autodiff through the activation at all, unlike folx.  Cost O(B*H*D),
#     the same order as the forward pass itself, with *no* combinatorial term.
#
#   Tier B (k>=2, mode="support"): Delta^k f = k! * sum_{|a|=k} D^(2a) f / a!
#     (the multinomial expansion of (sum_i d_i^2)^k). Every surviving `a` has
#     at most k nonzero entries; grouping by support lets each group be read
#     off one *local* multivariate jet restricted to that support (the first
#     layer's weight columns outside the support are dropped, folded into the
#     bias via W @ x0), costing comb(s + 2k, s) for a support of size s <= k
#     -- bounded by k, never by D. Exact while the number of support sets
#     (omnibias.core.contraction.support_jet_count) fits a budget.
#
#   Tier C (k>=2, mode="estimator"): beyond that budget, Delta^k f is a
#     normalised expectation of a directional 2k-th derivative over the unit
#     sphere (the spherical identity in
#     omnibias.core.contraction.polylaplacian_normalizer), estimated by
#     sampling directions through the existing directional omnibias.jax.jet.mlp_jet
#     -- unbiased in expectation, and its jet size never depends on D.
#
# Honest scope: Tier A is exact and ceiling-free, full stop. Tier B is exact
# while it fits its budget. Tier C is exact *in expectation*, with a reported
# standard error and a probabilistic (Hoeffding) enclosure -- never a
# omnibias.core.verified-grade sound one. The claim is "exact and ceiling-free
# for the Laplacian; exact-or-enclosed at any dimension for Delta^k", never
# "O(1) at arbitrary order" (see docs/honesty.md).
# ---------------------------------------------------------------------------


def deep_field_value_grad_laplacian(
    x: Array,
    layers: Sequence[LayerSpec],
    *,
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array, Array]:
    r"""``(f(x), grad_x f(x), nabla_x^2 f(x))`` for a deep MLP, no ceiling.

    Forward-Laplacian recursion (Tier A): carries the Jacobian ``J`` of the
    running activation w.r.t. the *original* input and the scalar Laplacian
    ``L`` of each unit through every layer, using the closed-form
    ``sigma'``/``sigma''`` fastpath (no autodiff through the activation).
    Memory is ``O(B * H * D)`` -- the same order as the forward pass -- with
    no ``comb(D + 2k, D)`` combinatorial term at any ``D``.

    Parameters
    ----------
    x : Array of shape ``(D,)`` or ``(B, D)``.
    layers : sequence of ``(W, b, spec)``
        As consumed by :func:`omnibias.jax.jet.mlp_jet` / :func:`omnibias.jax.jet_mv.mlp_jet_mv`.
        ``spec=None`` is a pure affine layer (used for the readout); every
        other layer computes ``sigma(W a + b)``.

    Returns
    -------
    value : Array of shape ``(C,)`` or ``(B, C)``
    grad : Array of shape ``(D, C)`` or ``(B, D, C)``
    laplacian : Array of shape ``(C,)`` or ``(B, C)``

    At ``depth=1`` (one hidden layer plus an affine readout) this reproduces
    :func:`neural_field_value_grad_laplacian` to floating-point round-off --
    the single-layer field is the ``D=1`` collapse of the deep recursion.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    x = jnp.asarray(x)
    batched = x.ndim > 1
    a = x if batched else x[None, :]
    B, dim = a.shape
    L = jnp.zeros_like(a)  # (B, D); Laplacian of the identity map is 0.
    J: Array | None = None  # sentinel for the identity Jacobian (never materialised)
    for W, b, spec in layers:
        W = jnp.asarray(W)
        u = jnp.matmul(a, W.T)
        if b is not None:
            b_arr = jnp.asarray(b)
            u = u + (b_arr if b_arr.ndim == 1 else b_arr)
        first_layer = J is None
        if first_layer:
            # First layer: (identity Jacobian) @ W.T collapses to W.T broadcast
            # over the batch, so the O(B*D^2) identity is never formed.
            J_u = jnp.broadcast_to(W.T[None, :, :], (B, dim, W.shape[0]))
        else:
            J_u = jnp.tensordot(J, W, axes=([-1], [-1]))  # (B, D, H_out)
        L_u = jnp.matmul(L, W.T)  # (B, H_out)
        if spec is None:
            a, J, L = u, J_u, L_u
            continue
        resolved = get_activation(spec)
        if resolved.fastpath is None:
            raise ValueError(
                f"activation {resolved.name!r} has no fast-path kernel, cannot "
                "use the deep-field Laplacian"
            )
        _check_piecewise_second_order(
            resolved.name, allow_almost_everywhere=allow_almost_everywhere
        )
        try:
            sigma_p = resolved.fastpath(u, 1)
            sigma_pp = resolved.fastpath(u, 2)
        except NotImplementedError as exc:
            raise ValueError(
                f"activation {resolved.name!r} fastpath does not support order "
                "2, required for the deep-field Laplacian"
            ) from exc
        a = resolved.forward(u)
        J = sigma_p[:, None, :] * J_u
        if first_layer:
            row_norm_sq = jnp.sum(W * W, axis=-1)  # (H,) == ||W_h||^2, batch-free
            lap_contrib = sigma_pp * row_norm_sq
        else:
            lap_contrib = sigma_pp * jnp.sum(J_u * J_u, axis=1)
        L = lap_contrib + sigma_p * L_u
    assert J is not None  # every loop iteration above assigns J; layers is non-empty
    value = a if batched else a[0]
    grad = J if batched else J[0]
    lap = L if batched else L[0]
    return value, grad, lap


def deep_field_laplacian(
    x: Array, layers: Sequence[LayerSpec], *, allow_almost_everywhere: bool = False
) -> Array:
    r"""``nabla_x^2 f(x)`` for a deep MLP, no ceiling at any ``D`` (Tier A).

    Thin wrapper over :func:`deep_field_value_grad_laplacian` discarding the
    value and gradient. See that function for the recursion and the
    depth-1 collapse onto :func:`neural_field_laplacian`.
    """
    _, _, lap = deep_field_value_grad_laplacian(
        x, layers, allow_almost_everywhere=allow_almost_everywhere
    )
    return lap


def restrict_first_layer(
    layers: Sequence[LayerSpec], x0: Array, support: tuple[int, ...]
) -> list[LayerSpec]:
    """Restrict ``layers`` to the input axes in ``support``, folding the rest into the bias.

    If ``x = x0 + E_support @ y`` (``y`` living on the ``support`` axes only,
    everything else pinned at ``x0``), then the first affine layer
    ``z = W x + b`` becomes ``z = W[:, support] @ y + (W @ x0 + b)`` -- exact,
    and the local weight matrix has only ``len(support)`` columns. Used by
    Tier B's per-support local jet and, in
    :mod:`omnibias.pinn.jax.fields.jet_mlp`, to isolate the spatial axes of a
    field that also carries a non-spatial (e.g. time) coordinate.
    """
    W0, b0, spec0 = layers[0]
    W0 = jnp.asarray(W0)
    cols = jnp.asarray(support)
    W_local = jnp.take(W0, cols, axis=1)
    shift = jnp.matmul(W0, jnp.asarray(x0))
    b_local = shift if b0 is None else shift + jnp.asarray(b0)
    return [(W_local, b_local, spec0), *layers[1:]]


def _polylaplacian_support(x: Array, layers: Sequence[LayerSpec], k: int) -> Array:
    """Tier B: exact ``Delta^k`` via the support-grouped local multivariate jet."""
    dim = x.shape[-1]
    terms = polylaplacian_support_terms(dim, k)

    def one_point(xi: Array) -> Array:
        total: Array | None = None
        for support, local_terms in terms:
            local_layers = restrict_first_layer(layers, xi, support)
            s = len(support)
            local_jet = mlp_jet_mv(jnp.zeros((s,), dtype=xi.dtype), local_layers, 2 * k)
            pos = index_position(s, 2 * k)
            for a_local, coeff in local_terms:
                alpha = tuple(2 * v for v in a_local)
                term = local_jet[pos[alpha]] * (multi_index_factorial(alpha) * coeff)
                total = term if total is None else total + term
        assert total is not None
        return total

    return jax.vmap(one_point)(x)


def _polylaplacian_estimator(
    x: Array,
    layers: Sequence[LayerSpec],
    k: int,
    *,
    n_directions: int,
    seed: int,
    key: Array | None = None,
    return_samples: bool = False,
) -> Array | tuple[Array, Array]:
    """Tier C: unbiased sphere-average estimator via the directional jet."""
    dim = x.shape[-1]
    rng_key = key if key is not None else jax.random.PRNGKey(seed)
    g = jax.random.normal(rng_key, (n_directions, dim), dtype=x.dtype)
    v = g / jnp.linalg.norm(g, axis=-1, keepdims=True)
    normalizer = polylaplacian_normalizer(dim, k)

    def one_point(xi: Array) -> Array | tuple[Array, Array]:
        def directional_2k(vi: Array) -> Array:
            jet = mlp_jet(xi, vi, layers, 2 * k)
            return jet_to_tower(jet)[2 * k]

        samples = jax.vmap(directional_2k)(v)  # (n_directions, C)
        scaled = normalizer * samples
        mean = jnp.mean(scaled, axis=0)
        if return_samples:
            return mean, scaled
        return mean

    if return_samples:
        means, scaled = jax.vmap(one_point, out_axes=(0, 0))(x)
        return means, scaled
    return jax.vmap(one_point)(x)


def _resolve_polylaplacian_mode(
    dim: int, k: int, mode: str, budget: int | None
) -> tuple[Mode, int]:
    resolved_budget = budget if budget is not None else DEFAULT_SUPPORT_BUDGET
    if k == 1:
        return "forward", resolved_budget
    if mode == "auto":
        return select_mode(dim, k, budget=resolved_budget), resolved_budget
    if mode not in ("forward", "support", "estimator"):
        raise ValueError(f"unknown mode {mode!r}")
    return mode, resolved_budget  # type: ignore[return-value]


def _tier_c_concentration(scaled_samples: Array) -> ConcentrationReport:
    """Build a Hoeffding report from the first batch point, first output."""
    flat = [float(x) for x in scaled_samples[0, :, 0]]
    lo = min(flat)
    hi = max(flat)
    if lo == hi:
        hi = lo + 1.0
    return hoeffding_enclosure(flat, value_range=(lo, hi))


def deep_field_polylaplacian(
    x: Array,
    layers: Sequence[LayerSpec],
    k: int,
    *,
    mode: str = "auto",
    n_directions: int = 256,
    budget: int | None = None,
    seed: int = 0,
    key: Array | None = None,
) -> Array:
    r"""``Delta^k f(x)`` for a deep MLP at any input dimension ``D``.

    ``k=1`` always routes to :func:`deep_field_laplacian` (Tier A: exact, no
    ceiling). For ``k >= 2``, ``mode="auto"`` (the default) picks Tier B
    (``"support"``, exact) while :func:`omnibias.core.contraction.select_mode`
    finds the number of support sets within ``budget``
    (:data:`omnibias.core.contraction.DEFAULT_SUPPORT_BUDGET` when
    ``budget`` is ``None``), else Tier C (``"estimator"``, exact in
    expectation over ``n_directions`` sampled unit directions).

    Parameters
    ----------
    x : Array of shape ``(D,)`` or ``(B, D)``.
    layers : sequence of ``(W, b, spec)``, as for :func:`deep_field_laplacian`.
    k : int >= 1
        Poly-Laplacian order.
    mode : {"auto", "support", "estimator"}
        Force a tier for ``k >= 2`` instead of the budget-driven choice.
    n_directions : int
        Sample count for Tier C (ignored otherwise).
    budget : int, optional
        Override :data:`omnibias.core.contraction.DEFAULT_SUPPORT_BUDGET`
        for the Tier B/C decision.
    seed : int
        PRNG seed for Tier C direction sampling when ``key`` is ``None``.
    key : Array, optional
        JAX PRNG key for Tier C; when supplied, ``seed`` is ignored and the
        caller can vary the key per step without forcing a ``jit`` retrace on
        ``seed``.

    Returns
    -------
    Array of shape ``(C,)`` or ``(B, C)``.

    Notes
    -----
    Tier B is exact to floating-point round-off; Tier C is exact only *in
    expectation*. Use :func:`deep_field_polylaplacian_with_report` for the
    resolved tier, sample standard error, and a probabilistic enclosure.
    Never claim "O(1) at arbitrary order" for Tier C: its cost is
    ``O(n_directions)`` regardless of ``D``, but it is a Monte Carlo
    estimator, not a closed form.
    """
    out, _ = deep_field_polylaplacian_with_report(
        x,
        layers,
        k,
        mode=mode,
        n_directions=n_directions,
        budget=budget,
        seed=seed,
        key=key,
    )
    return out


def deep_field_polylaplacian_with_report(
    x: Array,
    layers: Sequence[LayerSpec],
    k: int,
    *,
    mode: str = "auto",
    n_directions: int = 256,
    budget: int | None = None,
    seed: int = 0,
    key: Array | None = None,
) -> tuple[Array, PolylaplacianReport]:
    r"""``Delta^k f(x)`` plus metadata about which tier ran.

    Returns the same point estimate as :func:`deep_field_polylaplacian` and a
    :class:`~omnibias.core.contraction.PolylaplacianReport` carrying the
    resolved mode, ``D``, ``k``, the budget, and for Tier C the direction
    count, sample standard error, and a :class:`~omnibias.core.verified.sampled.ConcentrationReport`
    from :func:`~omnibias.core.verified.sampled.hoeffding_enclosure`.
    """
    if k < 1:
        raise ValueError(f"polylaplacian order k must be >= 1, got {k}")
    x = jnp.asarray(x)
    batched = x.ndim > 1
    x2 = x if batched else x[None, :]
    dim = x2.shape[-1]
    resolved_mode, resolved_budget = _resolve_polylaplacian_mode(dim, k, mode, budget)
    concentration = None
    if k == 1:
        out = deep_field_laplacian(x2, layers)
    elif resolved_mode == "forward":
        raise ValueError(
            "mode='forward' only applies to k=1 (the Laplacian); use "
            "'support' or 'estimator' for k >= 2"
        )
    elif resolved_mode == "support":
        out = _polylaplacian_support(x2, layers, k)
    elif resolved_mode == "estimator":
        out, scaled = _polylaplacian_estimator(
            x2,
            layers,
            k,
            n_directions=n_directions,
            seed=seed,
            key=key,
            return_samples=True,
        )
        concentration = _tier_c_concentration(scaled)
    else:
        raise ValueError(f"unknown mode {resolved_mode!r}")
    report = PolylaplacianReport(
        mode=resolved_mode,
        dim=dim,
        k=k,
        budget=resolved_budget,
        n_directions=n_directions if resolved_mode == "estimator" else None,
        concentration=concentration,
    )
    return (out if batched else out[0]), report


__all__ = [
    "deep_field_laplacian",
    "deep_field_polylaplacian",
    "deep_field_polylaplacian_with_report",
    "deep_field_value_grad_laplacian",
    "neural_field_hessian",
    "neural_field_laplacian",
    "neural_field_local_p4_over_psi",
    "neural_field_polylaplacian",
    "neural_field_value",
    "neural_field_value_and_laplacian",
    "neural_field_value_and_polylaplacian",
    "neural_field_value_grad_hessian",
    "neural_field_value_grad_laplacian",
    "restrict_first_layer",
]
