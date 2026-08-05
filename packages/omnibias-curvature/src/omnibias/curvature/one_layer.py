# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form parameter Gradient / Hessian / Fisher for one-layer fields.

Symbol map
----------

==================  ============  =================================================
Tensor              Shape         Meaning
==================  ============  =================================================
``x``               ``(B, D)``    input batch
``W``               ``(H, D)``    hidden-layer weights
``beta``            ``(H,)``      hidden-layer bias
``c``               ``(H,)``      output-layer weights
``b``               ``()``        output-layer bias
==================  ============  =================================================

Parameter vector layout (a flat ``(P,)`` array of length
``P = 1 + H + H + H*D``):

* index ``[0]``                            : ``b``
* indices ``[1 .. 1 + H)``                 : ``c``
* indices ``[1 + H .. 1 + 2*H)``           : ``beta``
* indices ``[1 + 2*H .. 1 + 2*H + H*D)``   : ``W`` flattened row-major

The flat layout matches :func:`jax.tree_util.tree_leaves` order on a
``(b, c, beta, W)`` tuple of (scalar, ``(H,)``, ``(H,)``, ``(H, D)``)
arrays, so consumers can call :func:`pack_params` / :func:`unpack_params`
to interconvert with their own pytree.

Per-sample gradient
-------------------

For :math:`z_n = W \cdot x_n + \beta`:

* :math:`\partial f / \partial b = 1`
* :math:`\partial f / \partial c_h = \sigma(z_{n,h})`
* :math:`\partial f / \partial \beta_h = c_h\,\sigma'(z_{n,h})`
* :math:`\partial f / \partial W_{h,j} = c_h\,\sigma'(z_{n,h})\,x_{n,j}`

Per-sample Hessian
------------------

Block-diagonal in :math:`h` (parameters for different hidden units do
not interact). Non-zero entries within a hidden-unit block:

* :math:`\partial^2 f / \partial c_h\,\partial \beta_h = \sigma'(z_{n,h})`
* :math:`\partial^2 f / \partial c_h\,\partial W_{h,j} = \sigma'(z_{n,h})\,x_{n,j}`
* :math:`\partial^2 f / \partial \beta_h^2 = c_h\,\sigma''(z_{n,h})`
* :math:`\partial^2 f / \partial \beta_h\,\partial W_{h,j} = c_h\,\sigma''(z_{n,h})\,x_{n,j}`
* :math:`\partial^2 f / \partial W_{h,j}\,\partial W_{h,j'} = c_h\,\sigma''(z_{n,h})\,x_{n,j}\,x_{n,j'}`

All others (including all interactions with ``b``) are zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.curvature.regularize import regularized_solve
from omnibias.jax.activations import get_activation

# ---------------------------------------------------------------------------
# Pack / unpack helpers
# ---------------------------------------------------------------------------


def pack_params(b: Array, c: Array, beta: Array, W: Array) -> Array:
    """Pack ``(b, c, beta, W)`` into a flat parameter vector.

    Order matches :func:`unpack_params` and the gradient / Hessian
    routines in this module.
    """
    return jnp.concatenate([
        jnp.asarray(b).reshape(-1),
        c.reshape(-1),
        beta.reshape(-1),
        W.reshape(-1),
    ])


def unpack_params(
    theta: Array, H: int, D: int,
) -> tuple[Array, Array, Array, Array]:
    """Inverse of :func:`pack_params` for the given hidden / input sizes."""
    if theta.shape[-1] != 1 + 2 * H + H * D:
        raise ValueError(
            f"unpack_params: expected last dim = {1 + 2*H + H*D} (= "
            f"1 + 2*H + H*D for H={H}, D={D}), got {theta.shape[-1]}"
        )
    b = theta[..., 0]
    c = theta[..., 1:1 + H]
    beta = theta[..., 1 + H:1 + 2 * H]
    W = theta[..., 1 + 2 * H:].reshape(theta.shape[:-1] + (H, D))
    return b, c, beta, W


# ---------------------------------------------------------------------------
# Per-sample gradient / Hessian
# ---------------------------------------------------------------------------


def _per_sample_blocks(
    x: Array, W: Array, beta: Array, c: Array, activation: str,
):
    """Compute the per-sample ``(sigma_z, sigma_p, sigma_pp)`` triplet."""
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {activation!r} has no fast-path kernel; required "
            "for the closed-form parameter Hessian."
        )
    z = W @ x + beta              # (H,)
    sigma_z = spec.forward(z)     # (H,)
    sigma_p = spec.fastpath(z, 1) # (H,)
    sigma_pp = spec.fastpath(z, 2)# (H,)
    return sigma_z, sigma_p, sigma_pp


def one_layer_param_grad(
    x: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
) -> Array:
    r"""Flat per-sample gradient :math:`\nabla_\theta f(x)` of length P.

    See module docstring for the parameter vector layout.
    """
    del b  # not used in gradient
    sigma_z, sigma_p, _ = _per_sample_blocks(x, W, beta, c, activation)
    grad_b = jnp.asarray(1.0)
    grad_c = sigma_z                                # (H,)
    grad_beta = c * sigma_p                         # (H,)
    grad_W = jnp.outer(c * sigma_p, x)              # (H, D)
    return jnp.concatenate([
        grad_b.reshape(-1),
        grad_c.reshape(-1),
        grad_beta.reshape(-1),
        grad_W.reshape(-1),
    ])


def one_layer_param_hessian(
    x: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
) -> Array:
    r"""Full per-sample parameter Hessian :math:`\nabla^2_\theta f(x)`.

    Returns a dense :math:`(P, P)` matrix. Mostly zeros — the non-zero
    block structure is described in the module docstring. For large
    :math:`H` and :math:`D`, use the Gauss-Newton Fisher
    (:func:`mse_gauss_newton_fisher`) or the Kronecker factors
    (:func:`kfac_kron_factors`) instead of materialising the full
    Hessian.

    The (residual-free) full Hessian is useful as a correctness oracle
    against :func:`jax.hessian`, and as the building block for a
    second-order optimiser that does not approximate via Gauss-Newton.

    Implementation note: ``H`` and ``D`` are static at JIT-trace time so
    the assembly uses a Python ``for h in range(H)`` loop that is
    unrolled into ``H`` scatter ops. This is fine for ``H ≤ 128``; for
    larger one-layer fields prefer the vectorised
    :func:`mse_gauss_newton_fisher` (no Hessian assembly) or
    :func:`kfac_kron_factors` (closed-form Kronecker).
    """
    _, sigma_p, sigma_pp = _per_sample_blocks(x, W, beta, c, activation)
    Hh = int(W.shape[0])
    Dd = int(W.shape[1])
    P = 1 + 2 * Hh + Hh * Dd

    Hess = jnp.zeros((P, P))

    # Index map (all concrete):
    #   b @ 0
    #   c @ [1, 1+H)
    #   β @ [1+H, 1+2H)
    #   W flattened row-major @ [1+2H, P)

    idx_c = jnp.arange(1, 1 + Hh)
    idx_beta = jnp.arange(1 + Hh, 1 + 2 * Hh)

    # c[h] ↔ β[h]: σ'(z_h)  (diagonal in h).
    Hess = Hess.at[idx_c, idx_beta].set(sigma_p)
    Hess = Hess.at[idx_beta, idx_c].set(sigma_p)

    # β[h] ↔ β[h]: c_h σ''(z_h).
    Hess = Hess.at[idx_beta, idx_beta].set(sigma_pp * c)

    # c[h] ↔ W[h, j] and β[h] ↔ W[h, j]: vectorised scatter.
    # W[h, j] flat index = 1 + 2H + h*D + j.
    idx_W_flat = jnp.arange(1 + 2 * Hh, P)                # (H*D,)
    idx_c_rep = jnp.repeat(idx_c, Dd)                     # (H*D,) — c[h] repeated D times
    idx_beta_rep = jnp.repeat(idx_beta, Dd)               # (H*D,) — β[h] repeated D times
    val_c_W = (sigma_p[:, None] * x[None, :]).reshape(-1)  # (H*D,)
    val_beta_W = ((sigma_pp * c)[:, None] * x[None, :]).reshape(-1)  # (H*D,)

    Hess = Hess.at[idx_c_rep, idx_W_flat].set(val_c_W)
    Hess = Hess.at[idx_W_flat, idx_c_rep].set(val_c_W)
    Hess = Hess.at[idx_beta_rep, idx_W_flat].set(val_beta_W)
    Hess = Hess.at[idx_W_flat, idx_beta_rep].set(val_beta_W)

    # W[h, :] ↔ W[h, :]: c_h σ''(z_h) x x^T, block-diagonal in h.
    # Loop over the static dimension H (Python-time unroll).
    sppc = sigma_pp * c                                   # (H,)
    xxT = jnp.outer(x, x)                                 # (D, D)
    for h in range(Hh):
        W_start = 1 + 2 * Hh + h * Dd
        Hess = jax.lax.dynamic_update_slice(
            Hess, sppc[h] * xxT,
            (W_start, W_start),
        )
    return Hess


# ---------------------------------------------------------------------------
# Gauss-Newton Fisher and Newton step (MSE loss)
# ---------------------------------------------------------------------------


def mse_gauss_newton_fisher(
    X: Array, Y: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
) -> tuple[Array, Array]:
    r"""Gauss-Newton Fisher and gradient for MSE loss
    :math:`L = \tfrac{1}{B}\sum_n (f(x_n) - y_n)^2`.

    Returns
    -------
    F : ``(P, P)``
        :math:`F = \tfrac{2}{B}\sum_n g_n g_n^T` — the positive-definite
        Gauss-Newton approximation to the loss Hessian. Equivalent to
        the Fisher information of a Gaussian observation model with
        unit variance.
    g_loss : ``(P,)``
        Gradient of the loss :math:`\nabla_\theta L = \tfrac{2}{B}\sum_n r_n g_n`,
        where :math:`r_n = f_\theta(x_n) - y_n`.

    For large ``B``, ``F`` is :math:`(P, P)` so memory is :math:`O(P^2)`.
    Use :func:`kfac_kron_factors` for the Kronecker-factored variant.
    """
    spec = get_activation(activation)
    B = X.shape[0]

    def per_sample(x_n):
        sigma_z = spec.forward(W @ x_n + beta)
        f = b + sigma_z @ c
        g = one_layer_param_grad(x_n, W, beta, c, b, activation)
        return f, g

    f_vals, gs = jax.vmap(per_sample)(X)            # (B,), (B, P)
    r = f_vals - Y                                  # (B,)
    F = (2.0 / B) * (gs.T @ gs)                     # (P, P)
    g_loss = (2.0 / B) * (gs.T @ r)                 # (P,)
    return F, g_loss


def mse_newton_step(
    X: Array, Y: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
    learning_rate: float = 1.0,
    damping: float = 1e-4,
) -> tuple[Array, Array, Array, Array]:
    r"""One Gauss-Newton step on the MSE loss.

    Updates ``(b, c, beta, W)`` via

    .. math::

        \theta \;\leftarrow\; \theta - \eta\,(F + \lambda I)^{-1}\,\nabla L

    with the closed-form Gauss-Newton Fisher ``F`` from
    :func:`mse_gauss_newton_fisher`.

    Returns
    -------
    (b_new, c_new, beta_new, W_new) : same shapes as the inputs.
    """
    F, g_loss = mse_gauss_newton_fisher(X, Y, W, beta, c, b, activation)
    # Delegates to the shared eps -> 0 collapse kernel; bit-identical at this damping.
    delta = regularized_solve(F, g_loss, eps=damping)
    theta = pack_params(b, c, beta, W)
    theta_new = theta - learning_rate * delta
    return unpack_params(theta_new, H=W.shape[0], D=W.shape[1])


# ---------------------------------------------------------------------------
# KFAC Kronecker factors (closed-form)
# ---------------------------------------------------------------------------


def kfac_kron_factors(
    X: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
) -> tuple[Array, Array]:
    r"""Closed-form KFAC :math:`(A, G)` factors for the hidden ``W`` block.

    For the linear layer :math:`z_n = W \cdot x_n + \beta` with downstream
    output :math:`f_n = b + \sigma(z_n) \cdot c`, the KFAC approximation
    to the parameter Fisher of ``W`` reads
    :math:`F_W \approx A \otimes G` with

    * :math:`A = \tfrac{1}{B}\sum_n x_n x_n^T \in \mathbb R^{D \times D}`
      — the input second moment.
    * :math:`G = \tfrac{1}{B}\sum_n (c \odot \sigma'(z_n))(c \odot \sigma'(z_n))^T \in \mathbb R^{H \times H}`
      — the pre-activation gradient covariance.

    Closed-form via :math:`\sigma'(z_n)` from the omnibias activation
    fast-path: no autograd backward needed. This is the building block
    that lifts the omnibias Hessian primitive into a KFAC-style
    preconditioner usable with a FermiNet trainer.

    Notes
    -----
    The Kronecker assumption :math:`F_W \approx A \otimes G` is an
    approximation; the closed-form bit here is the *exact* construction
    of :math:`G` (via :math:`\sigma'(z)`), not the Kronecker
    factorisation itself. The Kronecker assumption typically incurs
    a few-percent approximation error per layer (Martens & Grosse 2015)
    but is the standard KFAC trade-off.
    """
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise ValueError(
            f"activation {activation!r} has no fast-path kernel."
        )
    del b
    B = X.shape[0]
    Z = X @ W.T + beta[None, :]                # (B, H)
    sigma_p = spec.fastpath(Z, 1)              # (B, H)
    grad_pre = sigma_p * c[None, :]            # (B, H) (c ⊙ σ'(z_n))
    A = (X.T @ X) / B                          # (D, D)
    G = (grad_pre.T @ grad_pre) / B            # (H, H)
    return A, G


__all__ = [
    "kfac_kron_factors",
    "mse_gauss_newton_fisher",
    "mse_newton_step",
    "one_layer_param_grad",
    "one_layer_param_hessian",
    "pack_params",
    "unpack_params",
]
