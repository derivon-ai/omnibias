# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact-curvature sharpness-aware training for one-layer Riccati fields.

Motivation -- "SAM done right"
------------------------------

Sharpness-Aware Minimisation (Foret et al. 2021) replaces the training
loss :math:`L(\theta)` by its worst case inside an
:math:`\ell_2`-ball of radius :math:`\rho`,

.. math::

    L^{\mathrm{SAM}}_\rho(\theta)
        \;=\; \max_{\lVert\varepsilon\rVert_2 \le \rho} L(\theta+\varepsilon),

and approximates the inner maximisation by a *single gradient-ascent
step* :math:`\varepsilon^\star \approx \rho\,\nabla L / \lVert\nabla L\rVert`.
That approximation keeps only the **linear** term of the Taylor
expansion and needs a second forward/backward pass; it never sees the
curvature that actually makes a basin sharp or flat.

For the one-hidden-layer Riccati field
:math:`f_\theta(x) = b + \sum_h c_h\,\sigma(W_h\!\cdot x + \beta_h)`
omnibias gives the **closed-form** :math:`\sigma', \sigma''` (and, through
autodiff of the closed form, :math:`\sigma'''` -- the third derivative the
*gradient* of a curvature penalty needs). That lets us assemble the
**exact** parameter Hessian of the loss and its curvature functionals
analytically, so the SAM inner-max can be expanded to second order
*exactly* instead of being estimated by an ascent step:

.. math::

    L^{\mathrm{SAM}}_\rho(\theta) - L(\theta)
        \;=\; \rho\,\lVert\nabla L\rVert
             \;+\; \tfrac12\,\rho^2\,\lambda_{\max}(\nabla^2 L)
             \;+\; O(\rho^3).

What this module ships
----------------------

* :func:`mse_loss_hessian` -- the **exact** full parameter Hessian of the
  MSE loss (Gauss-Newton term *plus* the residual
  :math:`\tfrac{2}{B}\sum_n r_n\,\nabla^2_\theta f(x_n)` term). This is
  the "full Newton" curvature the ``one_layer`` roadmap flagged.
* :func:`hessian_trace`, :func:`hessian_frobenius_sq`,
  :func:`hessian_top_eigenvalue` -- the three curvature functionals that
  the loss-landscape literature uses as sharpness proxies
  (:math:`\operatorname{tr}H=\sum_i\lambda_i`, the Sobolev
  :math:`\lVert H\rVert_F^2=\sum_i\lambda_i^2`, and the canonical
  :math:`\lambda_{\max}(H)`).
* :func:`mse_curvature_sharpness` -- assemble the exact loss Hessian and
  return one of those functionals.
* :func:`sharpness_aware_loss` -- :math:`L + \lambda\,\mathcal S(\nabla^2 L)`,
  the differentiable regulariser used in a training loop.
* :func:`sam_sharpness_gap` / :func:`sam_objective` -- the ascent-free,
  exact second-order SAM surrogate.

Honesty caveats
---------------

* Everything here is **exact to floating point** *for the one-layer
  field*: the per-sample output Hessian and the :math:`\sigma^{(k)}` are
  closed form, so :func:`mse_loss_hessian` equals :func:`jax.hessian` of
  the loss to ~1e-9 (see the tests). The *only* approximations are the
  ones you opt into: the SAM surrogate keeps the second-order Taylor
  term (error :math:`O(\rho^3)`), and :math:`\lambda_{\max}` is clamped at
  0 so negative curvature is never rewarded.
* This is a **one-layer** primitive. The arbitrary-depth, torch port is
  :mod:`omnibias.curvature.torch` (matrix-free, via exact Hessian-vector
  products). The scientific claim is *not* "flat minima always generalise
  better" -- that is problem-dependent (No Free Lunch); the claim is that
  omnibias can measure and regularise curvature *exactly and cheaply*,
  where SAM only estimates its linear shadow.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.curvature.one_layer import (
    one_layer_param_grad,
    one_layer_param_hessian,
)
from omnibias.jax.activations import get_activation

# ---------------------------------------------------------------------------
# Exact loss value and full parameter Hessian (MSE)
# ---------------------------------------------------------------------------


def _batch_forward(
    X: Array, W: Array, beta: Array, c: Array, b: Array, activation: str,
) -> Array:
    """Vectorised forward ``f_theta(x_n)`` over the batch -> ``(B,)``."""
    spec = get_activation(activation)
    Z = X @ W.T + beta[None, :]        # (B, H)
    return b + spec.forward(Z) @ c     # (B,)


def mse_loss(
    X: Array, Y: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
) -> Array:
    r"""Mean-squared-error loss :math:`L = \tfrac1B\sum_n (f(x_n)-y_n)^2`.

    Uses the same ``1/B`` normalisation as
    :func:`omnibias.curvature.one_layer.mse_gauss_newton_fisher`, so the
    Gauss-Newton Fisher there is exactly the positive-definite part of
    :func:`mse_loss_hessian`.
    """
    f = _batch_forward(X, W, beta, c, b, activation)
    r = f - Y
    return jnp.mean(r * r)


def mse_loss_hessian(
    X: Array, Y: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
) -> tuple[Array, Array]:
    r"""Exact full parameter Hessian and gradient of the MSE loss.

    For :math:`L = \tfrac1B\sum_n r_n^2` with residual
    :math:`r_n = f_\theta(x_n) - y_n`,

    .. math::

        \nabla^2_\theta L
          \;=\; \underbrace{\tfrac2B\sum_n g_n g_n^\top}_{\text{Gauss-Newton}}
              \;+\; \underbrace{\tfrac2B\sum_n r_n\,\nabla^2_\theta f(x_n)}_{\text{residual}},

    where :math:`g_n = \nabla_\theta f(x_n)`. The Gauss-Newton term is
    :func:`~omnibias.curvature.one_layer.mse_gauss_newton_fisher`; the
    residual term uses the closed-form per-sample output Hessian
    :func:`~omnibias.curvature.one_layer.one_layer_param_hessian`. Unlike
    Gauss-Newton, this is the *true* loss Hessian (it can be indefinite
    away from a minimum), so it is the right object for a curvature /
    sharpness penalty.

    Returns
    -------
    H : ``(P, P)`` symmetric
        The exact loss Hessian.
    g_loss : ``(P,)``
        The loss gradient :math:`\nabla_\theta L = \tfrac2B\sum_n r_n g_n`.
    """
    spec = get_activation(activation)
    B = X.shape[0]

    def per_sample(x_n: Array, y_n: Array) -> tuple[Array, Array, Array]:
        z = W @ x_n + beta
        f = b + spec.forward(z) @ c
        r = f - y_n
        g = one_layer_param_grad(x_n, W, beta, c, b, activation)     # (P,)
        Hf = one_layer_param_hessian(x_n, W, beta, c, b, activation)  # (P, P)
        return r, g, Hf

    rs, gs, Hfs = jax.vmap(per_sample)(X, Y)         # (B,), (B,P), (B,P,P)
    gauss_newton = (2.0 / B) * (gs.T @ gs)           # (P, P)
    residual = (2.0 / B) * jnp.einsum("n,nij->ij", rs, Hfs)  # (P, P)
    H = gauss_newton + residual
    g_loss = (2.0 / B) * (gs.T @ rs)                 # (P,)
    return H, g_loss


# ---------------------------------------------------------------------------
# Curvature functionals (sharpness proxies)
# ---------------------------------------------------------------------------


def hessian_trace(H: Array) -> Array:
    r""":math:`\operatorname{tr}(H) = \sum_i \lambda_i` -- total curvature."""
    return jnp.trace(H)


def hessian_frobenius_sq(H: Array) -> Array:
    r""":math:`\lVert H\rVert_F^2 = \sum_i \lambda_i^2` -- the Sobolev sharpness.

    Smooth and always differentiable (no eigen-decomposition), which
    makes it the most robust default for a training-time penalty.
    """
    return jnp.sum(H * H)


def hessian_top_eigenvalue(H: Array) -> Array:
    r""":math:`\lambda_{\max}(H)` -- the canonical "sharpness".

    Symmetrises ``H`` and returns the largest eigenvalue via
    :func:`jax.numpy.linalg.eigvalsh`. Differentiable wherever the top
    eigenvalue is simple.
    """
    Hs = 0.5 * (H + H.T)
    return jnp.linalg.eigvalsh(Hs)[-1]


_MEASURES: dict[str, Callable[[Array], Array]] = {
    "trace": hessian_trace,
    "frobenius": hessian_frobenius_sq,
    "top_eig": hessian_top_eigenvalue,
}


def _apply_measure(measure: str, H: Array) -> Array:
    try:
        fn = _MEASURES[measure]
    except KeyError:
        raise ValueError(
            f"unknown sharpness measure {measure!r}; "
            f"choose from {sorted(_MEASURES)}"
        ) from None
    return fn(H)


def mse_curvature_sharpness(
    X: Array, Y: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
    measure: str = "frobenius",
) -> Array:
    r"""Exact curvature-based sharpness of the MSE loss at ``theta``.

    Assembles the exact loss Hessian (:func:`mse_loss_hessian`) and
    returns the chosen functional -- ``"trace"``
    (:math:`\operatorname{tr}H`), ``"frobenius"`` (:math:`\lVert H\rVert_F^2`,
    the default) or ``"top_eig"`` (:math:`\lambda_{\max}(H)`).
    """
    H, _ = mse_loss_hessian(X, Y, W, beta, c, b, activation)
    return _apply_measure(measure, H)


# ---------------------------------------------------------------------------
# Sharpness-aware objectives
# ---------------------------------------------------------------------------


def sharpness_aware_loss(
    X: Array, Y: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
    *,
    lam: float = 1e-2,
    measure: str = "frobenius",
) -> Array:
    r"""Curvature-regularised training loss :math:`L + \lambda\,\mathcal S(\nabla^2 L)`.

    A drop-in replacement for the plain MSE loss in a ``jax.grad`` /
    optimiser loop. Because :math:`\mathcal S` is built from the exact
    loss Hessian, differentiating this objective pulls in
    :math:`\sigma'''` **in closed form** (autodiff of the closed-form
    :math:`\sigma''`), so the penalty gradient is exact -- no
    finite-difference sharpness estimate, no extra forward/backward pass.
    """
    H, _ = mse_loss_hessian(X, Y, W, beta, c, b, activation)
    base = mse_loss(X, Y, W, beta, c, b, activation)
    return base + lam * _apply_measure(measure, H)


def sam_sharpness_gap(
    X: Array, Y: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
    *,
    rho: float = 0.05,
) -> Array:
    r"""Exact second-order surrogate of the SAM inner-max gap.

    .. math::

        \max_{\lVert\varepsilon\rVert\le\rho} L(\theta+\varepsilon) - L(\theta)
          \;\approx\; \rho\,\lVert\nabla L\rVert
                 \;+\; \tfrac12\,\rho^2\,\max\!\big(\lambda_{\max}(\nabla^2 L),\,0\big),

    exact to :math:`O(\rho^3)`. This is what standard SAM *estimates*
    with one ascent step (linear term only); here both terms are computed
    from the closed-form gradient and Hessian in a single pass. The
    curvature term is clamped at 0 so negative curvature is never
    rewarded (the gap is :math:`\ge 0`).
    """
    H, g = mse_loss_hessian(X, Y, W, beta, c, b, activation)
    gnorm = jnp.linalg.norm(g)
    lam_max = hessian_top_eigenvalue(H)
    return rho * gnorm + 0.5 * rho * rho * jnp.maximum(lam_max, 0.0)


def sam_objective(
    X: Array, Y: Array,
    W: Array, beta: Array, c: Array, b: Array,
    activation: str = "tanh",
    *,
    rho: float = 0.05,
) -> Array:
    r"""Ascent-free "SAM done right": :math:`L(\theta) + \text{gap}_\rho(\theta)`.

    Equivalent to the SAM objective expanded to exact second order (see
    :func:`sam_sharpness_gap`). Minimising it simultaneously lowers the
    loss and the worst-case curvature of the basin.
    """
    base = mse_loss(X, Y, W, beta, c, b, activation)
    return base + sam_sharpness_gap(
        X, Y, W, beta, c, b, activation, rho=rho
    )


__all__ = [
    "hessian_frobenius_sq",
    "hessian_top_eigenvalue",
    "hessian_trace",
    "mse_curvature_sharpness",
    "mse_loss",
    "mse_loss_hessian",
    "sam_objective",
    "sam_sharpness_gap",
    "sharpness_aware_loss",
]
