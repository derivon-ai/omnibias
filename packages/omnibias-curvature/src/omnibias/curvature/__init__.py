# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form second-order optimisation for one-layer Riccati fields.

This package promotes the omnibias closed-form Hessian
(:func:`omnibias.jax.neural_field_value_grad_hessian`, which is the
:math:`D \times D` Hessian *with respect to the input*) into a usable
**parameter** Fisher / Hessian for the one-hidden-layer scalar field

.. math::

    f_\theta(x) \;=\; b \;+\; \sum_h c_h\, \sigma(W_h \cdot x + \beta_h),

where :math:`\sigma` is one of the Riccati-class activations exposed by
:func:`omnibias.jax.activations.get_activation` (``tanh``, ``sigmoid``,
``softplus``, ``gaussian``, ``exp``).  The closed forms for
:math:`\sigma'(z)` and :math:`\sigma''(z)` mean every block of the
parameter Hessian / Fisher can be assembled algebraically — no autograd
needed.

What this v0.1 ships
--------------------

The :mod:`omnibias.curvature.one_layer` module provides:

* :func:`one_layer_param_grad` — per-sample :math:`\nabla_\theta f(x)`
  as a flat vector of length :math:`P = 1 + H + H + H \cdot D`.
* :func:`one_layer_param_hessian` — per-sample full Hessian
  :math:`\nabla^2_\theta f(x)`; block-diagonal-in-hidden-unit with
  :math:`(D+2) \times (D+2)` blocks.
* :func:`mse_gauss_newton_fisher` — Gauss-Newton Fisher
  :math:`F = \tfrac{2}{B}\sum_n g_n g_n^T` for MSE loss; the standard
  positive-definite curvature surrogate used by natural-gradient and
  KFAC.
* :func:`mse_newton_step` — one Newton-Gauss step
  :math:`\theta \leftarrow \theta - \eta\,(F + \lambda I)^{-1} \nabla L`.
* :func:`kfac_kron_factors` — closed-form KFAC :math:`(A, G)` Kronecker
  factors for the hidden block, computed from :math:`x_n x_n^T` and
  :math:`(c \odot \sigma'(z_n))(c \odot \sigma'(z_n))^T` statistics.

Exact-curvature sharpness ("SAM done right")
--------------------------------------------

The :mod:`omnibias.curvature.sharpness` module builds on the closed-form
Hessian to measure and regularise the *sharpness* of a basin exactly,
where Sharpness-Aware Minimisation only estimates its linear shadow:

* :func:`mse_loss_hessian` — the **exact full loss Hessian** (Gauss-Newton
  *plus* the :math:`r_n\,\nabla^2_\theta f` residual term).
* :func:`hessian_trace` / :func:`hessian_frobenius_sq` /
  :func:`hessian_top_eigenvalue` — the sharpness functionals
  :math:`\sum_i\lambda_i`, :math:`\sum_i\lambda_i^2`,
  :math:`\lambda_{\max}`.
* :func:`sharpness_aware_loss` — :math:`L + \lambda\,\mathcal S(\nabla^2 L)`,
  a differentiable curvature regulariser (its gradient pulls in
  :math:`\sigma'''` in closed form).
* :func:`sam_objective` — the ascent-free, exact second-order SAM
  surrogate :math:`L + \rho\lVert\nabla L\rVert + \tfrac12\rho^2\lambda_{\max}`.

These are the **one-layer, JAX** primitives. The arbitrary-depth,
**PyTorch** port lives in :mod:`omnibias.curvature.torch` (matrix-free via
exact Hessian-vector products) and requires the optional ``torch`` extra;
it is intentionally not imported here so this namespace stays JAX-only.

What's still on the roadmap
---------------------------

* Integration with ``kfac_jax`` as a custom layer registration so that
  a FermiNet trainer can drop in the closed-form Fisher
  block for the omnibias-envelope layer.

These follow-ups are tracked on the public roadmap in
``docs/roadmap.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("omnibias-curvature")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

from omnibias.curvature.glm_fisher import (
    fisher_information_metric,
    glm_fisher,
)
from omnibias.curvature.natural_gradient import (
    damped_solve,
    glm_loss_gradient,
    glm_natural_gradient_step,
    natural_gradient_step,
)
from omnibias.curvature.one_layer import (
    kfac_kron_factors,
    mse_gauss_newton_fisher,
    mse_newton_step,
    one_layer_param_grad,
    one_layer_param_hessian,
)
from omnibias.curvature.regularize import (
    CollapseResult,
    min_norm_solve,
    numerical_rank,
    rank_collapse,
    regularization_path,
    regularized_solve,
)
from omnibias.curvature.sharpness import (
    hessian_frobenius_sq,
    hessian_top_eigenvalue,
    hessian_trace,
    mse_curvature_sharpness,
    mse_loss,
    mse_loss_hessian,
    sam_objective,
    sam_sharpness_gap,
    sharpness_aware_loss,
)

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "CollapseResult",
    "__lineage__",
    "damped_solve",
    "fisher_information_metric",
    "glm_fisher",
    "glm_loss_gradient",
    "glm_natural_gradient_step",
    "hessian_frobenius_sq",
    "hessian_top_eigenvalue",
    "hessian_trace",
    "kfac_kron_factors",
    "min_norm_solve",
    "mse_curvature_sharpness",
    "mse_gauss_newton_fisher",
    "mse_loss",
    "mse_loss_hessian",
    "mse_newton_step",
    "natural_gradient_step",
    "numerical_rank",
    "one_layer_param_grad",
    "one_layer_param_hessian",
    "rank_collapse",
    "regularization_path",
    "regularized_solve",
    "sam_objective",
    "sam_sharpness_gap",
    "sharpness_aware_loss",
]
