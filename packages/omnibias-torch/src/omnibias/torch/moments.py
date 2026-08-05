# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Analytic moment propagation (PyTorch).

Pushes the moments of a random input through a network in closed form using the
exact multivariate jet (:func:`omnibias.torch.jet_mv.mlp_jet_mv`) -- a
deterministic alternative to Monte-Carlo uncertainty propagation.

* :func:`gaussian_moment_propagation` -- second-order Gaussian delta method for a
  full MLP ``f: R^D -> R^C``: the Jacobian gives ``J^T Sigma J`` and the Hessian
  the mean correction ``1/2 tr(H Sigma)``.
* :func:`delta_method_gaussian` / :func:`delta_method_moments` -- arbitrary-order
  univariate delta method, reusing the ring-generic
  :func:`omnibias.core.moments.delta_method_central_moments` elementwise over a
  closed-form derivative tower (e.g. an activation applied per unit).
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.moments import (
    delta_method_central_moments,
    gaussian_central_moments,
)
from omnibias.core.spec import ActivationSpec
from omnibias.torch.jet_mv import jet_gradient, jet_hessian, mlp_jet_mv

import torch
from torch import Tensor

Layer = tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | str | None]


def gaussian_moment_propagation(
    layers: Sequence[Layer],
    mean: Tensor,
    cov: Tensor,
    *,
    order: int = 2,
) -> tuple[Tensor, Tensor]:
    r"""Propagate a Gaussian ``N(mean, cov)`` through an MLP analytically.

    Parameters
    ----------
    layers:
        Sequence of ``(W, b, spec)`` as accepted by
        :func:`~omnibias.torch.jet_mv.mlp_jet_mv`.
    mean:
        Input mean, shape ``(D,)``.
    cov:
        Input covariance, shape ``(D, D)``.
    order:
        ``1`` for the first-order (Jacobian-only) propagation, ``>= 2`` to add
        the second-order Hessian mean correction.

    Returns
    -------
    (out_mean, out_cov)
        Output mean ``(C,)`` and covariance ``(C, C)``.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    mean = torch.as_tensor(mean)
    cov = torch.as_tensor(cov)
    dim = int(mean.shape[-1])
    jet_order = max(order, 2) if order >= 2 else 1
    jet = mlp_jet_mv(mean, layers, jet_order)
    value = jet[0]
    jac = jet_gradient(jet, dim, jet_order)  # (D, C)
    out_cov = torch.einsum("ic,ij,jd->cd", jac, cov, jac)
    if order >= 2:
        hess = jet_hessian(jet, dim, jet_order)  # (D, D, C)
        correction = 0.5 * torch.einsum("ijc,ij->c", hess, cov)
        out_mean = value + correction
    else:
        out_mean = value
    return out_mean, out_cov


def delta_method_moments(
    derivatives: Sequence[Tensor],
    central_in: Sequence[float | Tensor],
    *,
    order: int,
) -> dict[str, object]:
    r"""Elementwise analytic delta method from a derivative tower + central moments.

    ``derivatives = [f(mu), f'(mu), ...]`` are per-unit tensors (the closed-form
    tower); ``central_in = [mu_1=0, mu_2, ...]`` the input central moments.
    Returns ``{"mean", "variance", "central"}`` with tensor entries.
    """
    return delta_method_central_moments(list(derivatives), list(central_in), order)


def delta_method_gaussian(
    derivatives: Sequence[Tensor],
    variance: float | Tensor,
    *,
    order: int,
) -> dict[str, object]:
    """:func:`delta_method_moments` for a Gaussian input of the given ``variance``."""
    central = gaussian_central_moments(variance, order)
    return delta_method_central_moments(list(derivatives), central, order)


__all__ = [
    "delta_method_gaussian",
    "delta_method_moments",
    "gaussian_moment_propagation",
]
