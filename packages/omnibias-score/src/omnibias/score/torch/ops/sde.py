# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SDE / score operators (torch), composed from closed-form field primitives.

For an Ito diffusion ``dX = b(X) dt + sigma(X) dW`` with ``a = sigma sigma^T``:

- the **generator** ``L f = b . grad f + 1/2 tr(a hess f)``,
- the **Fokker-Planck adjoint** ``L* p = -div(b p) + 1/2 d_i d_j (a_ij p)``,
- the **score** ``grad log p = grad p / p``.

All spatial derivatives are the omnibias closed-form gradient / Hessian; only the
drift / diffusion coefficients are supplied by the caller. The Fokker-Planck
adjoint here assumes a spatially-constant diffusion ``a`` (the common
constant-noise case, e.g. Ornstein-Uhlenbeck).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import gradient, value
from omnibias.fields.torch.ops.high_order import hessian
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _quad(a: Tensor, hess: Tensor) -> Tensor:
    if a.dim() == 2:
        return torch.einsum("ij,bij->b", a, hess)
    return torch.einsum("bij,bij->b", a, hess)


def score(state: FieldState, name: str, *, eps: float = 0.0) -> Tensor:
    r"""Score ``grad log p = grad p / p`` of shape ``(B, d)``."""
    p = value(state, name)
    g = gradient(state, name)
    return g / (p.unsqueeze(-1) + eps)


def ito_generator(
    state: FieldState, name: str, *, drift: Tensor, diffusion: Tensor,
) -> Tensor:
    r"""Ito generator ``L f = b . grad f + 1/2 tr(a hess f)`` of shape ``(B,)``.

    Parameters
    ----------
    state, name
        The field and the scalar component ``f``.
    drift
        ``b`` of shape ``(B, d)``.
    diffusion
        ``a = sigma sigma^T`` of shape ``(d, d)`` or ``(B, d, d)``.
    """
    grad = gradient(state, name)
    hess = hessian(state, name)
    return torch.einsum("bi,bi->b", drift, grad) + 0.5 * _quad(diffusion, hess)


def fokker_planck(
    state: FieldState,
    name: str,
    *,
    drift: Tensor,
    diffusion: Tensor,
    drift_divergence: Tensor,
) -> Tensor:
    r"""Fokker-Planck adjoint ``L* p`` of shape ``(B,)`` (constant diffusion).

    .. math::

        L^* p = -\big((\nabla\cdot b)\,p + b\cdot\nabla p\big)
                + \tfrac12\,a_{ij}\,\partial_i\partial_j p.

    Parameters
    ----------
    drift
        ``b`` of shape ``(B, d)``.
    diffusion
        Constant ``a = sigma sigma^T`` of shape ``(d, d)`` or ``(B, d, d)``.
    drift_divergence
        ``div b`` of shape ``(B,)`` (supplied analytically by the caller).
    """
    p = value(state, name)
    gradp = gradient(state, name)
    hessp = hessian(state, name)
    transport = drift_divergence * p + torch.einsum("bi,bi->b", drift, gradp)
    return -transport + 0.5 * _quad(diffusion, hessp)


__all__ = ["fokker_planck", "ito_generator", "score"]
