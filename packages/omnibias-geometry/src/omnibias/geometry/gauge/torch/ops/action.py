# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Yang-Mills action and (anti-)self-duality (torch).

The action density is ``(1/4) F_{mu nu}^a F^{mu nu, a}`` with the coupling
absorbed into ``F`` (``F = dA + g[A, A]``); the total action integrates it over
spacetime. A field is *self-dual* when ``F = \tilde F`` (instanton) and
*anti-self-dual* when ``F = -\tilde F`` (anti-instanton).
"""

from __future__ import annotations

import torch
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge.torch.ops.hodge import dual_field_strength, signature_diagonal
from torch import Tensor


def action_density(F: Tensor, *, signature: tuple[int, ...]) -> Tensor:
    r"""Yang-Mills Lagrangian density ``(1/4) F_{mu nu}^a F^{mu nu, a}`` -> ``(B,)``."""
    eta = signature_diagonal(signature, dtype=F.dtype, device=F.device)
    return kernels.action_density(torch, F, eta)


def yang_mills_action(
    F: Tensor, *, signature: tuple[int, ...], weights: Tensor | None = None
) -> Tensor:
    r"""Total action ``S = int (1/4) F^2`` as a scalar.

    ``weights`` are optional per-sample quadrature weights (e.g. the cell volume
    of a grid); when omitted the densities are summed with unit weight.
    """
    density = action_density(F, signature=signature)
    if weights is None:
        return torch.sum(density)
    return torch.sum(density * weights)


def self_dual_projector(F: Tensor, *, signature: tuple[int, ...]) -> Tensor:
    r"""Self-dual part ``(1/2)(F + \tilde F)``."""
    return 0.5 * (F + dual_field_strength(F, signature=signature))


def anti_self_dual_projector(F: Tensor, *, signature: tuple[int, ...]) -> Tensor:
    r"""Anti-self-dual part ``(1/2)(F - \tilde F)``."""
    return 0.5 * (F - dual_field_strength(F, signature=signature))


def self_duality_defect(F: Tensor, *, signature: tuple[int, ...]) -> Tensor:
    r"""``F - \tilde F`` (zero for a self-dual field)."""
    fd = dual_field_strength(F, signature=signature)
    return F - fd


def anti_self_duality_defect(F: Tensor, *, signature: tuple[int, ...]) -> Tensor:
    r"""``F + \tilde F`` (zero for an anti-self-dual field)."""
    fd = dual_field_strength(F, signature=signature)
    return F + fd


__all__ = [
    "action_density",
    "anti_self_dual_projector",
    "anti_self_duality_defect",
    "self_dual_projector",
    "self_duality_defect",
    "yang_mills_action",
]
