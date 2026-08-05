# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Yang-Mills action and (anti-)self-duality (jax).

The jax twin of :mod:`omnibias.geometry.gauge.torch.ops.action`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge.jax.ops.hodge import dual_field_strength, signature_diagonal

Array = Any


def action_density(F: Array, *, signature: tuple[int, ...]) -> Array:
    r"""Yang-Mills Lagrangian density ``(1/4) F_{mu nu}^a F^{mu nu, a}`` -> ``(B,)``."""
    eta = signature_diagonal(signature, dtype=F.dtype)
    return kernels.action_density(jnp, F, eta)


def yang_mills_action(
    F: Array, *, signature: tuple[int, ...], weights: Array | None = None
) -> Array:
    r"""Total action ``S = int (1/4) F^2`` as a scalar."""
    density = action_density(F, signature=signature)
    if weights is None:
        return jnp.sum(density)
    return jnp.sum(density * weights)


def self_dual_projector(F: Array, *, signature: tuple[int, ...]) -> Array:
    r"""Self-dual part ``(1/2)(F + \tilde F)``."""
    return 0.5 * (F + dual_field_strength(F, signature=signature))


def anti_self_dual_projector(F: Array, *, signature: tuple[int, ...]) -> Array:
    r"""Anti-self-dual part ``(1/2)(F - \tilde F)``."""
    return 0.5 * (F - dual_field_strength(F, signature=signature))


def self_duality_defect(F: Array, *, signature: tuple[int, ...]) -> Array:
    r"""``F - \tilde F`` (zero for a self-dual field)."""
    return F - dual_field_strength(F, signature=signature)


def anti_self_duality_defect(F: Array, *, signature: tuple[int, ...]) -> Array:
    r"""``F + \tilde F`` (zero for an anti-self-dual field)."""
    return F + dual_field_strength(F, signature=signature)


__all__ = [
    "action_density",
    "anti_self_dual_projector",
    "anti_self_duality_defect",
    "self_dual_projector",
    "self_duality_defect",
    "yang_mills_action",
]
