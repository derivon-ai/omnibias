# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.pinn.operator: neural operator learning with closed-form derivatives.

A DeepONet ``G(u)(y) = b_0 + sum_k b_k(u) t_k(y)`` is linear in the trunk
basis, so every query-coordinate derivative is

    d^alpha G(u)(y) = sum_k b_k(u) d^alpha t_k(y)

and the trunk is an omnibias jet network. One trunk jet therefore yields every
mixed partial of the operator output up to a chosen order -- mesh-free, with no
finite differences and no periodic-grid requirement. That is the load-bearing
claim; operator *accuracy* is optimised, not proven.

This module re-exports the backend-free schemas (:class:`OperatorSpec`,
:class:`SensorGrid`, :func:`sample_fourier_ics`). The numeric drivers live
under ``omnibias.pinn.operator.torch`` and ``omnibias.pinn.operator.jax`` and are
imported lazily so ``import omnibias.pinn.operator`` never pulls in torch or jax.

Maturity: **alpha** submodule of the Beta ``omnibias-pinn`` distribution.

Naming
------
"Operator" in omnibias names three different objects; see
``docs/operator-surface.md`` ("Three senses of operator"). This submodule is
sense 3 -- neural operator learning (function-space to function-space maps) --
not :class:`~omnibias.torch.blocks.operator.OperatorBlock` and not a field
operator like ``grad`` / ``laplacian``.
"""

from __future__ import annotations

from omnibias.pinn.operator._core import (
    ConditioningSpec,
    OperatorSpec,
    SensorGrid,
    branch_coefficient_box,
    certify_heat_residual,
    enclose_heat_residual,
    sample_fourier_ics,
)

__all__ = [
    "ConditioningSpec",
    "OperatorSpec",
    "SensorGrid",
    "branch_coefficient_box",
    "certify_heat_residual",
    "enclose_heat_residual",
    "sample_fourier_ics",
]
