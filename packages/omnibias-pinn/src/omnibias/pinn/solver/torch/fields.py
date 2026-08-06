# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch ansatz factory (reuses the omnibias-pinn one-layer field).

The mesh-free ansatz is the closed-form :class:`OneLayerVectorField` from
``omnibias-pinn`` -- a single hidden layer whose every derivative is the exact
``sigma``-tower reduction. We do not re-implement it here; we only assemble the
right :class:`ComponentSpec` from a :class:`System`.
"""

from __future__ import annotations

from typing import Any

import torch
from omnibias.pinn.solver._core.hard import HardConditionPlan
from omnibias.pinn.solver._core.system import System
from omnibias.pinn.torch.cage.constrained import ConstrainedExpressionField
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField


def build_field(
    system: System,
    *,
    hidden: int = 64,
    activation: str = "tanh",
    weight_init_scale: float | None = None,
    bias_init: str = "zeros",
    dtype: torch.dtype = torch.float64,
    seed: int | None = 0,
    hard_conditions: HardConditionPlan | None = None,
) -> Any:
    """Build a one-layer omnibias field carrying every system component.

    When ``hard_conditions`` carries a non-empty plan the ansatz is wrapped in a
    :class:`~omnibias.pinn.torch.cage.constrained.ConstrainedExpressionField`, so
    the absorbed conditions hold identically rather than being penalised. The
    wrapper forwards the readout, which is what keeps the frozen-feature linear
    path linear.
    """
    if seed is not None:
        torch.manual_seed(seed)
    base = OneLayerVectorField(
        coordinate_spec=system.domain.coordinate_spec,
        components=system.component_spec(),
        hidden=hidden,
        base=activation,
        weight_init_scale=weight_init_scale,
        bias_init=bias_init,
        dtype=dtype,
    )
    if not hard_conditions:
        return base
    return ConstrainedExpressionField(
        base=base,
        conditions=hard_conditions.conditions,
        bounds=system.domain.bounds,
        passthrough_names=tuple(
            n
            for n in system.component_names()
            if n not in {c.component for c in hard_conditions.conditions}
        ),
        certify=False,  # the plan already sealed these certificates
    )


def freeze_features(field: Any) -> None:
    """Freeze the hidden layer so the readout is the only unknown (linear)."""
    field.W.weight.requires_grad_(False)
    field.W.bias.requires_grad_(False)


__all__ = [
    "ConstrainedExpressionField",
    "OneLayerVectorField",
    "build_field",
    "freeze_features",
]
