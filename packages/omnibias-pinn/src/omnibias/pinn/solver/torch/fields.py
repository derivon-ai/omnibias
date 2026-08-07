# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch ansatz factory (one-layer MLP or spectral Fourier time-head).

The mesh-free ansatz is either the closed-form
:class:`~omnibias.pinn.torch.fields.one_layer.OneLayerVectorField` or
:class:`~omnibias.pinn.torch.fields.spectral.SpectralVectorField`. We do not
re-implement them here; we only assemble the right
:class:`~omnibias.fields._core.components.ComponentSpec` from a
:class:`~omnibias.pinn.solver._core.system.System` and wrap hard conditions.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from omnibias.pinn.solver._core.hard import HardConditionPlan
from omnibias.pinn.solver._core.system import System
from omnibias.pinn.solver.torch.readout import freeze_features
from omnibias.pinn.torch.cage.constrained import ConstrainedExpressionField
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.pinn.torch.fields.spectral import SpectralVectorField


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
    basis: str = "mlp",
    K: int = 8,
    L: float | tuple[float, ...] = 2.0 * math.pi,
    time_hidden: int | None = None,
    time_depth: int = 1,
) -> Any:
    """Build an omnibias field carrying every system component.

    Parameters
    ----------
    basis
        ``"mlp"`` (default) builds a :class:`OneLayerVectorField`.
        ``"spectral"`` builds a :class:`SpectralVectorField` and requires the
        system domain to declare a time axis.
    K, L, time_hidden, time_depth
        Spectral-only kwargs. ``time_hidden`` defaults to ``hidden``.

    When ``hard_conditions`` carries a non-empty plan the ansatz is wrapped in a
    :class:`~omnibias.pinn.torch.cage.constrained.ConstrainedExpressionField`, so
    the absorbed conditions hold identically rather than being penalised. The
    wrapper forwards the readout, which is what keeps the frozen-feature linear
    path linear.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if basis == "mlp":
        base: Any = OneLayerVectorField(
            coordinate_spec=system.domain.coordinate_spec,
            components=system.component_spec(),
            hidden=hidden,
            base=activation,
            weight_init_scale=weight_init_scale,
            bias_init=bias_init,
            dtype=dtype,
        )
    elif basis == "spectral":
        if system.domain.time_axis is None:
            raise ValueError(
                "basis='spectral' requires a time axis on the system domain; "
                "SpectralVectorField is a space-time Fourier ansatz. Use "
                "basis='mlp' for steady / purely spatial problems "
                f"(got system {system.name!r})."
            )
        scale = 1.0 if weight_init_scale is None else float(weight_init_scale)
        base = SpectralVectorField(
            coordinate_spec=system.domain.coordinate_spec,
            components=system.component_spec(),
            K=K,
            L=L,
            time_hidden=hidden if time_hidden is None else int(time_hidden),
            time_depth=time_depth,
            activation=activation,
            weight_init_scale=scale,
            dtype=dtype,
        )
    else:
        raise ValueError(
            f"unknown basis {basis!r}; expected 'mlp' or 'spectral'"
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


__all__ = [
    "ConstrainedExpressionField",
    "OneLayerVectorField",
    "SpectralVectorField",
    "build_field",
    "freeze_features",
]
