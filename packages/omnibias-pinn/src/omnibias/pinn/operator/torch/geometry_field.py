# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Geometry-conditioned DeepONet fields with hard boundary factors (torch).

When an operator is conditioned on geometry probes, the free DeepONet field can
be wrapped with :class:`~omnibias.pinn.domain.torch.DistanceConstrainedField`
so each sample satisfies its own Dirichlet boundary by construction (where the
SDF primitive supports hard factors).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omnibias.pinn.domain._core.sdf import SDF
from omnibias.pinn.domain.torch import build_distance_constrained_field
from omnibias.pinn.operator.torch.deeponet import DeepONetField, DeepONetOperator
from omnibias.pinn.torch.cage.conservation import HardBoundaryField
from torch import Tensor


def condition_with_geometry(
    operator: DeepONetOperator,
    sensors: Tensor,
    *,
    parameters: Tensor | None = None,
    boundary: Tensor | None = None,
    geometry: Tensor | None = None,
    sdf: SDF | Sequence[SDF] | None = None,
    hard_bc: bool = True,
) -> DeepONetField | HardBoundaryField | list[HardBoundaryField]:
    """Condition the operator and optionally apply per-sample hard BC factors.

    When ``hard_bc`` is True and ``sdf`` is provided, each conditioned sample
    is wrapped as ``u = g + phi * NN`` via
    :class:`~omnibias.pinn.domain.torch.DistanceConstrainedField`. A single SDF
    is broadcast across the batch; a sequence of length ``F`` gives one cage per
    sample.

  Guarantee level: exact Dirichlet satisfaction where ``phi`` vanishes on the
  declared boundary; residual accuracy remains optimised, not proven.
    """
    field = operator.condition(
        sensors,
        parameters=parameters,
        boundary=boundary,
        geometry=geometry,
    )
    if not hard_bc or sdf is None:
        return field
    F = int(field._n_functions)
    if isinstance(sdf, Sequence) and not isinstance(sdf, (str, bytes)):
        sdfs = list(sdf)
        if len(sdfs) == 1 and F > 1:
            sdfs = sdfs * F
        if len(sdfs) != F:
            raise ValueError(
                f"sdf sequence length {len(sdfs)} != conditioned batch F={F}"
            )
        return [
            build_distance_constrained_field(field, s, bounded_names=tuple(field.components.names))
            for s in sdfs
        ]
    wrapped = build_distance_constrained_field(
        field,
        sdf,
        bounded_names=tuple(field.components.names),
    )
    return wrapped


def evaluate_geometry_batch(
    fields: Sequence[HardBoundaryField | DeepONetField],
    coords: Tensor,
) -> Tensor:
    """Evaluate a per-sample geometry-conditioned field on a shared query grid.

    ``coords`` has shape ``(Q, D)``. Returns values ``(F, Q, C)`` by evaluating
    each field on the same query set (no shared trunk-jet cache across cages).
    """
    from omnibias.pinn.torch import ops as tops

    if coords.ndim != 2:
        raise ValueError(f"coords must be 2-D (Q, D); got {tuple(coords.shape)}")
    rows: list[Tensor] = []
    for field in fields:
        if isinstance(field, DeepONetField):
            state = field.on_grid(coords)
        else:
            state = field(coords)
        rows.append(tops.value(state, field.components.names[0]))
    return torch.stack(rows, dim=0)


__all__ = ["condition_with_geometry", "evaluate_geometry_batch"]
