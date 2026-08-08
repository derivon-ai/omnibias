# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Distance-constrained PINN field for curved boundaries (JAX twin).

JAX cages are pytree nodes constructed via factories; this module wraps
:func:`~omnibias.pinn.jax.cage.make_hard_boundary_field` rather than
subclassing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from jax import Array
from omnibias.pinn.domain._core.sdf import Box, Halfspace, Sphere
from omnibias.pinn.domain.jax.sdf_jax import (
    DistanceFn,
    from_primitive,
    normalize_distance,
)
from omnibias.pinn.jax.cage.conservation import (
    HardBoundaryField,
    make_hard_boundary_field,
)
from omnibias.pinn.jax.fields.base import FieldBase


def build_distance_constrained_field(
    base: FieldBase,
    sdf: Sphere | Box | Halfspace | None = None,
    *,
    distance_fn: DistanceFn | None = None,
    boundary_value_fn: Callable[[Array], dict[str, Array]] | None = None,
    bounded_names: Sequence[str] | None = None,
    passthrough_names: tuple[str, ...] = (),
    groups: dict[str, tuple[str, ...]] | None = None,
    max_derivative_order: int = 4,
    normalize: bool = True,
) -> HardBoundaryField:
    """Build a hard-BC cage ``u = g + phi * NN`` from an SDF / ADF (JAX)."""
    if distance_fn is None:
        if sdf is None:
            raise ValueError("provide distance_fn or sdf")
        distance_fn = from_primitive(sdf)
        if normalize:
            distance_fn = normalize_distance(distance_fn)
    elif normalize and sdf is not None:
        distance_fn = normalize_distance(distance_fn)
    return make_hard_boundary_field(
        base=base,
        distance_fn=distance_fn,
        boundary_value_fn=boundary_value_fn,
        bounded_names=tuple(bounded_names) if bounded_names is not None else None,
        passthrough_names=passthrough_names,
        groups=groups,
        max_derivative_order=max_derivative_order,
    )


# Alias kept for API symmetry with the torch twin.
DistanceConstrainedField = HardBoundaryField


__all__ = [
    "DistanceConstrainedField",
    "build_distance_constrained_field",
]
