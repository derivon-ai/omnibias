# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX ansatz factory (reuses the omnibias-pinn one-layer field)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.jax.activations import get_activation
from omnibias.pinn.jax.fields.one_layer import (
    OneLayerVectorField,
    make_one_layer_vector_field,
)
from omnibias.pinn.solver._core.system import System


def build_field(
    system: System,
    *,
    hidden: int = 64,
    activation: str = "tanh",
    weight_init_scale: float | None = None,
    bias_init: str = "zeros",
    dtype: Any = jnp.float64,
    seed: int = 0,
) -> OneLayerVectorField:
    """Build a one-layer omnibias field carrying every system component."""
    return make_one_layer_vector_field(
        coordinate_spec=system.domain.coordinate_spec,
        components=system.component_spec(),
        hidden=hidden,
        base=activation,
        weight_init_scale=weight_init_scale,
        bias_init=bias_init,
        seed=seed,
        dtype=dtype,
    )


def with_readout(field: OneLayerVectorField, c: Any, b: Any) -> OneLayerVectorField:
    """Return a copy of ``field`` with a new (frozen) readout ``(c, b)``."""
    return OneLayerVectorField(
        coordinate_spec=field.coordinate_spec,
        components=field.components,
        spec=field.spec,
        W=field.W,
        beta=field.beta,
        c=c,
        b=b,
        hidden=field.hidden,
    )


def field_from_arrays(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    W: Any,
    beta: Any,
    c: Any,
    b: Any,
    activation: str = "tanh",
) -> OneLayerVectorField:
    """Build a field from explicit parameter arrays (used for backend parity)."""
    return OneLayerVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        spec=get_activation(activation),
        W=jnp.asarray(W),
        beta=jnp.asarray(beta),
        c=jnp.asarray(c),
        b=jnp.asarray(b),
        hidden=int(jnp.asarray(W).shape[0]),
    )


__all__ = ["OneLayerVectorField", "build_field", "field_from_arrays", "with_readout"]
