# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX ansatz factory (one-layer MLP or spectral Fourier time-head)."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.jax.activations import get_activation
from omnibias.pinn.jax.cage.constrained import (
    ConstrainedExpressionField,
    make_constrained_expression_field,
)
from omnibias.pinn.jax.fields.one_layer import (
    OneLayerVectorField,
    make_one_layer_vector_field,
)
from omnibias.pinn.jax.fields.spectral import (
    SpectralVectorField,
    make_spectral_vector_field,
)
from omnibias.pinn.solver._core.hard import HardConditionPlan
from omnibias.pinn.solver._core.system import System
from omnibias.pinn.solver.jax.readout import with_readout


def build_field(
    system: System,
    *,
    hidden: int = 64,
    activation: str = "tanh",
    weight_init_scale: float | None = None,
    bias_init: str = "zeros",
    dtype: Any = jnp.float64,
    seed: int = 0,
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
    :class:`~omnibias.pinn.jax.cage.constrained.ConstrainedExpressionField`, so
    the absorbed conditions hold identically rather than being penalised.
    """
    if basis == "mlp":
        base: Any = make_one_layer_vector_field(
            coordinate_spec=system.domain.coordinate_spec,
            components=system.component_spec(),
            hidden=hidden,
            base=activation,
            weight_init_scale=weight_init_scale,
            bias_init=bias_init,
            seed=seed,
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
        base = make_spectral_vector_field(
            coordinate_spec=system.domain.coordinate_spec,
            components=system.component_spec(),
            K=K,
            L=L,
            time_hidden=hidden if time_hidden is None else int(time_hidden),
            time_depth=time_depth,
            activation=activation,
            weight_init_scale=scale,
            seed=seed,
            dtype=dtype,
        )
    else:
        raise ValueError(
            f"unknown basis {basis!r}; expected 'mlp' or 'spectral'"
        )
    if not hard_conditions:
        return base
    return make_constrained_expression_field(
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


__all__ = [
    "ConstrainedExpressionField",
    "OneLayerVectorField",
    "SpectralVectorField",
    "build_field",
    "field_from_arrays",
    "with_readout",
]
