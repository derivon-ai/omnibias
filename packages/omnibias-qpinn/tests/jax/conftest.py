# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared fixtures for the jax backend tests.

Builds small, deterministic :class:`OneLayerVectorField` instances on
``jnp.float64`` (we enable ``jax_enable_x64`` because all parity tests
need double precision).
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import (
    OneLayerVectorField,
    make_one_layer_vector_field,
)
from omnibias.qpinn import make_psi_components


@pytest.fixture
def coord_xt() -> CoordinateSpec:
    return CoordinateSpec(axes=("x", "t"))


@pytest.fixture
def coord_x() -> CoordinateSpec:
    return CoordinateSpec(axes=("x",))


@pytest.fixture
def psi_field_xt(coord_xt: CoordinateSpec) -> OneLayerVectorField:
    spec = make_psi_components(name="psi")
    return make_one_layer_vector_field(
        coordinate_spec=coord_xt,
        components=spec,
        hidden=8,
        base="gaussian",
        dtype=jnp.float64,
        seed=0,
    )


@pytest.fixture
def psi_field_x(coord_x: CoordinateSpec) -> OneLayerVectorField:
    spec = make_psi_components(name="psi")
    return make_one_layer_vector_field(
        coordinate_spec=coord_x,
        components=spec,
        hidden=8,
        base="gaussian",
        dtype=jnp.float64,
        seed=0,
    )


@pytest.fixture
def coords_xt() -> jnp.ndarray:
    return jax.random.normal(jax.random.PRNGKey(1), (16, 2), dtype=jnp.float64)


@pytest.fixture
def coords_x() -> jnp.ndarray:
    return jnp.linspace(-2.0, 2.0, 17, dtype=jnp.float64)[:, None]
