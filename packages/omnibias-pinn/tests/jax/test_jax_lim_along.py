# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""End-to-end ``state.<component>.lim_along`` against a real jax field.

Shows the opt-in registry extension surfacing the model-level jet ``lim`` as an
attribute on a component view, with the actual (differentiable) limit computed by
a user closure built on :func:`omnibias.pinn.jax.losses.asymptotic_ratio`.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.fields import ops_registry  # noqa: E402
from omnibias.pinn._core.components import ComponentSpec  # noqa: E402
from omnibias.pinn._core.coords import CoordinateSpec  # noqa: E402
from omnibias.pinn.extensions import register_lim_along  # noqa: E402
from omnibias.pinn.jax.fields.one_layer import (  # noqa: E402
    make_one_layer_vector_field,
)
from omnibias.pinn.jax.losses import asymptotic_ratio  # noqa: E402


def setup_function(function):
    ops_registry.clear()


def teardown_function(function):
    ops_registry.clear()


def _state():
    y = (-np.pi + 2.0 * np.pi * np.arange(16) / 16).reshape(-1, 1)
    cspec = CoordinateSpec(axes=("y",), periodicity=(True,), time_axis=None)
    mspec = ComponentSpec(names=("theta",), groups={})
    field = make_one_layer_vector_field(
        coordinate_spec=cspec, components=mspec, hidden=6, base="tanh", seed=3
    )
    return field(jnp.asarray(y))


def _small_layers():
    rng = np.random.default_rng(0)
    W1 = jnp.asarray(rng.normal(size=(4, 1)))
    b1 = jnp.asarray(rng.normal(size=(4,)))
    W2 = jnp.asarray(rng.normal(size=(1, 4)))
    b2 = jnp.asarray(rng.normal(size=(1,)))
    return [(W1, b1, "tanh"), (W2, b2, None)]


def test_lim_along_attribute_returns_closure_value() -> None:
    register_lim_along()
    state = _state()
    layers = _small_layers()
    x0 = jnp.zeros(1)
    v = jnp.ones(1)
    expected = asymptotic_ratio(layers, x0, v, rate=1)
    state.extra["lim_along"] = {
        "theta": lambda: asymptotic_ratio(layers, x0, v, rate=1),
    }
    got = state.theta.lim_along
    assert float(got) == pytest.approx(float(expected), abs=1e-13)


def test_lim_along_without_extra_raises() -> None:
    register_lim_along()
    state = _state()
    with pytest.raises(KeyError):
        _ = state.theta.lim_along
