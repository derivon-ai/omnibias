# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for :mod:`omnibias.pinn.jax.diagnostics` (twin)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.components import ComponentSpec  # noqa: E402
from omnibias.pinn._core.coords import CoordinateSpec  # noqa: E402
from omnibias.pinn.jax.diagnostics import (  # noqa: E402
    autograd_phase_check,
    derivative_stability,
)
from omnibias.pinn.jax.fields.one_layer import (  # noqa: E402
    make_one_layer_vector_field,
)


def _one_layer_field_2d(seed: int = 0):
    coord = CoordinateSpec(("x", "y", "t"))
    components = ComponentSpec(("u", "v", "p"), groups={"velocity": ("u", "v")})
    return make_one_layer_vector_field(
        coordinate_spec=coord, components=components,
        hidden=4, base="tanh", seed=seed, dtype=jnp.float64,
    )


def test_derivative_stability_closed_form_matches_autograd():
    field = _one_layer_field_2d(seed=0)
    coords = jnp.asarray(np.random.default_rng(0).standard_normal((4, 3)))
    rows = derivative_stability(field, coords, component="u", max_order=2)
    assert len(rows) == 2
    for row in rows:
        if row.closed_form == row.closed_form:
            assert row.rel_diff < 1e-9, (
                f"order {row.order} rel_diff={row.rel_diff} "
                f"abs_diff={row.abs_diff}"
            )


def test_autograd_phase_check_returns_one_row_per_order():
    field = _one_layer_field_2d(seed=1)
    coords = jnp.asarray(np.random.default_rng(1).standard_normal((3, 3)))
    rows = autograd_phase_check(
        field, coords, component="u", max_order=2, repeats=2,
    )
    assert len(rows) == 2
    for row in rows:
        assert row.closed_form_seconds >= 0 or row.closed_form_seconds != row.closed_form_seconds
        assert row.autograd_seconds >= 0


def test_derivative_stability_validates_max_order():
    field = _one_layer_field_2d(seed=2)
    coords = jnp.asarray(np.random.default_rng(2).standard_normal((3, 3)))
    with pytest.raises(ValueError, match="max_order must be"):
        derivative_stability(field, coords, max_order=0)


def test_autograd_phase_check_validates_repeats():
    field = _one_layer_field_2d(seed=3)
    coords = jnp.asarray(np.random.default_rng(3).standard_normal((3, 3)))
    with pytest.raises(ValueError, match="repeats must be"):
        autograd_phase_check(field, coords, repeats=0)
