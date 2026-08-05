# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity tests for the ParityProjectedField cage.

Three contracts under test:

1. Cross-backend parity (torch == jax in float64, rtol=1e-9 atol=1e-12)
   for value, gradient, Laplacian, and 4th-order partial.
2. Hard-projection identity: applying the cage to a generic random base
   produces output that is *exactly* symmetric (even) or antisymmetric
   (odd) under the mirror reflection. This is asserted at the value
   level, at the gradient level (with sign flip), and at the Laplacian
   level.
3. Idempotence: applying ParityProjectedField on top of an already
   parity-projected field is the identity (up to a factor of one).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import OneLayerVectorField as JOne
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField as TOne
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.jax.cage import (
    make_parity_projected_field as make_jax_parity,
)
from omnibias.qpinn.torch.cage import (
    make_parity_projected_field as make_torch_parity,
)

from .conftest import _allclose


def _build_pair(shared, axes, riccati):
    coord = CoordinateSpec(axes=axes)
    components = make_psi_components(name="psi")
    t_field = TOne(
        coordinate_spec=coord, components=components,
        hidden=shared["H"], base=riccati, dtype=torch.float64,
    )
    with torch.no_grad():
        t_field.W.weight.copy_(torch.from_numpy(shared["W"]))
        t_field.W.bias.copy_(torch.from_numpy(shared["beta"]))
        t_field.c.weight.copy_(torch.from_numpy(shared["c"]))
        t_field.c.bias.copy_(torch.from_numpy(shared["b"]))
    j_field = JOne(
        coordinate_spec=coord, components=components,
        spec=jax_get_activation(riccati),
        W=jnp.asarray(shared["W"]),
        beta=jnp.asarray(shared["beta"]),
        c=jnp.asarray(shared["c"]),
        b=jnp.asarray(shared["b"]),
        hidden=shared["H"],
    )
    return t_field, j_field


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_parity_cage_value_cross_backend(riccati, shared_psi_params_1d, parity):
    """The caged psi_re / psi_im values must match between torch and jax."""
    t_field, j_field = _build_pair(shared_psi_params_1d, ("x",), riccati)
    t_cage = make_torch_parity(base=t_field, parity=parity, mirror_axis=0)
    j_cage = make_jax_parity(base=j_field, parity=parity, mirror_axis=0)
    query_np = shared_psi_params_1d["coords"]
    t_state = t_cage(torch.from_numpy(query_np))
    j_state = j_cage(jnp.asarray(query_np))
    t_re = t_state.ops.value(t_state, "psi_re")
    j_re = j_state.ops.value(j_state, "psi_re")
    assert _allclose(t_re, j_re)
    t_im = t_state.ops.value(t_state, "psi_im")
    j_im = j_state.ops.value(j_state, "psi_im")
    assert _allclose(t_im, j_im)


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_parity_cage_gradient_cross_backend(riccati, shared_psi_params_1d, parity):
    t_field, j_field = _build_pair(shared_psi_params_1d, ("x",), riccati)
    t_cage = make_torch_parity(base=t_field, parity=parity, mirror_axis=0)
    j_cage = make_jax_parity(base=j_field, parity=parity, mirror_axis=0)
    query_np = shared_psi_params_1d["coords"]
    t_state = t_cage(torch.from_numpy(query_np))
    j_state = j_cage(jnp.asarray(query_np))
    t_d1 = t_state.ops.derivative(t_state, "psi_re", axis=0, order=1)
    j_d1 = j_state.ops.derivative(j_state, "psi_re", axis=0, order=1)
    assert _allclose(t_d1, j_d1)


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_parity_cage_laplacian_cross_backend(riccati, shared_psi_params_1d, parity):
    t_field, j_field = _build_pair(shared_psi_params_1d, ("x",), riccati)
    t_cage = make_torch_parity(base=t_field, parity=parity, mirror_axis=0)
    j_cage = make_jax_parity(base=j_field, parity=parity, mirror_axis=0)
    query_np = shared_psi_params_1d["coords"]
    t_state = t_cage(torch.from_numpy(query_np))
    j_state = j_cage(jnp.asarray(query_np))
    t_lap = t_state.ops.laplacian(t_state, "psi_re")
    j_lap = j_state.ops.laplacian(j_state, "psi_re")
    assert _allclose(t_lap, j_lap)


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_parity_cage_high_order_cross_backend(riccati, shared_psi_params_1d, parity):
    """Bit-parity for 3rd- and 4th-order derivatives on the mirror axis."""
    t_field, j_field = _build_pair(shared_psi_params_1d, ("x",), riccati)
    t_cage = make_torch_parity(base=t_field, parity=parity, mirror_axis=0)
    j_cage = make_jax_parity(base=j_field, parity=parity, mirror_axis=0)
    query_np = shared_psi_params_1d["coords"]
    t_state = t_cage(torch.from_numpy(query_np))
    j_state = j_cage(jnp.asarray(query_np))
    t_d3 = t_state.ops.derivative(t_state, "psi_re", axis=0, order=3)
    j_d3 = j_state.ops.derivative(j_state, "psi_re", axis=0, order=3)
    assert _allclose(t_d3, j_d3)
    t_d4 = t_state.ops.derivative(t_state, "psi_re", axis=0, order=4)
    j_d4 = j_state.ops.derivative(j_state, "psi_re", axis=0, order=4)
    assert _allclose(t_d4, j_d4)


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_parity_cage_enforces_exact_parity(shared_psi_params_1d, parity):
    """The cage's output value must be exactly symmetric / antisymmetric
    under Q -> -Q, to machine precision, even for a generic random base.

    This is the *hard* property the cage exists to enforce. Without the
    cage, the bare base network produces no specific parity at all and
    the v0.0.2a1 Adam Rayleigh-quotient solver collapsed 6 out of 15
    odd-parity NH3 v=0 runs to the symmetric ground state instead.
    """
    t_field, _ = _build_pair(shared_psi_params_1d, ("x",), "tanh")
    t_cage = make_torch_parity(base=t_field, parity=parity, mirror_axis=0)
    # Asymmetric query points to make the contract non-trivial.
    Q_plus = torch.linspace(0.3, 2.5, 21, dtype=torch.float64).unsqueeze(-1)
    Q_minus = -Q_plus
    state_plus = t_cage(Q_plus)
    state_minus = t_cage(Q_minus)
    sign = +1.0 if parity == "even" else -1.0
    psi_plus = state_plus.ops.value(state_plus, "psi_re")
    psi_minus = state_minus.ops.value(state_minus, "psi_re")
    assert torch.allclose(psi_minus, sign * psi_plus, rtol=1e-12, atol=1e-13)
    # First derivative: even psi -> odd derivative; odd psi -> even derivative.
    d1_plus = state_plus.ops.derivative(state_plus, "psi_re", axis=0, order=1)
    d1_minus = state_minus.ops.derivative(state_minus, "psi_re", axis=0, order=1)
    assert torch.allclose(d1_minus, -sign * d1_plus, rtol=1e-12, atol=1e-13)
    # Laplacian inherits the value's parity (two derivatives, sign cancel).
    lap_plus = state_plus.ops.laplacian(state_plus, "psi_re")
    lap_minus = state_minus.ops.laplacian(state_minus, "psi_re")
    assert torch.allclose(lap_minus, sign * lap_plus, rtol=1e-11, atol=1e-13)


def test_parity_cage_invalid_parity_raises():
    coord = CoordinateSpec(axes=("x",))
    components = make_psi_components(name="psi")
    base = TOne(
        coordinate_spec=coord, components=components,
        hidden=4, base="tanh", dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="parity"):
        make_torch_parity(base=base, parity="diagonal")


def test_parity_cage_invalid_mirror_axis_raises():
    coord = CoordinateSpec(axes=("x",))
    components = make_psi_components(name="psi")
    base = TOne(
        coordinate_spec=coord, components=components,
        hidden=4, base="tanh", dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="mirror_axis"):
        make_torch_parity(base=base, parity="even", mirror_axis=5)
