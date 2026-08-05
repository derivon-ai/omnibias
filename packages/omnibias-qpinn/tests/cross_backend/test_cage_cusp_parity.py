# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity for the nuclear-cusp cage.

Feeds identical weights + nuclei/charges/rates through the torch and jax
:class:`NuclearCuspField` twins and asserts the caged value, gradient and
Laplacian agree to ``rtol=1e-9, atol=1e-12``.
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
from omnibias.qpinn.jax.cage.cusp import make_nuclear_cusp_field as make_jax_cusp
from omnibias.qpinn.torch.cage.cusp import make_nuclear_cusp_field as make_torch_cusp

from .conftest import _allclose


def _shared_3d(seed: int = 2026):
    rng = np.random.default_rng(seed)
    H, D, C = 8, 3, 2
    return {
        "H": H,
        "W": rng.normal(scale=0.5, size=(H, D)).astype(np.float64),
        "beta": rng.normal(scale=0.1, size=(H,)).astype(np.float64),
        "c": rng.normal(scale=0.5, size=(C, H)).astype(np.float64),
        "b": rng.normal(scale=0.1, size=(C,)).astype(np.float64),
        "coords": rng.normal(size=(6, D)).astype(np.float64),
        "nuclei": rng.normal(size=(2, 3)).astype(np.float64),
        "charges": np.array([1.0, 6.0], dtype=np.float64),
        "rates": np.array([0.5, 1.1], dtype=np.float64),
    }


def _build_pair(shared, riccati):
    coord = CoordinateSpec(axes=("x", "y", "z"))
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
    t_cage = make_torch_cusp(
        base=t_field,
        nuclei=torch.from_numpy(shared["nuclei"]),
        charges=torch.from_numpy(shared["charges"]),
        rates=torch.from_numpy(shared["rates"]),
    )
    j_cage = make_jax_cusp(
        base=j_field,
        nuclei=jnp.asarray(shared["nuclei"]),
        charges=jnp.asarray(shared["charges"]),
        rates=jnp.asarray(shared["rates"]),
    )
    return t_cage, j_cage


def test_cusp_cage_value_parity(riccati):
    shared = _shared_3d()
    t_cage, j_cage = _build_pair(shared, riccati)
    t_state = t_cage(torch.from_numpy(shared["coords"]))
    j_state = j_cage(jnp.asarray(shared["coords"]))
    for name in ("psi_re", "psi_im"):
        assert _allclose(
            t_state.ops.value(t_state, name), j_state.ops.value(j_state, name)
        )


def test_cusp_cage_gradient_parity(riccati):
    shared = _shared_3d()
    t_cage, j_cage = _build_pair(shared, riccati)
    t_state = t_cage(torch.from_numpy(shared["coords"]))
    j_state = j_cage(jnp.asarray(shared["coords"]))
    for name in ("psi_re", "psi_im"):
        for ax in range(3):
            assert _allclose(
                t_state.ops.derivative(t_state, name, axis=ax, order=1),
                j_state.ops.derivative(j_state, name, axis=ax, order=1),
            )


def test_cusp_cage_laplacian_parity(riccati):
    shared = _shared_3d()
    t_cage, j_cage = _build_pair(shared, riccati)
    t_state = t_cage(torch.from_numpy(shared["coords"]))
    j_state = j_cage(jnp.asarray(shared["coords"]))
    for name in ("psi_re", "psi_im"):
        assert _allclose(
            t_state.ops.laplacian(t_state, name),
            j_state.ops.laplacian(j_state, name),
        )


def test_cusp_cage_mixed_partial_parity(riccati):
    shared = _shared_3d()
    t_cage, j_cage = _build_pair(shared, riccati)
    t_state = t_cage(torch.from_numpy(shared["coords"]))
    j_state = j_cage(jnp.asarray(shared["coords"]))
    for name in ("psi_re", "psi_im"):
        assert _allclose(
            t_state.ops.mixed_partial(t_state, name, (0, 2), (1, 1)),
            j_state.ops.mixed_partial(j_state, name, (0, 2), (1, 1)),
        )
