# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity for incompressibility cage fields."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.jax.activations import get_activation as jax_get_activation
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.cage import (
    make_streamfunction_field,
    make_vector_potential_field,
)
from omnibias.pinn.jax.fields.spectral import SpectralVectorField as JaxField
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.cage import StreamfunctionField, VectorPotentialField
from omnibias.pinn.torch.fields.spectral import SpectralVectorField as TorchField


def _shared_spectral(
    *,
    axes: tuple[str, ...],
    periodicity: tuple[bool, ...],
    names: tuple[str, ...],
    k: int,
    time_hidden: int,
    n_pts: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_spatial = sum(1 for axis in axes if axis != "t")
    modes = (2 * k + 1) ** n_spatial
    out_dim = len(names) * modes
    return {
        "axes": axes,
        "periodicity": periodicity,
        "names": names,
        "K": k,
        "time_hidden": time_hidden,
        "W_t": rng.normal(scale=0.5, size=(time_hidden, 1)).astype(np.float64),
        "beta_t": rng.normal(scale=0.1, size=(time_hidden,)).astype(np.float64),
        "V": rng.normal(scale=0.2, size=(out_dim, time_hidden)).astype(np.float64),
        "b_t": rng.normal(scale=0.1, size=(out_dim,)).astype(np.float64),
        "coords": rng.uniform(0.05, 1.5, size=(n_pts, len(axes))).astype(np.float64),
    }


def _make_torch(shared: dict[str, object]) -> tuple[TorchField, torch.Tensor]:
    cspec = CoordinateSpec(
        axes=shared["axes"],
        periodicity=shared["periodicity"],
        time_axis="t",
    )
    field = TorchField(
        coordinate_spec=cspec,
        components=ComponentSpec(shared["names"]),
        K=shared["K"],
        time_hidden=shared["time_hidden"],
        time_depth=1,
        activation="tanh",
        dtype=torch.float64,
    )
    with torch.no_grad():
        field.W_t.copy_(torch.from_numpy(shared["W_t"]))
        field.beta_t.copy_(torch.from_numpy(shared["beta_t"]))
        field.V.copy_(torch.from_numpy(shared["V"]))
        field.b_t.copy_(torch.from_numpy(shared["b_t"]))
    return field, torch.from_numpy(shared["coords"])


def _make_jax(shared: dict[str, object]) -> tuple[JaxField, jax.Array]:
    cspec = CoordinateSpec(
        axes=shared["axes"],
        periodicity=shared["periodicity"],
        time_axis="t",
    )
    n_spatial = len(cspec.spatial_axes)
    field = JaxField(
        coordinate_spec=cspec,
        components=ComponentSpec(shared["names"]),
        spec=jax_get_activation("tanh"),
        W_t=jnp.asarray(shared["W_t"]),
        beta_t=jnp.asarray(shared["beta_t"]),
        inner_W=tuple(),
        inner_b=tuple(),
        V=jnp.asarray(shared["V"]),
        b_t=jnp.asarray(shared["b_t"]),
        K=shared["K"],
        L=tuple(2.0 * math.pi for _ in range(n_spatial)),
        time_hidden=shared["time_hidden"],
        time_depth=1,
    )
    return field, jnp.asarray(shared["coords"])


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def test_streamfunction_cage_cross_backend_parity() -> None:
    shared = _shared_spectral(
        axes=("t", "x", "y"),
        periodicity=(False, True, True),
        names=("psi", "p"),
        k=3,
        time_hidden=5,
        n_pts=7,
        seed=120,
    )
    t_base, t_coords = _make_torch(shared)
    j_base, j_coords = _make_jax(shared)
    t_cage = StreamfunctionField(
        base=t_base, psi="psi", velocity_names=("u", "v"),
        passthrough_names=("p",), spatial_axes=("x", "y"),
    )
    j_cage = make_streamfunction_field(
        base=j_base, psi="psi", velocity_names=("u", "v"),
        passthrough_names=("p",), spatial_axes=("x", "y"),
    )
    ts = t_cage(t_coords)
    js = j_cage(j_coords)
    np.testing.assert_allclose(_np(ts.u.value), _np(js.u.value), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_np(ts.v.value), _np(js.v.value), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(_np(ts.p.value), _np(js.p.value), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        _np(tops.divergence(ts, ("u", "v"))),
        _np(jops.divergence(js, ("u", "v"))),
        rtol=1e-9,
        atol=1e-12,
    )
    assert float(np.max(np.abs(_np(ts.velocity.div)))) < 1e-11


def test_vector_potential_cage_cross_backend_parity() -> None:
    shared = _shared_spectral(
        axes=("t", "x", "y", "z"),
        periodicity=(False, True, True, True),
        names=("A1", "A2", "A3", "p"),
        k=2,
        time_hidden=4,
        n_pts=5,
        seed=121,
    )
    t_base, t_coords = _make_torch(shared)
    j_base, j_coords = _make_jax(shared)
    t_cage = VectorPotentialField(
        base=t_base, A_components=("A1", "A2", "A3"),
        velocity_names=("u", "v", "w"), passthrough_names=("p",),
        spatial_axes=("x", "y", "z"),
    )
    j_cage = make_vector_potential_field(
        base=j_base, A_components=("A1", "A2", "A3"),
        velocity_names=("u", "v", "w"), passthrough_names=("p",),
        spatial_axes=("x", "y", "z"),
    )
    ts = t_cage(t_coords)
    js = j_cage(j_coords)
    for name in ("u", "v", "w", "p"):
        np.testing.assert_allclose(
            _np(getattr(ts, name).value),
            _np(getattr(js, name).value),
            rtol=1e-8,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        _np(tops.divergence(ts, ("u", "v", "w"))),
        _np(jops.divergence(js, ("u", "v", "w"))),
        rtol=1e-8,
        atol=1e-12,
    )
    assert float(np.max(np.abs(_np(ts.velocity.div)))) < 1e-10
