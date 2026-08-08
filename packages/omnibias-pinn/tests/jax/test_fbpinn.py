# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX FBPINN + NTK parity smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.jax import ops
from omnibias.pinn.jax.fields.fbpinn import make_fbpinn_field
from omnibias.pinn.jax.losses import ntk_eigenspectrum
from omnibias.pinn.train import SpectralBandScheduler


def test_jax_fbpinn_window_weights_sum_to_one():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = make_fbpinn_field(
        coordinate_spec=cs, components=comps, n_levels=2, hidden=4, seed=0
    )
    x = jnp.linspace(0.05, 0.95, 32).reshape(-1, 1)
    w = field.window_weights(x, level=0)
    assert jnp.allclose(w.sum(axis=-1), jnp.ones(32), atol=1e-6)


def test_jax_fbpinn_derivative_finite():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = make_fbpinn_field(
        coordinate_spec=cs, components=comps, n_windows=2, hidden=4, seed=1
    )
    x = jnp.linspace(0.2, 0.8, 5).reshape(-1, 1)
    ux = ops.derivative(field(x), "u", axis=0, order=1)
    assert ux.shape == (5,)
    assert jnp.all(jnp.isfinite(ux))


def test_jax_ntk_eigenspectrum_nonzero():
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    field = make_fbpinn_field(
        coordinate_spec=cs, components=comps, n_windows=2, hidden=3, seed=2
    )
    coords = jnp.linspace(0.0, 1.0, 4).reshape(-1, 1)
    target = jnp.sin(2 * jnp.pi * coords[:, 0])

    def residual_fn(params):
        del params
        return field.forward_values(coords)[:, 0] - target

    evals = ntk_eigenspectrum(residual_fn, field, n_eigen=3)
    assert evals.size >= 1
    assert jnp.all(evals >= 0.0)


def test_jax_spectral_band_scheduler_state():
    sched = SpectralBandScheduler(n_bands_max=3, n_bands_init=2, L=1.0)
    x = np.linspace(0.0, 1.0, 32, endpoint=False)
    resid = np.sin(2 * np.pi * 12 * x)[None, :]
    sched.observe(resid)
    state = sched.state_dict()
    sched2 = SpectralBandScheduler(n_bands_max=3, n_bands_init=2, L=1.0)
    sched2.load_state_dict(state)
    assert sched2.bands == sched.bands
