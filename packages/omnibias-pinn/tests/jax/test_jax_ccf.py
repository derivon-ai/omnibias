# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CCF self-similar residual (jax): exact substitution, field op, JIT."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn._core.components import ComponentSpec  # noqa: E402
from omnibias.pinn._core.coords import CoordinateSpec  # noqa: E402
from omnibias.pinn.jax import equations as jeq  # noqa: E402
from omnibias.pinn.jax.equations.cordoba_cordoba_fontelos import (  # noqa: E402
    ccf_residual_samples,
)
from omnibias.pinn.jax.fields.one_layer import make_one_layer_vector_field  # noqa: E402


def _grid(n: int) -> np.ndarray:
    return -np.pi + 2.0 * np.pi * np.arange(n) / n


def _np_hilbert(v: np.ndarray) -> np.ndarray:
    n = v.shape[-1]
    fk = np.fft.fft(v)
    m = np.fft.fftfreq(n) * n
    mult = -1j * np.sign(m)
    if n % 2 == 0:
        mult[n // 2] = 0.0
    return np.real(np.fft.ifft(fk * mult))


def test_ccf_residual_exact_closed_form_transport() -> None:
    # theta = cos(2y): closed-form residual computed by hand.
    y = _grid(96)
    lam = 0.6057
    theta = np.cos(2 * y)
    theta_y = -2.0 * np.sin(2 * y)
    # transport: (1+lam) y theta' - lam theta + H[theta] theta'
    #   H[cos2y] = sin2y -> nonlocal = sin2y * (-2 sin2y) = -2 sin^2(2y)
    expected = (1 + lam) * y * theta_y - lam * theta - 2.0 * np.sin(2 * y) ** 2
    got = np.asarray(
        ccf_residual_samples(jnp.asarray(y), jnp.asarray(theta), jnp.asarray(theta_y), lam)
    )
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_ccf_residual_exact_closed_form_flux() -> None:
    y = _grid(96)
    lam = 0.5
    theta = np.cos(2 * y)
    theta_y = -2.0 * np.sin(2 * y)
    # flux nonlocal = theta' H[theta] + theta H[theta']
    #   H[theta]=sin2y, H[theta']=H[-2 sin2y] = -2(-cos2y)=2 cos2y
    nonlocal_term = theta_y * np.sin(2 * y) + theta * (2.0 * np.cos(2 * y))
    expected = (1 + lam) * y * theta_y - lam * theta + nonlocal_term
    got = np.asarray(
        ccf_residual_samples(
            jnp.asarray(y), jnp.asarray(theta), jnp.asarray(theta_y), lam, form="flux"
        )
    )
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_ccf_residual_velocity_sign() -> None:
    y = _grid(64)
    theta = np.cos(2 * y) + 0.3 * np.cos(4 * y)
    theta_y = -2 * np.sin(2 * y) - 1.2 * np.sin(4 * y)
    base_lin = (1 + 0.4) * y * theta_y - 0.4 * theta
    r_plus = np.asarray(ccf_residual_samples(jnp.asarray(y), jnp.asarray(theta), jnp.asarray(theta_y), 0.4, velocity_sign=1.0))
    r_minus = np.asarray(ccf_residual_samples(jnp.asarray(y), jnp.asarray(theta), jnp.asarray(theta_y), 0.4, velocity_sign=-1.0))
    # the nonlocal contributions are opposite; their average is the linear part.
    np.testing.assert_allclose(0.5 * (r_plus + r_minus), base_lin, atol=1e-10)


def test_ccf_field_op_matches_numpy_substitution() -> None:
    y = _grid(64).reshape(-1, 1)
    cspec = CoordinateSpec(axes=("y",), periodicity=(True,), time_axis=None)
    mspec = ComponentSpec(names=("theta",), groups={})
    field = make_one_layer_vector_field(
        coordinate_spec=cspec, components=mspec, hidden=6, base="tanh", seed=3
    )
    state = field(jnp.asarray(y))
    out = jeq.cordoba_cordoba_fontelos(state, lam=0.6, form="transport")
    th = np.asarray(state.ops.value(state, "theta"))
    thy = np.asarray(state.ops.derivative(state, "theta", axis="y", order=1))
    expected = (1 + 0.6) * y[:, 0] * thy - 0.6 * th + _np_hilbert(th) * thy
    np.testing.assert_allclose(np.asarray(out.residual), expected, atol=1e-10)
    # hilbert field is exposed
    np.testing.assert_allclose(np.asarray(out.hilbert), _np_hilbert(th), atol=1e-10)


def test_ccf_residual_jit() -> None:
    y = _grid(48).reshape(-1, 1)
    cspec = CoordinateSpec(axes=("y",), periodicity=(True,), time_axis=None)
    mspec = ComponentSpec(names=("theta",), groups={})
    field = make_one_layer_vector_field(
        coordinate_spec=cspec, components=mspec, hidden=5, base="tanh", seed=1
    )

    def f(field, coords):
        return jeq.cordoba_cordoba_fontelos(field(coords), lam=0.6).residual

    coords = jnp.asarray(y)
    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(np.asarray(val), np.asarray(val_jit), rtol=1e-12, atol=1e-13)


def test_ccf_rejects_time_axis() -> None:
    cspec = CoordinateSpec(axes=("y", "t"), periodicity=(True, False), time_axis="t")
    mspec = ComponentSpec(names=("theta",), groups={})
    field = make_one_layer_vector_field(
        coordinate_spec=cspec, components=mspec, hidden=4, base="tanh", seed=0
    )
    coords = jnp.asarray(np.random.default_rng(0).normal(size=(16, 2)))
    with pytest.raises(ValueError, match="steady"):
        jeq.cordoba_cordoba_fontelos(field(coords))
