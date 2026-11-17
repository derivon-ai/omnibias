# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent high-precision values, attained collapse and tail Hessians."""

import mpmath as mp
import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.jax.confluence import normalized_sigmoid_band as jband
from omnibias.torch.confluence import normalized_sigmoid_band as tband


def reference(z, h, beta):
    return (1 + mp.cosh(beta * h)) / (mp.cosh(beta * z) + mp.cosh(beta * h))


@pytest.mark.parametrize("width", [0.0, 1e-12, 0.3, 20.0, -0.3])
@pytest.mark.parametrize("beta", [0.0, 0.7, 4.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_values_with_independent_hyperbolic_oracle(width, beta, dtype):
    z = np.array([-100.0, -5.0, -0.1, 0.0, 0.7, 5.0, 100.0], dtype=dtype)
    with mp.workdps(70):
        expected = np.array(
            [float(reference(mp.mpf(float(v)), mp.mpf(width), mp.mpf(beta))) for v in z],
            dtype=dtype,
        )
    actual = tband(torch.tensor(z), width, beta).numpy()
    compiled = jax.jit(lambda x, h, b: jband(x, h, b))(jnp.asarray(z), width, beta)
    tolerance = 2e-5 if dtype is np.float32 else 4e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(compiled, expected, rtol=tolerance, atol=tolerance)


def test_attained_zero_width_curvature_and_parameter_gradients():
    z = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    h = torch.tensor(0.0, dtype=z.dtype, requires_grad=True)
    beta = torch.tensor(2.0, dtype=z.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(tband, (z, h, beta))
    assert torch.autograd.gradgradcheck(tband, (z, h, beta))
    value = tband(z, h, beta)
    gradient = torch.autograd.grad(value, h, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient, h)[0]
    assert gradient.item() == pytest.approx(0.0, abs=1e-15)
    assert curvature.item() == pytest.approx((beta**2 * value * (1 - value) / 2).item(), rel=2e-12)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
def test_high_derivatives_of_the_collapsed_profile(order):
    z = torch.tensor(0.43, dtype=torch.float64, requires_grad=True)
    value = tband(z, 0.0, 1.7)
    for _ in range(order):
        value = torch.autograd.grad(value, z, create_graph=True)[0]
    with mp.workdps(70):
        expected = float(
            mp.diff(lambda x: reference(x, mp.mpf(0), mp.mpf("1.7")), mp.mpf("0.43"), order)
        )
    assert value.item() == pytest.approx(expected, rel=2e-10, abs=1e-12)


def test_float32_tail_hessians_remain_finite():
    z = torch.tensor([100.0, -100.0], dtype=torch.float32, requires_grad=True)
    h = torch.tensor(0.0, dtype=z.dtype, requires_grad=True)
    value = tband(z, h)
    gradient = torch.autograd.grad(value.sum(), (z, h), create_graph=True)
    for g in gradient:
        for second in torch.autograd.grad(g.sum(), (z, h), retain_graph=True):
            assert torch.isfinite(second).all()
    # On CPU Torch retains a representable subnormal, unlike two saturated sigmoids.
    assert torch.all(value > 0)

    def f(x):
        return jband(x, jnp.asarray(0.0, dtype=x.dtype)).sum()

    assert np.isfinite(jax.jit(jax.hessian(f))(jnp.array([100.0, -100.0], dtype=jnp.float32))).all()


def test_broadcast_vmap_and_live_width_temperature():
    z = np.array([[-0.4], [0.8]])
    h = np.array([[0.0, 0.2, 0.9]])
    tz, th = torch.tensor(z, requires_grad=True), torch.tensor(h, requires_grad=True)
    tb = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    actual = tband(tz, th, tb)
    tg = torch.autograd.grad(actual.square().sum(), (tz, th, tb))
    jg = jax.jit(jax.grad(lambda x, w, b: jnp.square(jband(x, w, b)).sum(), argnums=(0, 1, 2)))(
        jnp.asarray(z), jnp.asarray(h), jnp.asarray(2.0)
    )
    for a, b in zip(tg, jg, strict=True):
        np.testing.assert_allclose(a.detach(), b, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(lambda row: jband(row, jnp.asarray(h[0]), 2.0)))(jnp.asarray(z)),
        actual.detach(),
    )


def test_integer_inputs_are_refused():
    with pytest.raises(TypeError):
        tband(torch.tensor([1]), 0.0)
    with pytest.raises(TypeError):
        jband(jnp.array([1]), 0.0)


@pytest.mark.parametrize("width", [0.0, 0.3, 1.2])
def test_reported_slab_level_set_and_mass_against_quadrature(width):
    beta, threshold = 2.3, 0.4
    with mp.workdps(50):
        h, b, tau = mp.mpf(width), mp.mpf(beta), mp.mpf(threshold)
        radius = mp.acosh((1 + (1 - tau) * mp.cosh(b * h)) / tau) / b
        mass = mp.quad(lambda z: reference(z, h, b), [-mp.inf, 0, mp.inf])
        expected_mass = 4 / b if width == 0 else 2 * h / mp.tanh(b * h / 2)
        assert mp.almosteq(mass, expected_mass)
    radius_tensor = torch.tensor(float(radius), dtype=torch.float64)
    assert tband(radius_tensor, width, beta).item() == pytest.approx(threshold, rel=2e-13)
    assert tband(radius_tensor / 2, width, beta) > threshold
    assert tband(radius_tensor * 2, width, beta) < threshold


def test_signed_temperature_symmetry():
    z = torch.tensor([-0.7, 0.0, 1.2], dtype=torch.float64)
    torch.testing.assert_close(tband(z, 0.4, -2.0), tband(z, 0.4, 2.0))
    np.testing.assert_allclose(jband(jnp.asarray(z), 0.4, -2.0), tband(z, 0.4, 2.0))
