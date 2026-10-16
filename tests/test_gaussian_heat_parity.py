# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend parity and rigorous validation of a numerical implicit gradient."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_sensitivity import certify_gaussian_target_vjp
from omnibias.jax.gaussian_heat import gaussian_heat_inverse as jax_inverse
from omnibias.torch.gaussian_heat import gaussian_heat_inverse as torch_inverse


@pytest.mark.parametrize("adjoint", ["dense", "low_rank", "auto"])
def test_values_and_all_physical_parameter_gradients_match(adjoint):
    data = (np.array([[0.25, -0.5], [0.125, 0.375]]),
            np.array([[0., 0.], [1., 0.], [0., 1.], [0.5, -0.5]]),
            np.array([0., 0.25, -0.5, 0.125]), np.array(1.5), np.array(2.0))
    args_t = tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in data)
    args_j = tuple(jnp.asarray(x, dtype=jnp.float64) for x in data)
    output = torch_inverse(*args_t, tolerance=1e-12, adjoint=adjoint)
    assert bool(torch.all(output.converged))
    grad_t = torch.autograd.grad((output.value**2).sum(), args_t)

    def loss(*args):
        return jnp.sum(jax_inverse(*args, tolerance=1e-12, adjoint=adjoint).value**2)

    grad_j = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(*args_j)
    value_j = jax.jit(lambda *args: jax_inverse(*args, tolerance=1e-12, adjoint=adjoint))(*args_j)
    assert np.all(value_j.converged)
    np.testing.assert_allclose(output.value.detach().numpy(), value_j.value, rtol=2e-12, atol=2e-12)
    for actual, expected in zip(grad_t, grad_j, strict=True):
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=3e-10, atol=3e-11)
    # Common bias offsets do not change a mixture; translations shift its inverse.
    assert abs(float(grad_t[2].sum())) < 1e-11
    np.testing.assert_allclose(grad_t[1].sum(dim=0).detach().numpy(),
                               (2*output.value).sum(dim=0).detach().numpy(), rtol=3e-10, atol=3e-11)


def test_second_target_derivatives_match_across_implicit_rules():
    centers = np.array([[-1., 0.], [1., 0.5], [0., 1.]])
    biases = np.array([0.25, 0., -0.25])
    target = np.array([0.25, -0.125])

    def loss_t(v):
        value = torch_inverse(v, torch.tensor(centers), torch.tensor(biases), 1.5, 2.,
                              tolerance=1e-12, adjoint="low_rank").value
        return (value**2).sum()

    def loss_j(v):
        return jnp.sum(jax_inverse(v, jnp.asarray(centers), jnp.asarray(biases), 1.5, 2.,
                                  tolerance=1e-12, adjoint="low_rank").value**2)

    hessian_t = torch.autograd.functional.hessian(loss_t, torch.tensor(target, requires_grad=True))
    hessian_j = jax.jit(jax.hessian(loss_j))(jnp.asarray(target))
    np.testing.assert_allclose(hessian_t.detach().numpy(), hessian_j, rtol=2e-9, atol=2e-10)


def test_tensor_target_gradient_has_a_rigorous_independent_certificate():
    mp = pytest.importorskip("mpmath")
    target = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    centers = torch.tensor([[-1.], [1.]], dtype=torch.float64)
    biases = torch.tensor([0., 0.], dtype=torch.float64)
    result = torch_inverse(target, centers, biases, 1., 2., tolerance=1e-12)
    assert bool(result.converged)
    gradient, = torch.autograd.grad(0.75*result.value.sum(), (target,))
    # Every stored binary64 datum is injected exactly; no decimal rational guess.
    model = GaussianHeatMap(tuple(tuple(Fraction(value) for value in row) for row in centers.tolist()),
                            tuple(Fraction(value) for value in biases.tolist()), viscosity=1, time=2)
    certificate = certify_gaussian_target_vjp(
        model, tuple(Fraction(value) for value in target.tolist()),
        tuple(value/2 for value in result.value.tolist()), (0.75,), tuple(gradient.tolist()),
    )
    assert certificate is not None and certificate.error_bound < 1e-9
    with mp.workdps(80):
        root = mp.findroot(lambda x: (x-mp.tanh(x/4))/2-mp.mpf(1)/4, 1)
        expected = mp.mpf("1.5")/(1-mp.sech(root/4)**2/4)
        interval = certificate.enclosure[0]
        assert mp.mpf(interval.lo) <= expected <= mp.mpf(interval.hi)
        assert abs(expected-mp.mpf(gradient.item())) <= certificate.error_bound


def test_fourth_order_inverse_derivatives_against_high_precision_solution():
    mp = pytest.importorskip("mpmath")
    target = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    value = torch_inverse(target, [[-1.], [1.]], [0., 0.], time=2., tolerance=1e-12).value[0]

    def scalar_inverse(v):
        return jax_inverse(jnp.reshape(v, (1,)), [[-1.], [1.]], [0., 0.], time=2., tolerance=1e-12).value[0]

    derivative_j = scalar_inverse
    with mp.workdps(80):
        def exact_inverse(v):
            return mp.findroot(lambda x: x-mp.tanh(x/4)-2*v, 1)
        for order in range(1, 5):
            value, = torch.autograd.grad(value, target, create_graph=True)
            derivative_j = jax.grad(derivative_j)
            expected = float(mp.diff(exact_inverse, mp.mpf(1)/4, order))
            np.testing.assert_allclose(float(value.item()), expected, rtol=3e-9, atol=2e-10)
            np.testing.assert_allclose(float(jax.jit(derivative_j)(jnp.array(0.25))), expected,
                                       rtol=3e-9, atol=2e-10)


def test_both_backends_refuse_uncertified_numerical_contraction_regime():
    target = np.array([0.])
    centers = np.array([[-1.], [1.]])
    biases = np.zeros(2)
    output_t = torch_inverse(torch.tensor(target), torch.tensor(centers), torch.tensor(biases), time=0.25)
    output_j = jax.jit(lambda v: jax_inverse(v, jnp.asarray(centers), jnp.asarray(biases), time=0.25))(jnp.asarray(target))
    assert not bool(output_t.converged) and not bool(output_j.converged)
    assert torch.all(torch.isnan(output_t.value)) and np.all(np.isnan(output_j.value))
