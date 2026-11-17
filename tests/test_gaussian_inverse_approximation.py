# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""A certified inverse curve serves both numerical tensor evaluators."""

from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_chebyshev import approximate_gaussian_inverse
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_tube import certify_gaussian_inverse_tube
from omnibias.core.verified.inverse_gaussian_heat import inverse_gaussian_heat
from omnibias.jax.gaussian_chebyshev import evaluate_gaussian_inverse_approximation as jax_evaluate
from omnibias.torch.gaussian_chebyshev import (
    evaluate_gaussian_inverse_approximation as torch_evaluate,
)

jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='module')
def approximation():
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (F(1, 5), F(-1, 4), F(2, 7)), time=2)
    tube = certify_gaussian_inverse_tube(model)
    assert tube is not None
    result = approximate_gaussian_inverse(tube, (F(2, 5), F(-3, 7)), (1, F(1, 3)),
                                          half_width=F(1, 4), degree=24, ellipse_parameter=2)
    assert result is not None
    return result


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_batched_backend_parity_and_exact_nominal_polynomial(approximation, dtype):
    query = np.array([[-0.25, -0.125, 0.0], [0.125, 0.25, 0.5]], dtype=dtype)
    torch_result = torch_evaluate(torch.tensor(query), approximation)
    jax_result = jax.jit(lambda t: jax_evaluate(t, approximation))(jnp.asarray(query))
    np.testing.assert_array_equal(torch_result.valid.numpy(), jax_result.valid)
    assert not bool(torch_result.valid[-1, -1])
    assert np.isnan(torch_result.value.numpy()[-1, -1]).all()
    epsilon = 2e-6 if dtype == 'float32' else 2e-13
    np.testing.assert_allclose(torch_result.value.numpy(), jax_result.value, rtol=epsilon, atol=epsilon)
    for i, j in zip(*np.where(torch_result.valid.numpy()), strict=True):
        exact = np.array(list(map(float, approximation.evaluate(F(float(query[i, j]))))))
        np.testing.assert_allclose(torch_result.value.numpy()[i, j], exact, rtol=epsilon, atol=epsilon)


def test_query_first_and_second_derivative_parity(approximation):
    query = torch.tensor([-0.2, 0.0, 0.19], dtype=torch.float64, requires_grad=True)
    value = torch_evaluate(query, approximation).value
    gradient = torch.autograd.grad(value.square().sum(), query, create_graph=True)[0]
    hessian_diagonal = torch.autograd.grad(gradient.sum(), query)[0]
    def loss(t):
        return jnp.square(jax_evaluate(t, approximation).value).sum()
    jax_query = jnp.asarray(query.detach().numpy())
    np.testing.assert_allclose(gradient.detach().numpy(), jax.grad(loss)(jax_query), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(hessian_diagonal.detach().numpy(), jnp.diag(jax.hessian(loss)(jax_query)), rtol=2e-11, atol=2e-11)


def test_polynomial_enclosure_agrees_with_independent_point_certificate(approximation):
    for t in (F(-1, 4), F(-1, 11), F(0), F(1, 9), F(1, 4)):
        target = tuple(base+t*direction for base, direction in zip(approximation.base_target, approximation.direction, strict=True))
        root = inverse_gaussian_heat(approximation.tube.model, target, absolute_tolerance=1e-12)
        assert root.certificate is not None and root.tolerance_met
        for curve_box, root_box in zip(approximation.enclose(t), root.certificate.enclosure, strict=True):
            assert curve_box.lo <= root_box.lo <= root_box.hi <= curve_box.hi
