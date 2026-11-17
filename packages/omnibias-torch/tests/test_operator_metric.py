# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent dense-AD checks of the transported operator identities."""

import pytest
import torch
from omnibias.torch.experimental._operator_metric import (
    MetricState,
    laplacian_activation,
    laplacian_affine,
    laplacian_seed,
    metric_activation,
    metric_affine,
    metric_laplacian,
    metric_multiply,
    metric_seed,
)


def _oracle(function, x, gram):
    def lap(z):
        return (torch.func.hessian(function)(z) * gram).sum()
    hessian = torch.func.hessian(function)(x)
    d = x.numel()
    return MetricState(
        function(x), torch.func.grad(function)(x),
        torch.stack([hessian[i, j] for i in range(d) for j in range(i, d)]),
        torch.func.grad(lap)(x), (torch.func.hessian(lap)(x) * gram).sum(),
    )


@pytest.mark.parametrize("gram_data", [[[2., .3], [.3, 1.]], [[2., .3], [.3, -1.]]])
def test_weighted_product_and_composition(gram_data):
    x = torch.tensor([.2, -.3], dtype=torch.float64)
    gram = torch.tensor(gram_data, dtype=x.dtype, requires_grad=True)

    def u(z):
        return z[0]**4 * z[1] + z[0]**2 * z[1]**2 + z[1]**3

    def v(z):
        return z[0] * z[1]**4 + z[0]**3 + z[0] * z[1]

    a, b = _oracle(u, x, gram), _oracle(v, x, gram)
    actual = metric_multiply(a, b, gram, 2)
    wanted = _oracle(lambda z: u(z) * v(z), x, gram)
    for result, reference in zip(actual, wanted, strict=True):
        torch.testing.assert_close(result, reference, rtol=2e-12, atol=2e-12)
    actual = metric_activation(actual, gram, 2, "tanh")
    wanted = _oracle(lambda z: torch.tanh(u(z) * v(z)), x, gram)
    for result, reference in zip(actual, wanted, strict=True):
        torch.testing.assert_close(result, reference, rtol=2e-12, atol=2e-12)


def test_affine_transport_keeps_both_parameter_paths():
    torch.manual_seed(12)
    x = torch.tensor([.2, -.3, .1], dtype=torch.float64)
    basis = (torch.randn(2, 3, dtype=x.dtype) * .3).requires_grad_()
    weight = (torch.randn(3, 2, dtype=x.dtype) * .2).requires_grad_()
    bias = torch.randn(3, dtype=x.dtype, requires_grad=True)
    readout = torch.randn(1, 3, dtype=x.dtype, requires_grad=True)
    gram = basis @ basis.T

    def primal(z):
        return (readout @ torch.tanh(weight @ torch.tanh(basis @ z) + bias)).squeeze()

    state = metric_activation(metric_seed(basis @ x, 2), gram, 2)
    state = metric_activation(metric_affine(state, weight, bias), gram, 2)
    actual = metric_affine(state, readout).biharmonic.squeeze()
    reference = _oracle(primal, x, torch.eye(3, dtype=x.dtype)).biharmonic
    torch.testing.assert_close(actual, reference, rtol=2e-11, atol=2e-12)
    params = (basis, weight, bias, readout)
    grads = torch.autograd.grad(actual.square(), params, retain_graph=True)
    ref_grads = torch.autograd.grad(reference.square(), params)
    for gradient, wanted in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(gradient, wanted, rtol=2e-10, atol=2e-11)


def test_laplacian_and_biharmonic_trace_agree():
    x = torch.tensor([[.2, -.3], [-.1, .4]], dtype=torch.float64)
    weight = torch.tensor([[.5, .1], [-.2, .3]], dtype=x.dtype)
    gram = torch.tensor([[2., .3], [.3, 1.]], dtype=x.dtype)
    second = laplacian_activation(laplacian_affine(laplacian_seed(x, 2), weight), gram, "sin")
    fourth = metric_activation(metric_affine(metric_seed(x, 2), weight), gram, 2, "sin")
    torch.testing.assert_close(second.value, fourth.value)
    torch.testing.assert_close(second.gradient, fourth.gradient)
    torch.testing.assert_close(second.laplacian, metric_laplacian(fourth, gram, 2))


def test_transported_kernel_compiles_and_differentiates():
    x = torch.tensor([[.2, -.3]], dtype=torch.float64)
    basis = torch.tensor([[.4, -.2], [.1, .5]], dtype=x.dtype, requires_grad=True)

    def evaluate(value, matrix):
        z = value @ matrix.T
        return metric_activation(metric_seed(z, 2), matrix @ matrix.T, 2, "sin").biharmonic

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    actual = compiled(x, basis)
    torch.testing.assert_close(actual, evaluate(x, basis))
    assert torch.isfinite(torch.autograd.grad(actual.sum(), basis)[0]).all()
