# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
import pytest
import torch
from omnibias.torch.biharmonic import (
    BiharmonicState,
    biharmonic_activation,
    biharmonic_multiply,
    biharmonic_seed,
    mlp_biharmonic,
)


def _oracle(f, x):
    h = torch.func.hessian(f)(x)
    def lap(z):
        return torch.func.hessian(f)(z).trace()
    packed = torch.stack([h[i, j] for i in range(x.numel()) for j in range(i, x.numel())])
    return BiharmonicState(f(x), torch.func.grad(f)(x), packed,
                           torch.func.grad(lap)(x), torch.func.hessian(lap)(x).trace())


def test_product_and_activation_with_multivariate_polynomials():
    x = torch.tensor([.2, -.3, .4], dtype=torch.float64)
    def u(z):
        a, b, c = z.unbind()
        return a**4*c + a*a*b + a*b*c + a*c*c + b*b*c + b
    def v(z):
        a, b, c = z.unbind()
        return a*a*c + a*b**4 + a*b*b + a*c + a + b*b + b*c*c
    a, b = _oracle(u, x), _oracle(v, x)
    product = biharmonic_multiply(a, b)
    expected = _oracle(lambda z: u(z)*v(z), x)
    for actual, ref in zip(product, expected, strict=True):
        torch.testing.assert_close(actual, ref, rtol=1e-11, atol=1e-11)
    transformed = biharmonic_activation(a)
    expected = _oracle(lambda z: torch.tanh(u(z)), x)
    for actual, ref in zip(transformed, expected, strict=True):
        torch.testing.assert_close(actual, ref, rtol=1e-11, atol=1e-11)


def test_mlp_biharmonic_and_parameter_gradients():
    torch.manual_seed(9)
    w = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, dtype=torch.float64, requires_grad=True)
    readout = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([.1, -.2], dtype=torch.float64)
    def f(z):
        return (readout @ torch.tanh(w @ z + b)).squeeze()
    actual = mlp_biharmonic(x, [(w, b, "tanh"), (readout, None, None)])
    ref = _oracle(f, x)
    for a, r in zip(actual, ref, strict=True):
        torch.testing.assert_close(a.squeeze(0), r, rtol=2e-11, atol=2e-11)
    actual_grad = torch.autograd.grad(actual.biharmonic.square().sum(), (w, b, readout), retain_graph=True)
    ref_grad = torch.autograd.grad(ref.biharmonic.square(), (w, b, readout))
    for a, r in zip(actual_grad, ref_grad, strict=True):
        torch.testing.assert_close(a, r, rtol=5e-10, atol=5e-10)


def test_batch_and_budget():
    x = torch.randn(4, 2, dtype=torch.float64)
    w = torch.ones(3, 2, dtype=x.dtype)
    state = mlp_biharmonic(x, [(w, None, "tanh")])
    assert state.value.shape == (4, 3)
    assert state.hessian.shape == (4, 3, 3)
    expected = 4 * torch.stack([torch.func.hessian(torch.func.hessian(lambda z: torch.tanh(z).sum()))(w @ row)[0, 0, 0, 0] for row in x])
    # Each ridge has ||w||^4=4 and the same preactivation.
    torch.testing.assert_close(state.biharmonic[:, 0], expected)
    with pytest.raises(ValueError, match="budget"):
        mlp_biharmonic(x, [(w, None, "tanh")], max_state_bytes=1)
    with pytest.raises(TypeError, match="float32 or float64"):
        biharmonic_seed(x.half())


@pytest.mark.filterwarnings("ignore:Dynamo detected a call to a `functools.lru_cache`-wrapped function:UserWarning")
def test_compiled_biharmonic():
    x = torch.tensor([[.2, -.1]], dtype=torch.float64)
    w = torch.tensor([[.5, -.3]], dtype=torch.float64, requires_grad=True)
    def f(z, weight):
        return mlp_biharmonic(z, [(weight, None, "tanh")]).biharmonic
    compiled = torch.compile(f, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, w), f(x, w))
