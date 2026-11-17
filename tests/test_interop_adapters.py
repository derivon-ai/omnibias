# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepXDE and PINA adapters against the libraries' own operators (skipped when a library is not installed)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("DDE_BACKEND", "pytorch")

torch = pytest.importorskip("torch")


def _params_grad(loss, net):
    grads = torch.autograd.grad(loss, list(net.parameters()), allow_unused=True)
    return [
        torch.zeros_like(p) if g is None else g
        for p, g in zip(net.parameters(), grads, strict=True)
    ]


# ----------------------------------------------------------------------------------------------- DeepXDE


@pytest.fixture
def dde():
    dde = pytest.importorskip("deepxde")
    if dde.backend.backend_name != "pytorch":
        pytest.skip("the adapter is tested with the PyTorch backend of DeepXDE")
    dde.config.set_default_float("float64")
    return dde


def test_deepxde_laplacian_matches_dde_grad_hessian_and_trains_identically(dde):
    from omnibias.torch.deepxde_adapter import DeepXDEOperators

    torch.manual_seed(0)
    net = dde.nn.FNN([3, 16, 16, 2], "tanh", "Glorot normal")
    ops = DeepXDEOperators(net)
    x = torch.rand(7, 3, dtype=torch.float64, requires_grad=True)
    y = net(x)
    reference = sum(dde.grad.hessian(y, x, component=1, i=i, j=i) for i in range(3))
    ours = ops.laplacian(x, component=1)
    assert ours.shape == reference.shape == (7, 1)
    torch.testing.assert_close(ours, reference, rtol=1e-11, atol=1e-12)
    assert ops.chain_refusal is None and ops.explain(x, 1)["matched_patterns"]

    source = torch.linspace(-1, 1, 7, dtype=torch.float64).reshape(-1, 1)
    got = _params_grad(((ours - source) ** 2).mean(), net)
    want = _params_grad(((reference - source) ** 2).mean(), net)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
    # the plan reads live weights: an optimizer step is seen without recompiling
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05)
    y2 = net(x)
    ref2 = sum(dde.grad.hessian(y2, x, component=1, i=i, j=i) for i in range(3))
    torch.testing.assert_close(ops.laplacian(x, component=1), ref2, rtol=1e-11, atol=1e-12)


def test_deepxde_network_with_an_output_transform_runs_the_generic_route_and_says_so(dde):
    from omnibias.torch.deepxde_adapter import DeepXDEOperators

    torch.manual_seed(1)
    net = dde.nn.FNN([2, 12, 12, 1], "tanh", "Glorot normal")
    net.apply_output_transform(lambda x, y: x[:, :1] * (1 - x[:, :1]) * y)
    ops = DeepXDEOperators(net)
    x = torch.rand(5, 2, dtype=torch.float64, requires_grad=True)
    y = net(x)
    reference = sum(dde.grad.hessian(y, x, i=i, j=i) for i in range(2))
    torch.testing.assert_close(ops.laplacian(x), reference, rtol=1e-10, atol=1e-11)
    assert "output transform" in ops.chain_refusal
    assert "output transform" in ops.explain(x)["adapter_fallback"]


def test_deepxde_divergence_matches_the_jacobian_trace(dde):
    from omnibias.torch.deepxde_adapter import DeepXDEOperators

    torch.manual_seed(2)
    net = dde.nn.FNN([3, 10, 3], "tanh", "Glorot normal")
    ops = DeepXDEOperators(net)
    x = torch.rand(6, 3, dtype=torch.float64, requires_grad=True)
    y = net(x)
    reference = sum(dde.grad.jacobian(y, x, i=i, j=i) for i in range(3))
    torch.testing.assert_close(ops.divergence(x), reference, rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match="one component per input"):
        ops.divergence(x, components=[0, 1])


def test_deepxde_poisson_trains_the_same_with_the_adapter(dde):
    from omnibias.torch.deepxde_adapter import DeepXDEOperators

    def run(use_adapter):
        dde.config.set_random_seed(7)
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
        net = dde.nn.FNN([2, 20, 20, 1], "tanh", "Glorot normal")
        ops = DeepXDEOperators(net)

        def forcing(x):
            return -2 * np.pi**2 * torch.sin(np.pi * x[:, :1]) * torch.sin(np.pi * x[:, 1:])

        def pde(x, y):
            if use_adapter:
                lap = ops.laplacian(x)
            else:
                lap = dde.grad.hessian(y, x, i=0, j=0) + dde.grad.hessian(y, x, i=1, j=1)
            return lap - forcing(x)

        bc = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda _, on_boundary: on_boundary)
        data = dde.data.PDE(geom, pde, bc, num_domain=64, num_boundary=32)
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3)
        history, _ = model.train(iterations=15, display_every=1000, verbose=0)
        return np.asarray(history.loss_train)

    plain, planned = run(False), run(True)
    np.testing.assert_allclose(planned, plain, rtol=1e-7, atol=1e-9)


# ----------------------------------------------------------------------------------------------- PINA


@pytest.fixture
def pina():
    return pytest.importorskip("pina")


def test_pina_laplacian_matches_pina_operator_on_all_and_some_inputs(pina):
    from omnibias.torch.pina_adapter import explain, laplacian
    from pina import LabelTensor
    from pina.model import FeedForward
    from pina.operator import laplacian as pina_laplacian

    torch.manual_seed(0)
    model = FeedForward(
        input_dimensions=3, output_dimensions=2, inner_size=14, n_layers=2, func=torch.nn.Tanh
    ).double()
    input_ = LabelTensor(torch.rand(6, 3, dtype=torch.float64), labels=["x", "y", "t"])
    input_.requires_grad_(True)
    output_ = LabelTensor(model(input_), labels=["u", "v"])
    for components, d in ((["v"], ["x", "y", "t"]), (["u", "v"], ["x", "y"]), (["u"], ["t"])):
        want = pina_laplacian(output_, input_, components=components, d=d)
        got = laplacian(model, input_, components=components, d=d, output_labels=["u", "v"])
        assert got.labels == want.labels
        torch.testing.assert_close(
            got.as_subclass(torch.Tensor), want.as_subclass(torch.Tensor), rtol=1e-10, atol=1e-11
        )
    want_loss = (pina_laplacian(output_, input_, components=["u"], d=["x", "y", "t"]) ** 2).mean()
    got_loss = (
        laplacian(model, input_, components=["u"], output_labels=["u", "v"]).as_subclass(
            torch.Tensor
        )
        ** 2
    ).mean()
    for a, b in zip(_params_grad(got_loss, model), _params_grad(want_loss, model), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
    assert explain(model, input_)["matched_patterns"]
    with pytest.raises(ValueError, match="components"):
        laplacian(model, input_, components=["w"], output_labels=["u", "v"])


def test_pina_module_that_is_not_a_chain_runs_the_generic_route(pina):
    from omnibias.torch.pina_adapter import explain, laplacian
    from pina import LabelTensor
    from pina.operator import laplacian as pina_laplacian

    class Skip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(2, 8)
            self.b = torch.nn.Linear(8, 1)

        def forward(self, x):
            return self.b(torch.tanh(self.a(x))) + (x**3).sum(dim=-1, keepdim=True)

    torch.manual_seed(3)
    model = Skip().double()
    input_ = LabelTensor(torch.rand(5, 2, dtype=torch.float64), labels=["x", "y"]).requires_grad_(
        True
    )
    output_ = LabelTensor(model(input_), labels=["u"])
    want = pina_laplacian(output_, input_, components=["u"], d=["x", "y"])
    got = laplacian(model, input_, components=["u"], output_labels=["u"])
    torch.testing.assert_close(
        got.as_subclass(torch.Tensor), want.as_subclass(torch.Tensor), rtol=1e-10, atol=1e-11
    )
    assert not explain(model, input_)["matched_patterns"]
