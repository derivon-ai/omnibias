# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parameter higher AD uses independent formulas and live model state."""

import pytest
import torch
from omnibias.torch.experimental import (
    JetSpec,
    OperatorSpec,
    compile_jet,
    compile_operator,
    directional_seed,
)
from torch import nn


class Ridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
        self.alias = self.weight
        self.bias = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.register_buffer("scale", torch.tensor(1.2, dtype=torch.float64))

    def forward(self, x):
        z = self.weight * x + self.bias
        return {"sine": self.scale * torch.sin(z) + self.alias * x,
                "exp": torch.exp(z)}


def formula(model, x, direction):
    z = model.weight * x + model.bias
    slope = model.weight * direction
    return {"sine": -model.scale * slope.square() * torch.sin(z),
            "exp": slope.square() * torch.exp(z)}


def loss(output, target=0.0):
    return (output["sine"] - target).square().sum() + output["exp"].square().sum()


def second_gradients(value, variables):
    first = torch.autograd.grad(value, variables, create_graph=True)
    second = torch.autograd.grad(sum(g.sum() for g in first), variables)
    return first, second


def test_hybrid_shared_parameters_pytree_and_dependent_direction():
    model = Ridge()
    x = torch.tensor([[0.2], [-0.1]], dtype=torch.float64, requires_grad=True)
    program = compile_operator(model, (x,), operator=OperatorSpec.directional(2), batch_ndim=1)
    hybrid = program.hybrid(backend="aot_eager")
    assert {id(p) for p in hybrid.parameters()} == {id(p) for p in program.parameters()}
    direction = 2 * x
    actual = hybrid(x, direction=direction)
    expected = formula(model, x, 2*x)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
    variables = (*model.parameters(), x)
    got = second_gradients(loss(actual), variables)
    wanted = second_gradients(loss(expected), variables)
    for actual_order, expected_order in zip(got, wanted, strict=True):
        for a, b in zip(actual_order, expected_order, strict=True):
            torch.testing.assert_close(a, b, rtol=3e-12, atol=3e-12)
    assert hybrid.explain()["outer_fullgraph"] is False


def test_hybrid_repeated_arguments_keep_identity_aliases():
    class Product(nn.Module):
        def forward(self, a, b):
            return a * b + torch.sin(a)

    x = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    spec = JetSpec.directional(2)
    program = compile_jet(Product(), (x, x.detach().clone().requires_grad_()), spec=spec, argnums=(0, 1))
    hybrid = program.hybrid(backend="aot_eager")
    packed = directional_seed(x, 2*x, spec)
    actual = hybrid(packed, packed)[2]
    expected = 4*x.square() - 2*x.square()*torch.sin(x)
    got = second_gradients(actual, (x,))
    wanted = second_gradients(expected, (x,))
    for actual_order, expected_order in zip(got, wanted, strict=True):
        torch.testing.assert_close(actual_order[0], expected_order[0])


def test_hybrid_live_buffers_version_checks_and_outer_hooks():
    model = Ridge()
    x = torch.tensor([[0.2]], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.directional(2), batch_ndim=1)
    hybrid = program.hybrid(backend="aot_eager")
    calls = []
    hybrid.register_forward_hook(lambda *_: calls.append(True))
    model.scale.add_(0.3)
    actual = hybrid(x, direction=x)
    torch.testing.assert_close(actual["sine"], formula(model, x, x)["sine"])
    assert calls == [True]
    model.scale.add_(0.2)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss(actual).backward()


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_hvp_matches_analytic_loss_and_preserves_grads(backend):
    model = Ridge()
    x = torch.tensor([[0.2], [-0.1]], dtype=torch.float64)
    direction = torch.tensor([[0.9], [0.8]], dtype=x.dtype, requires_grad=True)
    target = torch.tensor(0.02, dtype=x.dtype)
    program = compile_operator(model, (x,), operator=OperatorSpec.directional(2), batch_ndim=1)
    program.compile(backend="aot_eager", fullgraph=True)
    program(x, direction=direction)
    # The canonical native graph must remain usable after Module.compile().
    hvp = program.compile_parameter_hvp(loss, backend=backend)
    params = dict(program.named_parameters())
    tangents = {name: torch.full_like(p, (i+1)/3) for i, (name, p) in enumerate(params.items())}
    for p in params.values():
        p.grad = torch.full_like(p, 7.0)
    expected_gradient = torch.autograd.grad(loss(formula(model, x, direction), target), tuple(params.values()),
                                             create_graph=True)
    expected = torch.autograd.grad(sum((g*tangents[name]).sum() for name, g in zip(params, expected_gradient, strict=True)),
                                   tuple(params.values()))
    actual = hvp(tangents, x, direction=direction, loss_args=(target,))
    assert actual.keys() == params.keys()
    for name, wanted in zip(params, expected, strict=True):
        torch.testing.assert_close(actual[name], wanted, rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(params[name].grad, torch.full_like(params[name], 7.0))
    with torch.no_grad():
        model.weight.add_(0.1)
        model.scale.add_(0.2)
    second = hvp(tangents, x, direction=direction, loss_args=(target,))
    assert any(not torch.equal(actual[name], second[name]) for name in params)
    assert direction.grad is None


def test_hvp_tangent_and_loss_contracts():
    model = nn.Linear(1, 1, dtype=torch.float64)
    x = torch.tensor([[0.2]], dtype=torch.float64)
    program = compile_jet(model, (x,), spec=JetSpec.directional(1))
    hvp = program.compile_parameter_hvp(lambda out: out.sum(), backend="aot_eager")
    packed = directional_seed(x, x, program.spec)
    with pytest.raises(ValueError, match="exactly"):
        hvp({}, packed)
    tangents = {name: torch.ones_like(p) for name, p in program.named_parameters()}
    result = hvp(tangents, packed)
    assert all(torch.equal(value, torch.zeros_like(value)) for value in result.values())
    name = next(iter(tangents))
    tangents[name] = tangents[name].float()
    with pytest.raises(ValueError, match="shape, dtype and device"):
        hvp(tangents, packed)


def test_hybrid_analytic_jet_and_gradcheck():
    class Sine(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    x = torch.tensor(0.2, dtype=torch.float64)
    program = compile_jet(Sine(), (x,), spec=JetSpec.directional(2), backward="analytic")
    hybrid = program.hybrid(backend="aot_eager")
    packed = torch.tensor([0.2, 0.4, 0.03], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(hybrid, (packed,))
    assert torch.autograd.gradgradcheck(hybrid, (packed,))


def test_parameter_ad_preserves_original_capture_after_python_attribute_change():
    class Switch(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
            self.use_cos = False

        def forward(self, x):
            z = self.weight * x
            return torch.cos(z) if self.use_cos else torch.sin(z)

    source = Switch()
    x = torch.tensor(0.2, dtype=torch.float64)
    program = compile_jet(source, (x,), spec=JetSpec.directional(1))
    packed = directional_seed(x, x, program.spec)
    expected = program(packed)
    source.use_cos = True
    hybrid = program.hybrid(backend="aot_eager")
    torch.testing.assert_close(hybrid(packed), expected)
    hvp = program.compile_parameter_hvp(lambda y: y.square().sum(), backend="aot_eager")
    params = dict(program.named_parameters())
    first = torch.autograd.grad(expected.square().sum(), source.weight, create_graph=True)[0]
    wanted = torch.autograd.grad(first, source.weight)[0]
    actual = hvp({name: torch.ones_like(p) for name, p in params.items()}, packed)
    torch.testing.assert_close(next(iter(actual.values())), wanted)


@pytest.mark.slow
@pytest.mark.timeout(480)
@pytest.mark.parametrize("order", [16, 32])
def test_high_order_hybrid_and_parameter_hvp_against_closed_form(order):
    class Exponential(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
            self.bias = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

        def forward(self, x):
            return torch.exp(self.weight*x + self.bias)

    source = Exponential()
    x = torch.tensor([0.2], dtype=torch.float64)
    program = compile_operator(source, (x,), operator=OperatorSpec.mixed((order,)))
    hybrid = program.hybrid(backend="aot_eager")
    expected = source.weight**order * torch.exp(source.weight*x + source.bias)
    torch.testing.assert_close(hybrid(x), expected, rtol=3e-12, atol=0)
    got = second_gradients(hybrid(x).sum(), tuple(source.parameters()))
    want = second_gradients(expected.sum(), tuple(source.parameters()))
    for actual_order, expected_order in zip(got, want, strict=True):
        for a, b in zip(actual_order, expected_order, strict=True):
            torch.testing.assert_close(a, b, rtol=3e-12, atol=0)
    params = dict(program.named_parameters())
    tangents = {name: torch.full_like(p, 0.3 if p is source.weight else -0.2)
                for name, p in params.items()}
    result = program.compile_parameter_hvp(lambda output: output.sum(), backend="aot_eager")(tangents, x)
    value = (source.weight**order * torch.exp(source.weight*x + source.bias)).squeeze()
    log_slope = order/source.weight + x.squeeze()
    for name, parameter in params.items():
        wanted = (value * (0.3*(log_slope**2 - order/source.weight**2) - 0.2*log_slope)
                  if parameter is source.weight else value*(0.3*log_slope - 0.2))
        torch.testing.assert_close(result[name], wanted, rtol=3e-12, atol=0)
