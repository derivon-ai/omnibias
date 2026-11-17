# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent derivative oracles for the experimental export compiler."""

import math

import pytest
import torch
from omnibias.torch.experimental import (
    JetBudget,
    JetCompileError,
    JetSpec,
    coefficients_to_derivatives,
    compile_jet,
    directional_seed,
    extract_derivative,
    register_jet_rule,
    seed_jet,
)
from torch import nn


class Function(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


def derivatives(function, order):
    t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    out = function(t)
    rows = [out]
    for _ in range(order):
        out = torch.autograd.grad(out, t, create_graph=True)[0]
        rows.append(out)
    return torch.stack(rows)


@pytest.mark.parametrize("order", [0, 2, 8, 16, 32])
def test_exponential_all_requested_orders(order):
    x = torch.tensor([0.3], dtype=torch.float64)
    spec = JetSpec.directional(order)
    program = compile_jet(Function(torch.exp), (x,), spec=spec)
    got = coefficients_to_derivatives(program(directional_seed(x, torch.ones_like(x), spec)), spec)
    torch.testing.assert_close(got, torch.exp(x).expand(order + 1, 1), rtol=2e-12, atol=1e-14)
    raw = program(directional_seed(x, torch.ones_like(x), spec))
    torch.testing.assert_close(extract_derivative(raw, spec, order), torch.exp(x), rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_nonlinear_path_parameter_and_direction_gradients(backward):
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    x = torch.tensor([0.3, -0.2], dtype=torch.float64)
    v = torch.tensor([0.4, 0.7], dtype=torch.float64, requires_grad=True)
    acceleration = torch.tensor([0.2, -0.1], dtype=torch.float64)
    spec = JetSpec.directional(3)
    program = compile_jet(model, (x,), spec=spec, backward=backward)
    assert list(program.parameters())[0] is list(model.parameters())[0]
    jet = seed_jet(x, spec, {(1,): v, (2,): acceleration})
    got = coefficients_to_derivatives(program(jet), spec)[:, 0]
    expected = derivatives(lambda t: model(x + t * v + t * t * acceleration)[0], 3)
    torch.testing.assert_close(got, expected, rtol=2e-11, atol=2e-12)
    params = tuple(model.parameters()) + (v,)
    actual_grad = torch.autograd.grad((got[3] + got[0]).square(), params, create_graph=True)
    expected_grad = torch.autograd.grad((expected[3] + expected[0]).square(), params, create_graph=True)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-11)
    actual_second = torch.autograd.grad(actual_grad[0].square().sum(), params, allow_unused=True)
    expected_second = torch.autograd.grad(expected_grad[0].square().sum(), params, allow_unused=True)
    for a, b in zip(actual_second, expected_second, strict=True):
        if a is not None and b is not None:
            torch.testing.assert_close(a, b, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 4, 3)])
def test_constant_weight_linear_batches_coefficients_and_preserves_gradients(shape):
    model = nn.Linear(3, 2).double()
    x = torch.linspace(-0.3, 0.5, math.prod(shape), dtype=torch.float64).reshape(shape)
    spec = JetSpec.directional(3)
    program = compile_jet(model, (x,), spec=spec)
    packed = torch.linspace(-0.2, 0.4, spec.size * x.numel(), dtype=x.dtype).reshape(spec.size, *shape).requires_grad_()
    actual = program(packed)
    expected = torch.stack([torch.nn.functional.linear(row, model.weight, model.bias if k == 0 else None)
                            for k, row in enumerate(packed)])
    torch.testing.assert_close(actual, expected)
    parameters = (packed, model.weight, model.bias)
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
    expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
    for got, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, reference)
    linears = [node for node in program.graph_module.graph.nodes
               if node.target in (torch.nn.functional.linear, torch.ops.aten.linear.default)]
    assert len(linears) == 1


def test_batched_linear_accepts_spatial_bias_coefficients():
    class LinearWithBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[0.2, -0.4], [0.5, 0.3]], dtype=torch.float64))

        def forward(self, value, bias):
            return torch.nn.functional.linear(value, self.weight, bias)

    model = LinearWithBias()
    spec = JetSpec.directional(2)
    x, bias = torch.ones(3, 2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    program = compile_jet(model, (x, bias), spec=spec, argnums=(0, 1))
    input_jet = directional_seed(x, x * 0.2, spec)
    bias_jet = seed_jet(bias, spec, {(1,): bias * 0.3, (2,): bias * -0.1})
    expected = torch.stack([torch.nn.functional.linear(row, model.weight, bias_row)
                            for row, bias_row in zip(input_jet, bias_jet, strict=True)])
    torch.testing.assert_close(program(input_jet, bias_jet), expected)


def test_mixed_partial_polynomial_and_output_tree():
    spec = JetSpec.mixed([(2, 1), (0, 3)])
    model = Function(lambda x: {"poly": x[0] ** 3 * x[1] ** 2, "other": (x.sum(),)})
    x = torch.tensor([0.3, -0.4], dtype=torch.float64)
    program = compile_jet(model, (x,), spec=spec)
    output = program(seed_jet(x, spec, {(1, 0): torch.tensor([1., 0.], dtype=x.dtype), (0, 1): torch.tensor([0., 1.], dtype=x.dtype)}))
    torch.testing.assert_close(extract_derivative(output["poly"], spec, (2, 1)), 12 * x[0] * x[1])
    assert extract_derivative(output["other"][0], spec, (0, 3)) == 0


def test_dynamic_scalar_outputs_keep_their_original_type():
    spec = JetSpec.directional(2)
    x = torch.ones((2, 3), dtype=torch.float64)
    model = Function(lambda x: {"value": torch.exp(x), "batch": x.shape[0], "tag": "jet"})
    program = compile_jet(model, (x,), spec=spec,
                          dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},))
    compiled = torch.compile(program, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch in (2, 4):
        value = x[:1].expand(batch, 3)
        result = compiled(directional_seed(value, torch.ones_like(value), spec))
        assert result["batch"] == batch
        assert isinstance(result["batch"], int)
        assert result["tag"] == "jet"
        torch.testing.assert_close(result["value"][2], torch.exp(value) / 2)


def test_untracked_and_identity_outputs_obey_byte_budget():
    spec = JetSpec.directional(2, budget=JetBudget(max_tensor_bytes=64))

    class ConstantOutput(nn.Module):
        def forward(self, x, constant):
            return {"input": x, "constant": constant}

    with pytest.raises(ValueError, match="budget"):
        compile_jet(ConstantOutput(), (torch.ones(1), torch.ones(100)), spec=spec)
    with pytest.raises(ValueError, match="budget"):
        compile_jet(Function(lambda x: x), (torch.ones(2, 1),), spec=spec,
                    dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},))


@pytest.mark.parametrize("function", [
    torch.sin, torch.cos, torch.sigmoid, torch.erf, torch.erfc,
    lambda x: torch.nn.functional.gelu(x, approximate="none"),
    lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
    torch.nn.functional.silu, torch.nn.functional.softplus,
    lambda x: torch.sqrt(x + 2), lambda x: torch.log(x + 2),
    lambda x: 1 / (x + 2),
])
@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_smooth_rules_against_nested_ad(function, backward):
    x = torch.tensor(0.23, dtype=torch.float64)
    spec = JetSpec.directional(4)
    program = compile_jet(Function(function), (x,), spec=spec, backward=backward)
    got = coefficients_to_derivatives(program(directional_seed(x, torch.ones_like(x), spec)), spec)
    expected = derivatives(lambda t: function(x + t), 4)
    torch.testing.assert_close(got, expected, rtol=2e-10, atol=2e-11)


def test_softmax_large_logits_and_parameter_gradient():
    model = Function(lambda x: torch.softmax(x * x, dim=-1))
    x = torch.tensor([30., 30.1, 29.9], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([0.3, -0.2, 0.1], dtype=x.dtype)
    spec = JetSpec.directional(3)
    program = compile_jet(model, (x,), spec=spec)
    got = coefficients_to_derivatives(program(directional_seed(x, v, spec)), spec)
    expected = torch.stack([derivatives(lambda t, i=i: model(x + t * v)[i], 3) for i in range(3)], dim=-1)
    torch.testing.assert_close(got, expected, rtol=2e-9, atol=1e-10)
    assert torch.isfinite(torch.autograd.grad(got.square().sum(), x)[0]).all()


def test_matmul_both_operands_and_structural_operations():
    def function(x):
        y = x.reshape(2, 2)
        z = torch.cat([y @ y.T, y], dim=0)
        return torch.stack([z.sum(), z.mean()])
    x = torch.tensor([0.2, 0.3, -0.1, 0.7], dtype=torch.float64)
    v = torch.tensor([0.4, -0.1, 0.5, 0.2], dtype=x.dtype)
    spec = JetSpec.directional(2)
    program = compile_jet(Function(function), (x,), spec=spec)
    got = coefficients_to_derivatives(program(directional_seed(x, v, spec)), spec)
    expected = torch.stack([derivatives(lambda t, i=i: function(x + t * v)[i], 2) for i in range(2)], dim=-1)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("dimensions", [1, 2])
def test_convolution_gradients(dimensions):
    cls = nn.Conv1d if dimensions == 1 else nn.Conv2d
    model = nn.Sequential(cls(1, 2, 2), nn.Tanh(), nn.Flatten()).double()
    shape = (1, 1) + (3,) * dimensions
    x = torch.linspace(-0.3, 0.4, math.prod(shape), dtype=torch.float64).reshape(shape)
    v = torch.ones_like(x) * 0.2
    spec = JetSpec.directional(2)
    program = compile_jet(model, (x,), spec=spec)
    got = coefficients_to_derivatives(program(directional_seed(x, v, spec)), spec).sum(dim=(1, 2))
    expected = derivatives(lambda t: model(x + t * v).sum(), 2)
    torch.testing.assert_close(got, expected, rtol=1e-11, atol=1e-12)
    assert torch.isfinite(torch.autograd.grad(got[2], model[0].weight)[0]).all()


@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_dynamic_batches_compile_fullgraph_and_guards(backward):
    x = torch.ones(2, 3, dtype=torch.float64)
    model = nn.Sequential(nn.Linear(3, 2), nn.Tanh()).double()
    spec = JetSpec.directional(2)
    program = compile_jet(model, (x,), spec=spec, backward=backward, dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},))
    compiled = torch.compile(program, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch in (2, 5):
        primal = torch.ones(batch, 3, dtype=x.dtype)
        packed = directional_seed(primal, primal, spec)
        actual, expected = compiled(packed), program(packed)
        torch.testing.assert_close(actual, expected)
        grad_actual = torch.autograd.grad(actual.square().sum(), model[0].weight)[0]
        grad_expected = torch.autograd.grad(expected.square().sum(), model[0].weight)[0]
        torch.testing.assert_close(grad_actual, grad_expected)
    with pytest.raises(AssertionError, match="bound"):
        program(directional_seed(torch.ones(9, 3, dtype=x.dtype), torch.ones(9, 3, dtype=x.dtype), spec))


@pytest.mark.parametrize("function", [torch.relu, lambda x: x + torch.rand_like(x), lambda x: x.detach()])
def test_unsupported_operations_are_source_located(function):
    with pytest.raises(JetCompileError, match="(unsupported|detach)") as error:
        compile_jet(Function(function), (torch.ones(2),), spec=JetSpec.directional(2))
    assert "test_jet_compiler.py" in str(error.value)


def test_rejects_half_and_tensor_budget():
    with pytest.raises(JetCompileError, match="float32 or float64"):
        compile_jet(Function(torch.exp), (torch.ones(2, dtype=torch.float16),), spec=JetSpec.directional(2))
    spec = JetSpec.directional(2, budget=JetBudget(max_tensor_bytes=8))
    with pytest.raises(ValueError, match="budget"):
        directional_seed(torch.ones(2), torch.ones(2), spec)


def test_nonzero_padding_only_affects_primal_and_tuple_shapes():
    model = Function(lambda x: torch.nn.functional.pad(x, (2, 1), value=3).unbind())
    x = torch.tensor([0.2, 0.3], dtype=torch.float64)
    spec = JetSpec.directional(2)
    output = compile_jet(model, (x,), spec=spec)(directional_seed(x, torch.ones_like(x), spec))
    torch.testing.assert_close(torch.stack(output), torch.tensor([[3., 0., 0.], [3., 0., 0.], [0.2, 1., 0.], [0.3, 1., 0.], [3., 0., 0.]], dtype=x.dtype))


def test_nonselected_arguments_remain_live_and_scalars_are_guarded():
    class Product(nn.Module):
        def forward(self, x, weight, count):
            return torch.exp(x @ weight) + count
    x = torch.tensor([0.2, 0.3], dtype=torch.float64)
    weight = torch.tensor([0.5, 0.7], dtype=x.dtype, requires_grad=True)
    spec = JetSpec.directional(2)
    program = compile_jet(Product(), (x, weight, 3), spec=spec)
    packed = directional_seed(x, torch.ones_like(x), spec)
    got = program(packed, weight, 3)
    expected = derivatives(lambda t: torch.exp((x + t) @ weight) + 3, 2)
    torch.testing.assert_close(coefficients_to_derivatives(got, spec), expected)
    assert torch.autograd.grad(got.sum(), weight)[0].norm() > 0
    assert not torch.equal(got, program(packed, 2 * weight, 3))
    with pytest.raises(JetCompileError, match="specialized"):
        program(packed, weight, 4)


def test_analytic_mixed_adjoint_and_double_backward():
    spec = JetSpec.mixed([(2, 1), (1, 2)])
    x = torch.tensor([0.2, 0.3], dtype=torch.float64)
    model = Function(lambda x: torch.nn.functional.gelu(torch.tanh(x)).sum())
    native = compile_jet(model, (x,), spec=spec)
    analytic = compile_jet(model, (x,), spec=spec, backward="analytic")
    packed = torch.linspace(-0.2, 0.4, spec.size * 2, dtype=x.dtype).reshape(spec.size, 2).requires_grad_()
    a, b = native(packed), analytic(packed)
    ga = torch.autograd.grad(a.square().sum(), packed, create_graph=True)[0]
    gb = torch.autograd.grad(b.square().sum(), packed, create_graph=True)[0]
    torch.testing.assert_close(ga, gb, rtol=2e-11, atol=2e-12)
    gga = torch.autograd.grad(ga.square().sum(), packed)[0]
    ggb = torch.autograd.grad(gb.square().sum(), packed)[0]
    torch.testing.assert_close(gga, ggb, rtol=2e-10, atol=2e-11)


def test_rejects_intermediate_resource_overrun_and_mutation():
    x = torch.ones(2, dtype=torch.float64)
    spec = JetSpec.directional(2, JetBudget(max_tensor_bytes=64))
    with pytest.raises(JetCompileError, match="budget"):
        compile_jet(Function(lambda x: x.repeat(100)), (x,), spec=spec)
    with pytest.raises(JetCompileError, match="(mutation|unsupported)"):
        compile_jet(Function(lambda x: x.add_(1)), (x,), spec=JetSpec.directional(2))


def test_seed_rejects_inconsistent_dtype_and_shape():
    spec = JetSpec.directional(2)
    with pytest.raises(ValueError, match="dtype"):
        seed_jet(torch.ones(2), spec, {(1,): torch.ones(2, dtype=torch.float64)})
    with pytest.raises(ValueError, match="shape"):
        seed_jet(torch.ones(2), spec, {(1,): torch.ones(3)})


def test_custom_rule_extension():
    from omnibias.core.jet_series import activation
    def cosh_rule(args, kwargs, spec, ops):
        positive = activation(args[0], spec, "exp", ops)
        negative = activation([-a for a in args[0]], spec, "exp", ops)
        return [(a + b) / 2 for a, b in zip(positive, negative, strict=True)]
    register_jet_rule(torch.ops.aten.cosh.default, cosh_rule)
    x = torch.tensor(0.3, dtype=torch.float64)
    spec = JetSpec.directional(4)
    got = compile_jet(Function(torch.cosh), (x,), spec=spec)(directional_seed(x, torch.ones_like(x), spec))
    torch.testing.assert_close(coefficients_to_derivatives(got, spec), derivatives(lambda t: torch.cosh(x + t), 4))


@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_func_jvp_and_vmap(backward):
    x = torch.tensor([0.2, 0.3], dtype=torch.float64)
    spec = JetSpec.directional(3)
    program = compile_jet(Function(torch.tanh), (x,), spec=spec, backward=backward)
    packed = directional_seed(x, torch.ones_like(x), spec)
    tangent = torch.ones_like(packed) * 0.2
    _, actual = torch.func.jvp(program, (packed,), (tangent,))
    native = compile_jet(Function(torch.tanh), (x,), spec=spec)
    _, expected = torch.func.jvp(native, (packed,), (tangent,))
    torch.testing.assert_close(actual, expected)
    batch = torch.stack([packed, packed + 0.1])
    torch.testing.assert_close(torch.vmap(program)(batch), torch.stack([program(a) for a in batch]))
