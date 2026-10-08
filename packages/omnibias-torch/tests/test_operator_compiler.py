# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent operator oracles and structural planner contract checks."""

import math
from collections import OrderedDict, defaultdict, namedtuple
from fractions import Fraction

import mpmath as mp
import pytest
import torch
from omnibias.core.operator_spec import OperatorPlanningBudget
from omnibias.torch.experimental import JetBudget, JetCompileError, OperatorSpec, compile_operator
from torch import nn


class Function(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


def mixed(function, x, alpha):
    coordinate = x.detach().clone().requires_grad_(True)
    value = function(coordinate).sum()
    for axis, order in enumerate(alpha):
        for _ in range(order):
            value = torch.autograd.grad(value, coordinate, create_graph=True)[0][axis]
    return value


@pytest.mark.parametrize("alpha", [(1, 1), (2, 1), (2, 3)])
def test_read_once_repeated_mixed_derivatives_and_scalar_outer_cut(alpha):
    def formula(x):
        s = torch.sin(x[0]) + torch.exp(x[1])
        return torch.sin(s) * torch.exp(s)
    x = torch.tensor([0.13, -0.2], dtype=torch.float64)
    model = Function(formula)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed(alpha))
    expected = mixed(model, x, alpha)
    torch.testing.assert_close(program(x), expected, rtol=2e-11, atol=2e-12)
    assert "general-taylor-fallback" not in program.explain()["matched_patterns"]
    assert "scalar-cut" in program.explain()["matched_patterns"]


@pytest.mark.parametrize("order", [24, 32])
def test_square_free_separable_operator_avoids_exponential_layout(order):
    class Separable(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.linspace(0.4, 0.6, order, dtype=torch.float64))
            self.offset = nn.Parameter(torch.tensor(0.03, dtype=torch.float64))

        def forward(self, x):
            return torch.sin(torch.tanh(x * self.weight).sum(-1) + self.offset)

    model = Separable()
    x = torch.linspace(0.01, 0.02, order, dtype=torch.float64)
    request = OperatorSpec.mixed((1,) * order, budget=JetBudget(max_coefficients=1000))
    program = compile_operator(model, (x,), operator=request)
    actual = program(x)
    with mp.workdps(80):
        w = [mp.mpf(float(v)) for v in model.weight.detach()]
        points = [mp.mpf(float(v)) for v in x]
        s = sum(mp.tanh(a*b) for a, b in zip(w, points, strict=True)) + mp.mpf(float(model.offset.detach()))
        expected = mp.diff(mp.sin, s, order) * mp.fprod(a / mp.cosh(a*b)**2 for a, b in zip(w, points, strict=True))
    assert float(actual.detach()) == pytest.approx(float(expected), rel=3e-12, abs=1e-30)
    actual.backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    assert model.offset.grad is not None and torch.isfinite(model.offset.grad)
    assert program.explain()["coefficient_count"] == order + 1
    with pytest.raises(ValueError, match="budget"):
        compile_operator(model, (x,), operator=request, strategy="reference")


def test_disjoint_product_with_zero_primal_has_no_division_singularity():
    model = Function(lambda x: torch.exp(torch.sin(x[0]) * torch.cos(x[1])))
    x = torch.tensor([0.0, 0.2], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed((2, 2)))
    torch.testing.assert_close(program(x), mixed(model, x, (2, 2)), rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("constant", [0, 2])
def test_ridge_polynomial_constant_term_retains_scaled_readout_bias_and_gradient(constant):
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    x = torch.tensor([0.2, -0.1], dtype=torch.float64)
    terms = {(2, 0): 1, (0, 0): constant}
    program = compile_operator(model, (x,), operator=OperatorSpec.polynomial(terms))
    expected = constant * model(x).sum() + mixed(model, x, (2, 0))
    torch.testing.assert_close(program(x).squeeze(), expected)
    actual_grad = torch.autograd.grad(program(x).sum(), model[-1].bias, allow_unused=True)[0]
    if constant:
        torch.testing.assert_close(actual_grad, torch.full_like(model[-1].bias, constant))
    else:
        assert actual_grad is None or torch.equal(actual_grad, torch.zeros_like(actual_grad))


def test_batched_ridge_direction_and_parameter_graph_compiles():
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)).double()
    x = torch.randn(2, 3, dtype=torch.float64) * 0.2
    direction = torch.randn_like(x, requires_grad=True)
    program = compile_operator(model, (x,), operator=OperatorSpec.directional(4), batch_ndim=1)
    assert next(program.parameters()) is next(model.parameters())
    t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    expected = model(x + t * direction).sum()
    for _ in range(4):
        expected = torch.autograd.grad(expected, t, create_graph=True)[0]
    actual = program(x, direction=direction).sum()
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=1e-12)
    parameters = (*model.parameters(), direction)
    got = torch.autograd.grad(actual + model(x).sum(), parameters)
    want = torch.autograd.grad(expected + model(x).sum(), parameters)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-10, atol=1e-11)
    compiled = torch.compile(program, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(x, direction=direction), program(x, direction=direction))


@pytest.mark.parametrize("power", [1, 2, 3])
def test_affine_bottleneck_weighted_operator_transport(power):
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(4, 2), nn.Tanh(), nn.Linear(2, 3), nn.Sigmoid(), nn.Linear(3, 1)).double()
    x = torch.linspace(-0.2, 0.2, 4, dtype=torch.float64)
    request = OperatorSpec.laplacian(power, planning_budget=OperatorPlanningBudget(max_graph_nodes=50_000))
    program = compile_operator(model, (x,), operator=request)
    reference = compile_operator(model, (x,), operator=request, strategy="reference")
    torch.testing.assert_close(program(x), reference(x), rtol=5e-10, atol=2e-12)
    assert any("bottleneck" in pattern for pattern in program.explain()["matched_patterns"])
    got = torch.autograd.grad(program(x).square().sum() + model(x).sum(), tuple(model.parameters()))
    want = torch.autograd.grad(reference(x).square().sum() + model(x).sum(), tuple(model.parameters()))
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-9, atol=1e-11)


def test_factorized_ridge_metric_and_dynamic_weights():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(5, 2), nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).double()
    x = torch.linspace(-0.1, 0.2, 5, dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.laplacian(2))
    assert "factorized-ridge-metric" in program.explain()["matched_patterns"]
    value = program(x).detach().clone()
    with torch.no_grad():
        model[0].weight.add_(0.1)
    assert not torch.equal(value, program(x))
    # Independent nested Hessians include the changed, live parameters.
    def scalar(v):
        return model(v).sum()

    def lap(v):
        return torch.func.hessian(scalar)(v).trace()
    expected = torch.func.hessian(lap)(x).trace()
    torch.testing.assert_close(program(x).squeeze(), expected, rtol=2e-11, atol=1e-12)


@pytest.mark.parametrize("strategy", ["auto", "reference"])
@pytest.mark.parametrize("function", [lambda x: x.sum(0), lambda x: x.t(), lambda x: x[0]])
def test_batch_coupling_is_hard_rejected(function, strategy):
    x = torch.ones(2, 3, dtype=torch.float64)
    with pytest.raises(JetCompileError, match="batch"):
        compile_operator(Function(function), (x,), operator=OperatorSpec.mixed((1, 0, 0)), batch_ndim=1, strategy=strategy)


@pytest.mark.parametrize("function", [lambda x: 2 + x, lambda x: 2 * x])
def test_constant_left_arithmetic_preserves_batch(function):
    x = torch.ones(2, 3, dtype=torch.float64)
    program = compile_operator(Function(function), (x,), operator=OperatorSpec.mixed((1, 0, 0)), batch_ndim=1)
    expected = torch.zeros_like(x)
    expected[:, 0] = 1 if function(x)[0, 0] == 3 else 2
    torch.testing.assert_close(program(x), expected)


def test_reference_seed_and_readout_remain_inside_fx_and_preserve_output_tree():
    model = Function(lambda x: {"a": torch.softmax(x, dim=-1), "b": (torch.sin(x),)})
    x = torch.tensor([0.1, -0.2], dtype=torch.float64)
    request = OperatorSpec.mixed((1, 1))
    program = compile_operator(model, (x,), operator=request)
    expected = torch.func.jacfwd(torch.func.jacrev(lambda v: torch.softmax(v, dim=-1)))(x)[:, 0, 1]
    result = program(x)
    torch.testing.assert_close(result["a"], expected)
    torch.testing.assert_close(result["b"][0], torch.zeros_like(x))
    assert "general-taylor-fallback" in program.explain()["matched_patterns"]


@pytest.mark.parametrize("order", [16, 32])
def test_high_order_ridge_uses_stable_recurrence_default(order):
    model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid(), nn.Linear(1, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[0].bias.zero_()
        model[2].weight.fill_(1)
        model[2].bias.zero_()
    x = torch.tensor([4.0], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed((order,)))
    with mp.workdps(80):
        expected = mp.diff(lambda z: 1 / (1 + mp.exp(-z)), 4, order)
    assert float(program(x).detach()) == pytest.approx(float(expected), rel=3e-11)
    assert program.explain()["activation_method"] == "recurrence"


@pytest.mark.parametrize("power,states", [(1, 5), (2, 14)])
def test_coupled_operator_selects_apolar_with_live_parameter_gradients(power, states):
    class Coupled(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

        def forward(self, x):
            return torch.sin(self.scale * (x[0] * x[1] + x[1] * x[2] + x[2] * x[0]))

    model = Coupled()
    x = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.laplacian(power))
    reference = model
    for _ in range(power):
        previous = reference

        def reference(v, previous=previous):
            return torch.func.hessian(previous)(v).trace()

    actual, expected = program(x), reference(x)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    torch.testing.assert_close(torch.autograd.grad(actual, model.scale)[0], torch.autograd.grad(expected, model.scale)[0], rtol=2e-10, atol=2e-12)
    assert program.explain()["selected_strategy"] == "apolar"
    assert program.explain()["coefficient_count"] == states


@pytest.mark.parametrize("point", [-0.2, 1.5])
def test_bottleneck_softplus_source_options_match_nested_derivatives(point):
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(3, 1), nn.Softplus(beta=2, threshold=1), nn.Linear(1, 2), nn.Tanh(), nn.Linear(2, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(1 / 3)
        model[0].bias.zero_()
    x = torch.full((3,), point, dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.laplacian(2))

    def lap(v):
        return torch.func.hessian(lambda u: model(u).sum())(v).trace()

    expected = torch.func.hessian(lap)(x).trace()
    torch.testing.assert_close(program(x).squeeze(), expected, rtol=2e-10, atol=2e-12)
    assert "affine-bottleneck-polynomial-transport" in program.explain()["matched_patterns"]


@pytest.mark.parametrize("strategy", ["auto", "reference"])
@pytest.mark.parametrize("function", [lambda x: torch.cat((x, x), dim=0), lambda x: torch.stack((x, x), dim=0), lambda x: x.unbind(0)[0]])
def test_batch_sequence_operations_are_hard_rejected(function, strategy):
    x = torch.ones(2, 3, dtype=torch.float64)
    with pytest.raises(JetCompileError, match="batch"):
        compile_operator(Function(function), (x,), operator=OperatorSpec.mixed((1, 0, 0)), batch_ndim=1, strategy=strategy)


def test_dynamic_batch_ridge_guards_and_output_shape():
    class Ridge(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)).double()

        def forward(self, x):
            return self.layers(x).squeeze(-1)

    model = Ridge()
    x = torch.zeros(2, 3, dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed((1, 1, 0)), batch_ndim=1,
                               dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},))
    larger = torch.linspace(-0.2, 0.2, 15, dtype=torch.float64).reshape(5, 3)
    expected = torch.stack([mixed(model, row, (1, 1, 0)) for row in larger])
    torch.testing.assert_close(program(larger), expected)
    assert program(larger).shape == (5,)
    with pytest.raises(AssertionError, match="bound|range|dimension"):
        program(torch.zeros(9, 3, dtype=torch.float64))


def test_polynomial_provider_opt_in_is_explicit_and_semantics_match_low_order():
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    x = torch.tensor([0.1, -0.2], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed((2, 1)), activation_method="polynomial")
    torch.testing.assert_close(program(x).squeeze(), mixed(model, x, (2, 1)))
    assert program.explain()["polynomial_accuracy_note"]


@pytest.mark.parametrize("strategy", ["auto", "reference"])
def test_tracked_broadcast_cannot_shift_example_axis_into_coordinates(strategy):
    # B == D makes this legal broadcasting while nevertheless coupling rows.
    x = torch.ones(3, 3, dtype=torch.float64)
    with pytest.raises(JetCompileError, match="batch"):
        compile_operator(Function(lambda v: v + v.sum(-1)), (x,), operator=OperatorSpec.mixed((1, 0, 0)), batch_ndim=1, strategy=strategy)


@pytest.mark.parametrize("strategy", ["auto", "reference"])
def test_graph_emission_stops_at_resource_limit(strategy):
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    request = OperatorSpec.mixed((8, 0), planning_budget=OperatorPlanningBudget(max_graph_nodes=50))
    with pytest.raises(JetCompileError, match="node budget.*during emission"):
        compile_operator(model, (torch.ones(2, dtype=torch.float64),), operator=request, strategy=strategy)


def test_polynomial_large_exact_weight_crosses_backend_boundary_as_float():
    x = torch.tensor([0.1], dtype=torch.float64)
    model = nn.Sequential(nn.Linear(1, 1), nn.Tanh()).double()
    coefficient = 2**70
    request = OperatorSpec.polynomial({(2,): coefficient})
    program = compile_operator(model, (x,), operator=request)
    torch.testing.assert_close(program(x).squeeze(), mixed(model, x, (2,)) * float(coefficient))


def test_zero_order_ridge_term_preserves_captured_gelu_primal():
    model = nn.Sequential(nn.Linear(1, 1), nn.GELU(), nn.Linear(1, 1)).double()
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[0].bias.zero_()
        model[2].weight.fill_(1)
        model[2].bias.zero_()
    x = torch.tensor([-10.0], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.polynomial({(0,): 2}))
    torch.testing.assert_close(program(x), 2 * model(x), rtol=0, atol=0)


@pytest.mark.parametrize("coefficient", [2**1100, Fraction(1, 2**1100)])
def test_unrepresentable_static_operator_weights_refuse(coefficient):
    request = OperatorSpec.polynomial({(1,): coefficient})
    with pytest.raises(JetCompileError, match="not representable.*binary64"):
        compile_operator(Function(torch.sin), (torch.tensor([0.1], dtype=torch.float64),), operator=request)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("alpha", [(1, 0, 1, 0), (1, 1, 1, 1)])
def test_tensorized_separable_subset_duplicate_branches_and_shared_parameters(reduction, alpha):
    class Separable(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.6, dtype=torch.float64))
            self.bias = nn.Parameter(torch.tensor(-0.1, dtype=torch.float64))

        def forward(self, x):
            u = x * self.weight + self.bias
            branches = torch.tanh(u) + torch.sin(u) * torch.cos(x)
            cut = branches.sum(-1, keepdim=True) if reduction == "sum" else branches.mean(-1, keepdim=True)
            return torch.sin(cut * self.weight) * torch.exp(cut) + self.bias

    model = Separable()
    x = torch.tensor([[0.1, -0.2, 0.3, 0.15], [-0.1, 0.05, 0.2, -0.3]], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed(alpha), batch_ndim=1)
    expected = torch.stack([mixed(model, row, alpha) for row in x]).unsqueeze(-1)
    actual = program(x)
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-12)
    got = torch.autograd.grad(actual.sum(), tuple(model.parameters()))
    want = torch.autograd.grad(expected.sum(), tuple(model.parameters()))
    for left, right in zip(got, want, strict=True):
        torch.testing.assert_close(left, right, rtol=5e-10, atol=3e-12)
    assert "tensorized-separable-scalar-cut" in program.explain()["matched_patterns"]


def test_separable_vector_fastpath_refuses_spatial_bypass():
    model = Function(lambda x: torch.sin(torch.tanh(x).sum()) * torch.cos(x[0]))
    x = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed((1, 1, 1)))
    torch.testing.assert_close(program(x), mixed(model, x, (1, 1, 1)), rtol=3e-11, atol=3e-12)
    assert "tensorized-separable-scalar-cut" not in program.explain()["matched_patterns"]


_OutputPair = namedtuple("OperatorOutputPair", ["first", "second"])


@pytest.mark.parametrize("kind", ["ordered", "default", "named"])
def test_reference_preserves_keyed_output_container_types(kind):
    def formula(x):
        first, second = torch.softmax(x, dim=-1), torch.sin(x)
        if kind == "ordered":
            return OrderedDict([("first", first), ("second", second)])
        if kind == "default":
            return defaultdict(float, {"first": first, "second": second})
        return _OutputPair(first, second)

    model = Function(formula)
    x = torch.tensor([0.1, -0.2], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed((1, 1)), strategy="reference")
    result = program(x)
    expected = torch.func.jacfwd(torch.func.jacrev(lambda v: torch.softmax(v, dim=-1)))(x)[:, 0, 1]
    assert type(result) is type(model(x))
    first, second = (result.first, result.second) if kind == "named" else (result["first"], result["second"])
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, torch.zeros_like(x))
    if kind == "default":
        assert result.default_factory is float


def test_flat_guard_fastpath_retains_argument_dtype_rank_and_dimension_checks():
    x = torch.ones(2, 3, dtype=torch.float64)
    program = compile_operator(Function(torch.sin), (x,), operator=OperatorSpec.mixed((1, 0, 0)), batch_ndim=1)
    with pytest.raises(JetCompileError, match="tree"):
        program({"value": x})
    with pytest.raises(JetCompileError, match="tree"):
        program(x, x)
    with pytest.raises(JetCompileError, match="dtype"):
        program(x.float())
    with pytest.raises(AssertionError, match="rank"):
        program(x.unsqueeze(0))
    with pytest.raises(AssertionError, match="dimension"):
        program(torch.ones(2, 4, dtype=torch.float64))


def test_flat_guard_fastpath_retains_shared_dynamic_batch_constraints():
    class Scaled(nn.Module):
        def forward(self, x, scale):
            return torch.sin(x) * scale

    x = torch.ones(2, 3, dtype=torch.float64)
    scale = torch.ones(2, 1, dtype=torch.float64)
    batch = torch.export.Dim("shared_batch", min=1, max=6)
    program = compile_operator(Scaled(), (x, scale), operator=OperatorSpec.mixed((1, 0, 0)), batch_ndim=1,
                               dynamic_shapes=({0: batch}, {0: batch}))
    larger = torch.ones(3, 3, dtype=torch.float64)
    result = program(larger, torch.ones(3, 1, dtype=torch.float64))
    expected = torch.zeros_like(larger)
    expected[:, 0] = math.cos(1)
    torch.testing.assert_close(result, expected)
    with pytest.raises(AssertionError, match="shared dynamic"):
        program(larger, scale)
    with pytest.raises(AssertionError, match="above.*bound"):
        program(torch.ones(7, 3, dtype=torch.float64), torch.ones(7, 1, dtype=torch.float64))


def test_ridge_terminal_shape_replay_drops_unused_primal_readout_bias():
    class ShapeWrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()

        def forward(self, x):
            return self.layers(x).squeeze(-1).unsqueeze(-1).clone()

    model = ShapeWrapped()
    x = torch.tensor([[0.1, -0.2], [0.3, 0.1]], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed((1, 1)), batch_ndim=1)
    expected = torch.stack([mixed(model, row, (1, 1)) for row in x]).unsqueeze(-1)
    torch.testing.assert_close(program(x), expected)
    assert not any(node.op == "get_attr" and str(node.target).endswith("layers.2.bias") for node in program.graph_module.graph.nodes)
    assert tuple(program.parameters())[-1] is model.layers[-1].bias


def test_operator_call_preserves_original_parameter_and_module_backward_hooks():
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    x = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed((1, 1)))
    weight = model[0].weight
    expected = torch.autograd.grad(program(x).sum(), weight)[0]
    backward_events = []
    parameter_hook = weight.register_hook(lambda gradient: gradient * 2)
    module_hook = program.register_full_backward_hook(lambda module, inputs, outputs: backward_events.append((inputs, outputs)))
    try:
        program(x).sum().backward()
    finally:
        parameter_hook.remove()
        module_hook.remove()
    torch.testing.assert_close(weight.grad, 2 * expected)
    assert len(backward_events) == 1
    assert backward_events[0][0][0].shape == x.shape


def test_inplace_compile_preserves_forward_hooks_and_live_parameter_gradients():
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    x = torch.tensor([0.1, -0.2], dtype=torch.float64)
    program = compile_operator(model, (x,), operator=OperatorSpec.mixed((1, 1)))
    expected = program(x)
    hook = program.register_forward_hook(lambda module, inputs, output: output + 1)
    program.compile(backend="aot_eager", fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(program(x), expected.detach() + 1)
    hook.remove()
    result = program(x)
    torch.testing.assert_close(result, expected)
    got = torch.autograd.grad(result.sum(), model[0].weight)[0]
    want = torch.autograd.grad(expected.sum(), model[0].weight)[0]
    torch.testing.assert_close(got, want)
