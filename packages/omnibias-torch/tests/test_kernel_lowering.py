# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent coefficient oracles and compiler contracts for kernel lowering."""

import math

import mpmath as mp
import pytest
import torch
from omnibias.core.operator_spec import OperatorPlanningBudget
from omnibias.torch.experimental import (
    JetCompileError,
    JetSpec,
    OperatorSpec,
    compile_jet,
    compile_operator,
    directional_seed,
)
from omnibias.torch.experimental._kernel_lowering import (
    REGION_THRESHOLD,
    estimate_inline_nodes,
    packed_activation,
)
from omnibias.torch.experimental.compiler import _OPS
from torch import nn


class Function(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, value):
        return self.function(value)


def oracle(name, z):
    if name == "sigmoid":
        return 1 / (1 + mp.exp(-z))
    if name == "reciprocal":
        return 1 / z
    if name == "softplus":
        return mp.log(1 + mp.exp(z))
    if name == "silu":
        return z / (1 + mp.exp(-z))
    if name == "gelu":
        return z * mp.erfc(-z / mp.sqrt(2)) / 2
    if name == "gelu_approx":
        return z * (1 + mp.tanh(mp.mpf(float(math.sqrt(2 / math.pi))) * (z + mp.mpf(.044715) * z**3))) / 2
    if name == "gaussian":
        return mp.exp(-z*z / 2)
    if name == "gaussian_normalized":
        return mp.exp(-z*z / 2) / mp.sqrt(2 * mp.pi)
    if name == "gaussian_rbf":
        return mp.exp(-z*z)
    return getattr(mp, name)(z)


@pytest.mark.parametrize("name", ["tanh", "sigmoid", "exp", "sin", "cos", "log", "sqrt",
                                  "reciprocal", "erf", "erfc", "gelu", "gelu_approx", "softplus",
                                  "silu", "gaussian", "gaussian_normalized", "gaussian_rbf"])
@pytest.mark.parametrize("derivative", [False, True])
def test_packed_directional_systems_against_independent_nonlinear_path(name, derivative):
    spec = JetSpec.directional(5)
    coefficients = [.35, .21, -.04, .01, -.002, .0003]
    a = torch.tensor(coefficients, dtype=torch.float64)
    actual = packed_activation(a, spec, name, _OPS, derivative=derivative)
    with mp.workdps(80):
        def path(t):
            return sum(mp.mpf(c) * t**k for k, c in enumerate(coefficients))
        function = (lambda z: mp.diff(lambda x: oracle(name, x), z)) if derivative else lambda z: oracle(name, z)
        expected = torch.tensor([float(v) for v in mp.taylor(lambda t: function(path(t)), 0, 5)], dtype=a.dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-15)


@pytest.mark.parametrize("name", ["exp", "tanh", "log", "gelu"])
def test_packed_mixed_uses_multiindex_degrees_and_ordered_pairs(name):
    spec = JetSpec.mixed([(2, 2)])
    a = torch.tensor([.4] + [(-1)**k * .08 / (k+1) for k in range(1, spec.size)], dtype=torch.float64)
    actual = packed_activation(a, spec, name, _OPS)
    with mp.workdps(70):
        def function(x, y):
            value = sum(mp.mpf(float(c)) * x**alpha[0] * y**alpha[1] for c, alpha in zip(a, spec.indices, strict=True))
            return oracle(name, value)
        expected = torch.tensor([float(mp.diff(function, (0, 0), alpha) / fact)
                                 for alpha, fact in zip(spec.indices, spec.factorials, strict=True)], dtype=a.dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-15)


@pytest.mark.parametrize("name,center", [("tanh", 20.), ("sigmoid", -40.), ("gelu", -10.), ("gelu_approx", -10.)])
def test_packed_order32_preserves_stable_tail_initialization(name, center):
    spec = JetSpec.directional(32)
    coefficients = [center, 1., .07] + [0.] * 30
    a = torch.tensor(coefficients, dtype=torch.float64)
    actual = packed_activation(a, spec, name, _OPS)
    with mp.workdps(110):
        expected = torch.tensor([float(v) for v in mp.taylor(
            lambda t: oracle(name, mp.mpf(center) + t + mp.mpf(.07)*t*t), 0, 32)], dtype=a.dtype)
    torch.testing.assert_close(actual, expected, rtol=3e-9, atol=1e-290)


@pytest.mark.parametrize("mode", ["packed", "regional"])
@pytest.mark.parametrize("backward", ["native", "analytic"])
def test_explicit_kernels_preserve_native_higher_gradients_and_compile(mode, backward):
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    x = torch.tensor([[.1, -.2], [.3, .2]], dtype=torch.float64)
    spec = JetSpec.directional(4)
    packed = directional_seed(x, torch.ones_like(x), spec).requires_grad_()
    program = compile_jet(model, (x,), spec=spec, kernel_lowering=mode, backward=backward)
    native = compile_jet(model, (x,), spec=spec, kernel_lowering="inline")
    left, right = program(packed), native(packed)
    torch.testing.assert_close(left, right, rtol=2e-11, atol=2e-13)
    inputs = (packed, model[0].weight)
    ga = torch.autograd.grad(left.square().sum(), inputs, create_graph=True)
    gb = torch.autograd.grad(right.square().sum(), inputs, create_graph=True)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-11, atol=3e-13)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga), inputs)
    hb = torch.autograd.grad(sum(g.square().sum() for g in gb), inputs)
    for a, b in zip(ha, hb, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-10, atol=3e-12)
    compiled = torch.compile(program, backend="aot_eager", fullgraph=True)
    result = compiled(packed)
    torch.testing.assert_close(result, right)
    assert torch.isfinite(torch.autograd.grad(result.square().sum(), model[0].weight)[0]).all()


def test_auto_regions_reuse_identical_activations_and_count_internal_graphs():
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    x = torch.tensor([[.1, -.2]], dtype=torch.float64)
    spec = JetSpec.directional(32)
    assert estimate_inline_nodes(spec, "tanh") > REGION_THRESHOLD
    assert estimate_inline_nodes(JetSpec.directional(16), "tanh") < REGION_THRESHOLD
    program = compile_jet(model, (x,), spec=spec)
    report = program.explain()
    assert len(report["kernel_lowering"]["regions"]) == 1
    region = report["kernel_lowering"]["regions"][0]
    assert region["selected"] == "regional"
    assert report["graph_nodes"] > region["graph_nodes"] > REGION_THRESHOLD
    calls = [node for node in program.graph_module.graph.nodes if node.op == "call_module"]
    assert len(calls) == 2 and calls[0].target == calls[1].target
    canonical = program._canonical_factory()
    assert canonical.explain()["kernel_lowering"]["requested"] == "inline"
    assert tuple(program.parameters())[0] is tuple(canonical.parameters())[0]
    packed = directional_seed(x, torch.ones_like(x), spec)
    torch.testing.assert_close(program(packed), canonical(packed), rtol=2e-12, atol=2e-14)


def test_auto_uses_packed_if_public_regional_api_is_missing(monkeypatch):
    monkeypatch.setattr(torch.compiler, "nested_compile_region", None)
    spec = JetSpec.directional(32)
    x = torch.tensor([.3], dtype=torch.float64)
    program = compile_jet(Function(torch.tanh), (x,), spec=spec)
    assert program.explain()["kernel_lowering"]["regions"][0]["selected"] == "packed"
    with pytest.raises(JetCompileError, match="nested_compile_region"):
        compile_jet(Function(torch.tanh), (x,), spec=spec, kernel_lowering="regional")


@pytest.mark.parametrize("mode", ["regional", "packed"])
def test_operator_budget_applies_before_activation_region_is_materialized(mode):
    request = OperatorSpec.directional(32, planning_budget=OperatorPlanningBudget(max_graph_nodes=60))
    with pytest.raises(JetCompileError, match="budget"):
        compile_operator(Function(torch.tanh), (torch.tensor([.3], dtype=torch.float64),),
                         operator=request, kernel_lowering=mode, strategy="reference")


def test_canonical_factory_preserves_capture_when_source_python_flags_change():
    class MutableSource(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_cos = False
            self.weight = nn.Parameter(torch.tensor(.8, dtype=torch.float64))

        def forward(self, x):
            return self.weight * (torch.cos(x) if self.use_cos else torch.sin(x))

    source = MutableSource()
    x = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
    program = compile_operator(source, (x,), operator=OperatorSpec.directional(3), kernel_lowering="regional")
    source.use_cos = True
    canonical = program._canonical_factory()
    torch.testing.assert_close(program(x, direction=x), canonical(x, direction=x))
    with torch.no_grad():
        source.weight.add_(.2)
    torch.testing.assert_close(canonical(x, direction=x), -source.weight * torch.cos(x) * x**3)


@pytest.mark.parametrize("state_kind", ["parameter", "buffer"])
@pytest.mark.parametrize("frontend", ["jet", "operator"])
def test_capture_refuses_tensor_identity_alias_between_input_and_model_state(state_kind, frontend):
    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            value = torch.tensor([.4], dtype=torch.float64)
            if state_kind == "parameter":
                self.weight = nn.Parameter(value)
            else:
                self.register_buffer("weight", value)

        def forward(self, x):
            return self.weight * x

    model = Stateful()
    with pytest.raises(JetCompileError, match="capture example tensor aliases model state"):
        if frontend == "jet":
            compile_jet(model, (model.weight,), spec=JetSpec.directional(0))
        else:
            compile_operator(model, (model.weight,), operator=OperatorSpec.directional(0))
    # Distinct views retain separate export roles even if storage is shared.
    program = compile_jet(model, (model.weight.detach().view(-1),), spec=JetSpec.directional(0))
    torch.testing.assert_close(program(torch.tensor([[.7]], dtype=torch.float64)),
                               torch.tensor([[.28]], dtype=torch.float64))


def test_float32_packed_kernel_preserves_dtype_and_raw_derivatives():
    spec = JetSpec.directional(12)
    x = torch.tensor([.2, -.1], dtype=torch.float32)
    program = compile_jet(Function(torch.exp), (x,), spec=spec, kernel_lowering="packed")
    normalized = program(directional_seed(x, torch.ones_like(x), spec))
    facts = torch.tensor([float(math.factorial(k)) for k in range(spec.size)], dtype=x.dtype)
    raw = normalized * facts[:, None]
    assert raw.dtype == x.dtype
    torch.testing.assert_close(raw, x.exp().expand_as(raw), rtol=3e-6, atol=0)


def test_capture_refuses_duplicate_examples_but_preserves_runtime_aliases():
    class Product(nn.Module):
        def forward(self, left, right):
            return left * right

    x = torch.tensor([.3], dtype=torch.float64)
    with pytest.raises(JetCompileError, match="distinct example tensors"):
        compile_jet(Product(), (x, x), spec=JetSpec.directional(1), argnums=(0,))
    with pytest.raises(JetCompileError, match="distinct example tensors"):
        compile_operator(Product(), (x, x), operator=OperatorSpec.directional(1))
    program = compile_jet(Product(), (x, x.clone()), spec=JetSpec.directional(1), argnums=(0, 1))
    packed = torch.tensor([[.4], [1.]], dtype=x.dtype)
    torch.testing.assert_close(program(packed, packed), torch.tensor([[.16], [.8]], dtype=x.dtype))
