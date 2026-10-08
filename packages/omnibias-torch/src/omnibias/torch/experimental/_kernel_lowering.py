# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded activation regions and packed polynomial-system recurrences.

Packed kernels reduce one coefficient at a time. Their largest product has
at most one layout axis, never an all-pairs-by-feature tensor. All differential
equations, initializers and monomial plans come from the shared core rules.
"""

from __future__ import annotations

from collections.abc import Callable
from math import pi, sqrt
from typing import Any, cast

from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import (
    _ALIASES,
    _BUILTIN_PLANS,
    _DERIVATIVE_PLANS,
    _DERIVATIVE_SYSTEMS,
    _initial,
    _scale,
    _stable_initial_rhs,
    activation,
    activation_derivative,
)
from omnibias.core.jet_systems import SYSTEMS

import torch
from torch import Tensor, nn
from torch.fx import Graph, GraphModule, Proxy
from torch.fx.proxy import GraphAppendingTracer

REGION_THRESHOLD = 2048
KERNEL_LOWERINGS = frozenset({"auto", "inline", "regional", "packed"})


def validate_lowering(mode: str) -> None:
    if mode not in KERNEL_LOWERINGS:
        raise ValueError("kernel_lowering must be auto, inline, regional or packed")


def estimate_inline_nodes(spec: JetSpec, name: str, derivative: bool = False) -> int:
    """Conservative operation estimate without materializing an FX graph."""
    name = _ALIASES.get(name, name)
    count = spec.size
    pairs = sum(map(len, spec.pairs))
    product = 2 * pairs + count + 2
    if name in {"identity", "linear"}:
        return 2 * count + 4
    if name == "reciprocal":
        return 2 * (pairs - count) + 3 * count + 5 + (product if derivative else 0)
    if name == "silu":
        return estimate_inline_nodes(spec, "sigmoid", derivative) + product + count + (
            estimate_inline_nodes(spec, "sigmoid") if derivative else 0)
    if name == "gelu_approx":
        return (5 if derivative else 3) * product + 8 * count + estimate_inline_nodes(spec, "tanh", derivative) + (
            estimate_inline_nodes(spec, "tanh") if derivative else 0)
    system = _DERIVATIVE_SYSTEMS[name] if derivative else SYSTEMS[name]
    nodes, polynomials = (_DERIVATIVE_PLANS if derivative else _BUILTIN_PLANS)[name]
    return (3 * len(system.states) * (pairs - count) + 2 * len(nodes) * pairs
            + count * (2 * sum(map(len, polynomials)) + len(system.states) + 3) + 32)


def _product_at(a: Any, b: Any, spec: JetSpec, position: int, *, positive: bool = False) -> Any:
    if spec.mode == "directional":
        start = int(positive)
        return (a[start:position + 1] * b[:position + 1 - start].flip(0)).sum(0)
    pairs = spec.pairs[position]
    left = [i for i, _ in pairs if i or not positive]
    right = [j for i, j in pairs if i or not positive]
    return (a[left] * b[right]).sum(0)


def packed_product(a: Any, b: Any, spec: JetSpec) -> Any:
    return torch.stack([_product_at(a, b, spec, k) for k in range(spec.size)])


def _append(previous: Any, coefficient: Any, position: int) -> Any:
    row = coefficient.unsqueeze(0)
    return torch.cat((previous, row), dim=0) if position else row


def _identity(a: Any) -> Any:
    return torch.cat((torch.ones_like(a[:1]), torch.zeros_like(a[1:])), dim=0)


def packed_activation(a: Any, spec: JetSpec, name: str, ops: Any, *, derivative: bool = False) -> Any:
    """Evaluate shared differential systems using per-coefficient reductions."""
    name = _ALIASES.get(name, name)
    if name in {"identity", "linear"}:
        return _identity(a) if derivative else a
    if name == "reciprocal":
        result = (torch.ones_like(a[0]) / a[0]).unsqueeze(0)
        for k in range(1, spec.size):
            result = _append(result, -_product_at(a, result, spec, k, positive=True) / a[0], k)
        return -packed_product(result, result, spec) if derivative else result
    if name == "silu":
        sig = packed_activation(a, spec, "sigmoid", ops)
        if derivative:
            return sig + packed_product(a, packed_activation(a, spec, "sigmoid", ops, derivative=True), spec)
        return packed_product(a, sig, spec)
    if name == "gelu_approx":
        square = packed_product(a, a, spec)
        inner = (a + packed_product(square, a, spec) * 0.044715) * sqrt(2 / pi)
        # 0.5*x*(1+tanh(h)) == x*sigmoid(2*h); the latter preserves
        # the small negative-tail value and all its derivative coefficients.
        blend = packed_activation(2 * inner, spec, "sigmoid", ops)
        if derivative:
            inner_prime = (_identity(a) + square * (3 * 0.044715)) * (2 * sqrt(2 / pi))
            correction = packed_product(a, packed_product(
                packed_activation(2 * inner, spec, "sigmoid", ops, derivative=True), inner_prime, spec), spec)
            return blend + correction
        return packed_product(a, blend, spec)
    system = (_DERIVATIVE_SYSTEMS if derivative else SYSTEMS)[name]
    nodes, polynomials = (_DERIVATIVE_PLANS if derivative else _BUILTIN_PLANS)[name]
    initial = [_initial(state, a[0], ops) for state in system.states]
    stable_rhs = _stable_initial_rhs(name, a[0], initial, ops)
    zero = torch.zeros_like(a[0])
    states = [value.unsqueeze(0) for value in initial]
    values = [a, *states, _identity(a), *[zero.unsqueeze(0) for _ in nodes]]
    rhs = [zero.unsqueeze(0) for _ in states]
    weighted = (a.movedim(0, -1) * a.new_tensor(spec.degrees)).movedim(-1, 0)
    first_product = len(states) + 2
    output = []

    def polynomial(poly: Any, k: int) -> Any:
        total = zero
        for coefficient, source in poly:
            total = total + _scale(values[source][k], coefficient)
        return total

    for k, degree in enumerate(spec.degrees):
        if k:
            for index in range(len(states)):
                value = _product_at(weighted, rhs[index], spec, k, positive=True) / degree
                states[index] = _append(states[index], value, k)
                values[index + 1] = states[index]
        for offset, (left, right) in enumerate(nodes):
            product = _product_at(values[left], values[right], spec, k)
            at = first_product + offset
            values[at] = _append(values[at], product, k)
        for index, poly in enumerate(polynomials[:-1]):
            value = stable_rhs[index] if k == 0 and stable_rhs is not None else polynomial(poly, k)
            rhs[index] = _append(rhs[index], value, k)
        output.append(polynomial(polynomials[-1], k))
    if derivative and name in {"tanh", "sigmoid"} and stable_rhs is not None:
        output[0] = stable_rhs[0]
    return torch.stack(output)


def activation_graph(spec: JetSpec, name: str, ops: Any, *, derivative: bool = False,
                     packed: bool = False, graph_factory: Callable[[], Graph] = Graph) -> GraphModule:
    graph = graph_factory()
    tracer = GraphAppendingTracer(graph)
    value: Any = Proxy(graph.placeholder("packed"), tracer)
    if packed:
        result = packed_activation(value, spec, name, ops, derivative=derivative)
    else:
        function = activation_derivative if derivative else activation
        result = torch.stack(function([value[k] for k in range(spec.size)], spec, name, ops))
    graph.output(result.node)
    return GraphModule(nn.Module(), graph, "PackedJetActivation" if packed else "JetActivationRegion")


class _RegionalActivation(nn.Module):
    def __init__(self, graph: GraphModule) -> None:
        super().__init__()
        self.graph = graph

        def invoke(packed: Tensor) -> Tensor:
            return cast(Tensor, self.graph(packed))

        self.region = torch.compiler.nested_compile_region(invoke)

    def forward(self, packed: Tensor) -> Tensor:
        return cast(Tensor, self.region(packed))


class KernelLowering:
    """One compiler invocation's reusable, budgeted activation kernels."""

    def __init__(self, mode: str, ops: Any, tracer: GraphAppendingTracer, holder: nn.Module,
                 graph_factory: Callable[[], Graph] = Graph) -> None:
        validate_lowering(mode)
        self.mode, self.ops, self.tracer, self.holder = mode, ops, tracer, holder
        self.graph_factory = graph_factory
        self.records: list[dict[str, Any]] = []
        self.modules: dict[tuple[JetSpec, str, bool], nn.Module] = {}
        self.names: dict[tuple[JetSpec, str, bool], str] = {}

    def select(self, spec: JetSpec, name: str, derivative: bool = False) -> str:
        estimate = estimate_inline_nodes(spec, name, derivative)
        available = callable(getattr(torch.compiler, "nested_compile_region", None))
        if self.mode == "auto":
            if spec.mode != "directional" or estimate <= REGION_THRESHOLD:
                return "inline"
            return "regional" if available else "packed"
        if self.mode == "regional" and not available:
            raise ValueError("regional kernel lowering requires torch.compiler.nested_compile_region; use packed or auto")
        return self.mode

    def module(self, spec: JetSpec, name: str, derivative: bool = False) -> nn.Module:
        key = spec, name, derivative
        if key not in self.modules:
            selected = self.select(spec, name, derivative)
            graph = activation_graph(spec, name, self.ops, derivative=derivative,
                                     packed=selected == "packed", graph_factory=self.graph_factory)
            self.modules[key] = _RegionalActivation(graph) if selected == "regional" else graph
            self.records.append({"activation": name, "derivative": derivative,
                                 "layout": spec.mode, "coefficients": spec.size,
                                 "selected": selected, "estimated_inline_nodes": estimate_inline_nodes(spec, name, derivative),
                                 "graph_nodes": len(list(graph.graph.nodes))})
        return self.modules[key]

    def activate(self, rows: list[Any], spec: JetSpec, name: str) -> list[Any]:
        if self.select(spec, name) == "inline":
            return activation(rows, spec, name, self.ops)
        key = spec, name, False
        module = self.module(spec, name)
        if key not in self.names:
            self.names[key] = f"jet_kernel_{len(self.holder._modules)}"
            self.holder.add_module(self.names[key], module)
        result: Any = self.tracer.create_proxy("call_module", self.names[key], (torch.stack(rows),), {})
        return [result[k] for k in range(spec.size)]

    def explain(self) -> dict[str, Any]:
        return {"requested": self.mode, "regional_threshold_nodes": REGION_THRESHOLD,
                "regions": tuple(dict(row) for row in self.records),
                "packed_scope": "one coefficient reduction; no all-pairs feature tensor"}
