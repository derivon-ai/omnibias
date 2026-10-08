# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tensorized square-free derivatives through a proven separable scalar cut."""

from __future__ import annotations

from math import factorial
from typing import Any

from omnibias.core.jet_algebra import JetSpec
from omnibias.core.operator_spec import OperatorSpec

import torch
from torch import Tensor, nn
from torch.fx import Node
from torch.fx.proxy import GraphAppendingTracer

from ._operator_ir import NotStructured
from ._operator_lowering import Lowering, activation_name
from .compiler import _SHAPE, _map, _rule, _Series


def _op(node: Node) -> str:
    pieces = str(node.target).split(".")
    return pieces[1] if len(pieces) > 1 and pieces[0] == "aten" else str(node.target)


def _region(node: Node, stop: Node, dependencies: dict[Node, bool]) -> set[Node]:
    """Collect tracked ancestors iteratively, stopping at a proposed cut."""
    pending, result = [node], set()
    while pending:
        current = pending.pop()
        if current in result or not dependencies.get(current, False):
            continue
        result.add(current)
        if current is not stop:
            pending.extend(current.all_input_nodes)
    return result


def separable_derivative(output: Node, selected: Node, nodes: list[Node], env: dict[Node, Any],
                         dependencies: dict[Node, bool], request: OperatorSpec, lowering: Lowering,
                         batch_ndim: int, tracer: GraphAppendingTracer, holder: nn.Module) -> Any:
    """Lower D_alpha F(sum u_j(x_j)) for square-free alpha without scalarization.

    Every tracked tensor before the sum must retain the input shape and use
    pointwise operations. The sum must dominate every spatial path to the
    output. These are graph properties; parameter values establish no support.
    """
    if request.mode != "mixed" or request.alpha is None or any(n not in {0, 1} for n in request.alpha):
        raise NotStructured("tensorized separable lowering requires a square-free mixed query")
    order = sum(request.alpha)
    if not order:
        return env[output]
    source = selected.meta["val"]
    source_shape = tuple(map(str, source.shape))
    ancestors = _region(output, selected, dependencies)
    pointwise = {"add", "sub", "mul", "div", "true_divide", "pow", "square", "neg",
                 "clone", "alias", "contiguous", "view", "reshape", "_unsafe_view", "_to_copy",
                 "zeros_like", "ones_like", "full_like"}
    candidate: tuple[Node, Node, set[Node], set[Node]] | None = None
    for cut in reversed(nodes):
        if cut not in ancestors or _op(cut) not in {"sum", "mean"} or not isinstance(cut.args[0], Node):
            continue
        leaf = cut.args[0]
        leaf_value = leaf.meta.get("val")
        if not isinstance(leaf_value, Tensor) or tuple(map(str, leaf_value.shape)) != source_shape:
            continue
        arguments: Any = cut.args
        dimensions: Any = arguments[1] if len(arguments) > 1 else cut.kwargs.get("dim")
        dimensions = tuple(range(source.ndim)) if dimensions is None else ((dimensions,) if isinstance(dimensions, int) else tuple(dimensions))
        if tuple(d % source.ndim for d in dimensions) != (source.ndim - 1,):
            continue
        outer = _region(output, cut, dependencies)
        if selected in outer:
            continue
        inner = _region(leaf, selected, dependencies)
        valid = True
        for node in inner - {selected}:
            value = node.meta.get("val")
            op = _op(node)
            if op in _SHAPE:
                continue
            if (not isinstance(value, Tensor) or tuple(map(str, value.shape)) != source_shape or
                    (op not in pointwise and activation_name(op, node.args, dict(node.kwargs)) is None)):
                valid = False
                break
        if valid:
            candidate = cut, leaf, inner, outer
            break
    if candidate is None:
        raise NotStructured("no dominating sum of pointwise independent features")
    cut, leaf, inner, outer = candidate
    request.planning_budget.check_graph(0, len(inner) + len(outer) * (order + 1))

    def lift(region: set[Node], seed: Node, layout: JetSpec) -> dict[Node, Any]:
        rows = [env[seed], torch.ones_like(env[seed])] + [torch.zeros_like(env[seed])] * (layout.order - 1)
        values: dict[Node, Any] = {seed: _Series(rows)}
        for node in nodes:
            if node not in region or node is seed:
                continue
            args = _map(node.args, lambda n: values.get(n, env[n]) if isinstance(n, Node) else n)
            kwargs = _map(node.kwargs, lambda n: values.get(n, env[n]) if isinstance(n, Node) else n)
            active = activation_name(_op(node), node.args, dict(node.kwargs))
            if active is not None and isinstance(args[0], _Series):
                name, options = active
                if layout.order == 1:
                    slope = lowering.tower(args[0].rows[0], 1, name, options)[1]
                    result = _Series([env[node], slope * args[0].rows[1]])
                elif node.args[0] is seed:
                    result = _Series(lowering.tower(env[seed], layout.order, name, options))
                else:
                    result = _Series(lowering.activate(args[0].rows, layout, name, options))
            else:
                result = _rule(node.target, args, kwargs, layout, "native", tracer, holder, lowering.kernels)
            if isinstance(result, _Series):
                result.rows[0] = env[node]
            values[node] = result
        return values

    first = lift(inner, selected, lowering.layout(1))[leaf]
    if not isinstance(first, _Series):
        raise NotStructured("separable branch did not produce a Taylor value")
    slope = first.rows[1]
    if _op(cut) == "mean":
        slope = slope / len(request.alpha)
    axes = [i for i, n in enumerate(request.alpha) if n]
    if len(axes) != len(request.alpha):
        slope = slope[..., axes]
    factor = slope.prod(dim=-1)
    output_rank = output.meta["val"].ndim
    if output_rank > batch_ndim:
        factor = factor[(Ellipsis,) + (None,) * (output_rank - batch_ndim)]
    outer_values = lift(outer, cut, lowering.layout(order))
    result = outer_values[output]
    if not isinstance(result, _Series):
        raise NotStructured("the outer scalar region did not produce a Taylor value")
    lowering.patterns.add("tensorized-separable-scalar-cut")
    return result.rows[order] * float(factorial(order)) * factor
