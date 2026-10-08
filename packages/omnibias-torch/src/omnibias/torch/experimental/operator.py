# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Structural operator lowering ahead of the general Taylor coefficient engine."""

from __future__ import annotations

import operator as python_operator
from math import comb, factorial, isfinite, prod
from typing import Any

from omnibias.core.jet_algebra import JetSpec
from omnibias.core.operator_spec import OperatorSpec

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fx import Graph, GraphModule, Node, Proxy
from torch.fx.proxy import GraphAppendingTracer
from torch.utils import _pytree

from ._kernel_lowering import KernelLowering, validate_lowering
from ._operator_apolar import apolar_derivative, select_apolar
from ._operator_ir import (
    NotStructured,
    TensorExpr,
    build_tensor_rule,
    feature_shape,
    variable,
)
from ._operator_lowering import Lowering, activation_name
from ._operator_separable import separable_derivative
from .compiler import (
    _ACTIVATIONS,
    _CREATION,
    _CUSTOM_RULES,
    _LINEAR,
    _OPS,
    _SHAPE,
    JetCompileError,
    JetProgram,
    _CapturedGraph,
    _check_tensor_bound,
    _emit_output,
    _map,
    _metadata_examples,
    _reject_capture_state_aliases,
    _validate_tensor,
    compile_jet,
)


class OperatorProgram(JetProgram):
    """Ordinary-input derivative module sharing the original Parameter objects."""

    def __init__(self, *args: Any, operator: OperatorSpec, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.operator = operator

    def forward(self, *args: Any, direction: Tensor | None = None) -> Any:
        if self.operator.mode == "directional":
            if direction is None:
                raise JetCompileError("a directional operator requires runtime direction=...")
            return super().forward(*args, direction)
        if direction is not None:
            raise JetCompileError("direction is only accepted for directional operators")
        return super().forward(*args)


class _BoundedGraph(Graph):
    """Bound speculative FX emission as well as the final live graph."""

    def __init__(self, maximum: int, emitted: list[int] | None = None) -> None:
        super().__init__()
        self.maximum = maximum
        self.emitted = [0] if emitted is None else emitted

    def create_node(self, *args: Any, **kwargs: Any) -> Node:
        if self.emitted[0] >= self.maximum:
            raise JetCompileError(f"operator graph node budget {self.maximum} exceeded during emission; lower derivative order or simplify the model")
        self.emitted[0] += 1
        return super().create_node(*args, **kwargs)


def _op(node: Node) -> str:
    text = str(node.target).split(".")
    return text[1] if len(text) > 1 and text[0] == "aten" else str(node.target)


def _node_arg(node: Node, index: int) -> Node:
    argument = node.args[index]
    if not isinstance(argument, Node):
        raise NotStructured(f"{node.name} argument {index} is not a tensor node")
    return argument


def _shape_guard(value: Any, exported: Any) -> tuple[Any, Any]:
    if not isinstance(value, Tensor):
        return None, value
    dims = []
    for dim in value.shape:
        if isinstance(dim, int):
            dims.append((dim, None, None))
        else:
            if not dim.node.expr.is_Symbol:
                raise JetCompileError("derived symbolic input dimensions are unsupported")
            bound = exported.range_constraints.get(dim.node.expr)
            lower = int(bound.lower) if bound is not None and str(bound.lower) not in {"-oo", "oo", "-int_oo", "int_oo"} else None
            upper = int(bound.upper) if bound is not None and str(bound.upper) not in {"-oo", "oo", "-int_oo", "int_oo"} else None
            if upper is None:
                raise JetCompileError("dynamic operator dimensions require finite upper bounds")
            dims.append((str(dim), lower, upper))
    return value.dtype, dims


def _audit(node: Node, tracked: bool, batch_ndim: int, dependencies: dict[Node, bool]) -> None:
    args: Any = node.args
    kwargs: Any = node.kwargs
    allowed = _LINEAR | _SHAPE | _CREATION | set(_ACTIVATIONS) | {
        "add", "sub", "mul", "div", "true_divide", "pow", "square", "linear", "mm", "matmul", "bmm", "addmm",
        "conv1d", "conv2d", "convolution", "cat", "stack", "gelu", "softplus", "softmax", "_softmax",
        "zeros_like", "ones_like", "full_like", "detach", "_to_copy",
    }
    special = {python_operator.getitem, python_operator.add, python_operator.sub,
               python_operator.mul, python_operator.floordiv, python_operator.mod}
    op = _op(node)
    if node.target not in special and node.target not in _CUSTOM_RULES and op not in allowed:
        raise JetCompileError(f"unsupported primitive {node.target}; register an explicit jet rule")
    if not tracked:
        return
    if op == "detach":
        raise JetCompileError("detach of a differentiated value is unsupported")
    value = node.meta.get("val")
    if isinstance(value, Tensor) and value.dtype not in {torch.float32, torch.float64}:
        raise JetCompileError("a differentiated operation produced a non-floating tensor")
    if not batch_ndim:
        return
    if node.target in _CUSTOM_RULES and op not in allowed and node.target not in special:
        raise JetCompileError("batch independence is not declared by this custom jet rule")
    # A spatial operator is per example. Refuse transformations whose batch
    # independence cannot be established structurally, including zero-valued
    # couplings that merely happen to vanish on example inputs.
    first = node.args[0] if node.args else None
    incoming = first.meta.get("val") if isinstance(first, Node) else None
    if not isinstance(first, Node) or not dependencies.get(first, False):
        incoming = next((parent.meta.get("val") for parent in node.all_input_nodes if dependencies.get(parent, False) and isinstance(parent.meta.get("val"), Tensor)), None)
    if op in {"linear", "mm", "matmul", "addmm"}:
        matrix = node.args[-1] if op == "addmm" else node.args[1]
        if isinstance(matrix, Node) and dependencies.get(matrix, False):
            raise JetCompileError("right-hand spatial matrix can couple batch examples")
        left = args[1] if op == "addmm" else args[0]
        left_value = left.meta.get("val") if isinstance(left, Node) else None
        if isinstance(left_value, Tensor) and dependencies.get(left, False) and left_value.ndim <= batch_ndim:
            raise JetCompileError("operator matrix product contracts a batch axis")
    if op in {"add", "sub", "mul", "div", "true_divide", "pow", "addmm", "linear"} and isinstance(value, Tensor):
        for parent in node.all_input_nodes:
            parent_value = parent.meta.get("val")
            if dependencies.get(parent, False) and isinstance(parent_value, Tensor) and parent_value.ndim != value.ndim:
                raise JetCompileError("a tracked broadcast would shift batch axes; explicitly preserve batch dimensions")
    if op in {"sum", "mean", "softmax", "_softmax"} and isinstance(incoming, Tensor):
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        dims = tuple(range(incoming.ndim)) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
        if any(d % incoming.ndim < batch_ndim for d in dims):
            raise JetCompileError("operator reduction mixes batch axes; use batch_ndim=0 on a single example")
    if op in {"permute", "transpose", "t", "numpy_T", "bmm", "conv1d", "conv2d", "convolution", "repeat", "tile", "flip", "roll", "pad", "constant_pad_nd"}:
        raise JetCompileError(f"batch independence is not proved for {node.target}; compile single examples")
    if op in {"select", "slice"} and isinstance(incoming, Tensor) and int(args[1]) % incoming.ndim < batch_ndim:
        raise JetCompileError("operator indexing mixes batch axes")
    if op in {"cat", "stack", "split", "split_with_sizes", "unbind"} and isinstance(incoming, Tensor):
        position = 1 if op in {"cat", "stack", "unbind"} else 2
        dim = args[position] if len(args) > position else kwargs.get("dim", 0)
        rank = incoming.ndim + int(op == "stack")
        if int(dim) % rank < batch_ndim:
            raise JetCompileError("operator concatenation or split mixes batch axes")
    if isinstance(incoming, Tensor) and isinstance(value, Tensor):
        if len(value.shape) < batch_ndim or tuple(map(str, incoming.shape[:batch_ndim])) != tuple(map(str, value.shape[:batch_ndim])):
            raise JetCompileError("operator operation changes batch axes")


def _scalar_ir(exported: Any, env: dict[Node, Any], selected: Node,
               dependencies: dict[Node, bool], batch_ndim: int, request: OperatorSpec,
               failures: list[str]) -> dict[Node, Any]:
    expressions: dict[Node, Any] = {}
    work = 0
    for node in exported.graph_module.graph.nodes:
        if node.op == "output":
            continue
        value = node.meta.get("val")
        shape = tuple(value.shape) if isinstance(value, Tensor) else ()
        if node is selected:
            dimension = int(shape[-1])
            work += dimension
            request.planning_budget.check_graph(0, work)
            expressions[node] = TensorExpr(env[node], shape, (dimension,), tuple(variable(env[node][..., i], i) for i in range(dimension)))
        elif node.op == "placeholder" or not dependencies.get(node, False):
            expressions[node] = TensorExpr(env[node], shape) if isinstance(value, Tensor) else env[node]
        elif isinstance(value, Tensor):
            try:
                features = feature_shape(shape, batch_ndim)
                count = prod(features)
                if _op(node) == "linear":
                    count *= int(node.args[1].meta["val"].shape[-1])
                work += count
                request.planning_budget.check_graph(0, work)
                if count > request.planning_budget.max_graph_nodes:
                    raise NotStructured("scalar feature expansion exceeds the graph planning budget")
                args = _map(node.args, lambda n: expressions[n] if isinstance(n, Node) else n)
                kwargs = _map(node.kwargs, lambda n: expressions[n] if isinstance(n, Node) else n)
                expressions[node] = build_tensor_rule(_op(node), args, kwargs, env[node], shape, batch_ndim, activation_name)
            except (NotStructured, KeyError, TypeError, IndexError) as exc:
                expressions[node] = None
                failures.append(f"{node.name}: {exc}")
        else:
            expressions[node] = env[node]
    return expressions


def _chain(output: Node, selected: Node, dependencies: dict[Node, bool]) -> list[tuple[Node, str]]:
    layers: list[tuple[Node, str]] = []
    node = output
    while node is not selected:
        if node.op != "call_function" or not node.args or not isinstance(node.args[0], Node):
            raise NotStructured("not a linear/activation chain")
        op = _op(node)
        if op in {"view", "reshape", "squeeze", "unsqueeze", "clone", "alias", "contiguous"}:
            if layers:
                raise NotStructured("interior shape transformation")
            node = node.args[0]
            continue
        if op == "linear":
            if any(dependencies.get(arg, False) for arg in node.args[1:] if isinstance(arg, Node)):
                raise NotStructured("spatially varying affine parameters")
        elif activation_name(op, node.args, dict(node.kwargs)) is None:
            raise NotStructured("not a linear/activation chain")
        layers.append((node, op))
        node = _node_arg(node, 0)
    return list(reversed(layers))


def _restore_chain_shapes(result: Any, output: Node, terminal: Node, env: dict[Node, Any],
                          tracer: GraphAppendingTracer) -> Any:
    """Replay proven terminal shape operations without evaluating the primal."""
    wrappers = []
    node = output
    while node is not terminal:
        wrappers.append(node)
        node = _node_arg(node, 0)
    for node in reversed(wrappers):
        rest = _map(node.args[1:], lambda n: env[n] if isinstance(n, Node) else n)
        kwargs = _map(node.kwargs, lambda n: env[n] if isinstance(n, Node) else n)
        result = tracer.create_proxy("call_function", node.target, (result, *rest), kwargs)
    return result


def _ridge(layers: list[tuple[Node, str]], env: dict[Node, Any], request: OperatorSpec,
           lowering: Lowering, direction: Any) -> Any:
    nonlinear = [i for i, (_, op) in enumerate(layers) if op != "linear"]
    if len(nonlinear) != 1 or nonlinear[0] == 0:
        raise NotStructured("not a ridge chain")
    index = nonlinear[0]
    node, op = layers[index]
    name, options = activation_name(op, node.args, dict(node.kwargs)) or ("", {})
    weights = [env[_node_arg(layer, 1)] for layer, _ in layers[:index]]
    dim = int(_node_arg(layers[0][0], 1).meta["val"].shape[-1])
    z = env[_node_arg(node, 0)]
    towers: dict[int, list[Any]] = {}
    constant_weight = (sum(float(c) for alpha, c in request.terms if not any(alpha))
                       if request.mode == "polynomial" else int(request.order == 0))

    def derivative(order: int) -> Any:
        if order == 0:
            return env[node]
        if order not in towers:
            towers[order] = lowering.tower(z, order, name, options)
        return towers[order][order] * float(factorial(order))

    if request.mode == "directional":
        slope = direction
        for weight in weights:
            slope = F.linear(slope, weight)
        out = derivative(request.order) * slope ** request.order
    elif request.mode == "laplacian":
        assert request.power is not None
        if len(weights) > 1 and int(_node_arg(layers[0][0], 1).meta["val"].shape[0]) < dim:
            gram = weights[0] @ weights[0].t()
            after = weights[1]
            for weight in weights[2:]:
                after = weight @ after
            norm2 = ((after @ gram) * after).sum(-1)
            lowering.patterns.add("factorized-ridge-metric")
        else:
            total = weights[0]
            for weight in weights[1:]:
                total = weight @ total
            norm2 = (total * total).sum(-1)
        out = derivative(2 * request.power) * norm2 ** request.power
    else:
        out = None
        for alpha, coefficient in request.resolve_terms(dim):
            axes = [i for i, n in enumerate(alpha) if n]
            factor: Any = 1
            if axes:
                columns = weights[0][:, axes]
                for weight in weights[1:]:
                    columns = weight @ columns
                for column, axis in enumerate(axes):
                    factor = factor * columns[:, column] ** alpha[axis]
            term = derivative(sum(alpha)) * factor * float(coefficient)
            out = term if out is None else out + term
    for layer, _ in layers[index + 1:]:
        out = F.linear(out, env[_node_arg(layer, 1)])
        # A zero-order request also retains the affine readout bias.
        if constant_weight and len(layer.args) > 2 and layer.args[2] is not None:
            out = out + env[_node_arg(layer, 2)] * constant_weight
    lowering.patterns.add("ridge-contraction")
    return out


def _transport(layers: list[tuple[Node, str]], env: dict[Node, Any], request: OperatorSpec,
               lowering: Lowering, dimension: int) -> Any:
    """Transport exact operators through a structurally affine bottleneck."""
    if not layers or layers[0][1] != "linear":
        raise NotStructured("no affine bottleneck")
    candidates = []
    for i, (node, op) in enumerate(layers):
        if op != "linear":
            break
        candidates.append((int(_node_arg(node, 1).meta["val"].shape[0]), i))
    latent, cut_index = min(candidates)
    if latent >= dimension:
        raise NotStructured("affine cut does not reduce coordinate dimension")
    transform = env[_node_arg(layers[0][0], 1)]
    for node, _ in layers[1:cut_index + 1]:
        transform = env[_node_arg(node, 1)] @ transform
    z = env[layers[cut_index][0]]
    gram = transform @ transform.t()
    tail = layers[cut_index + 1:]
    metric_variants_supported = all(op == "linear" or not (activation_name(op, node.args, dict(node.kwargs)) or ("", {}))[1]
                                    for node, op in tail)
    if request.mode == "laplacian" and request.power in {1, 2} and lowering.method == "recurrence" and metric_variants_supported:
        from ._operator_metric import (
            laplacian_activation,
            laplacian_affine,
            laplacian_seed,
            metric_activation,
            metric_affine,
            metric_seed,
        )
        second = request.power == 2
        state_size = 2 + 2 * latent + latent * (latent + 1) // 2 if second else latent + 2
        lowering.check_state(state_size)
        state: Any = metric_seed(z, latent) if second else laplacian_seed(z, latent)
        for node, op in tail:
            if op == "linear":
                bias = env[_node_arg(node, 2)] if len(node.args) > 2 and node.args[2] is not None else None
                state = (metric_affine if second else laplacian_affine)(state, env[_node_arg(node, 1)], bias)
            else:
                name, options = activation_name(op, node.args, dict(node.kwargs)) or ("", {})
                if options:
                    raise NotStructured("metric state lacks this activation's source options")
                state = metric_activation(state, gram, latent, name) if second else laplacian_activation(state, gram, name)
        lowering.patterns.add("affine-bottleneck-weighted-state")
        lowering.max_coefficients = max(lowering.max_coefficients, state_size)
        return state.biharmonic if second else state.laplacian
    if request.mode == "directional":
        raise NotStructured("directional transport already uses the univariate fallback")
    # General P(D_x) becomes P(B^T D_z). Expand only in the small latent
    # coordinate space, keeping every coefficient a live tensor expression.
    maximum = comb(latent + request.order, request.order)
    request.budget.check_layout(maximum, 0)
    request.planning_budget.check_graph(0, maximum * max(1, request.order))
    zero = (0,) * latent

    def times(a: dict[tuple[int, ...], Any], b: dict[tuple[int, ...], Any]) -> dict[tuple[int, ...], Any]:
        request.planning_budget.check_graph(0, len(a) * len(b) * latent)
        result: dict[tuple[int, ...], Any] = {}
        for alpha, ca in a.items():
            for beta, cb in b.items():
                key = tuple(x + y for x, y in zip(alpha, beta, strict=True))
                result[key] = result.get(key, 0) + ca * cb
        if len(result) > request.planning_budget.max_symbol_terms:
            raise NotStructured("transported symbol exceeds its term budget")
        return result

    symbol: dict[tuple[int, ...], Any] = {}
    if request.mode == "laplacian":
        quadratic = {}
        for i in range(latent):
            for j in range(i, latent):
                alpha = tuple(int(k == i) + int(k == j) for k in range(latent))
                quadratic[alpha] = gram[i, j] * (1 if i == j else 2)
        symbol = {zero: 1}
        for _ in range(request.power or 0):
            symbol = times(symbol, quadratic)
    else:
        for alpha, coefficient in request.resolve_terms(dimension):
            term = {zero: float(coefficient)}
            for axis, power in enumerate(alpha):
                linear = {tuple(int(k == i) for k in range(latent)): transform[i, axis] for i in range(latent)}
                for _ in range(power):
                    term = times(term, linear)
            for beta, value in term.items():
                symbol[beta] = symbol.get(beta, 0) + value
    layout = JetSpec.mixed(symbol, request.budget)
    lowering.check_state(layout.size)
    rows = []
    for alpha in layout.indices:
        if not any(alpha):
            rows.append(z)
        elif sum(alpha) == 1:
            axis = alpha.index(1)
            rows.append(torch.stack([torch.ones_like(z[..., i]) if i == axis else torch.zeros_like(z[..., i]) for i in range(latent)], dim=-1))
        else:
            rows.append(torch.zeros_like(z))
    for node, op in tail:
        if op == "linear":
            packed = F.linear(torch.stack(rows), env[_node_arg(node, 1)])
            rows = [packed[i] for i in range(layout.size)]
            if len(node.args) > 2 and node.args[2] is not None:
                rows[0] = rows[0] + env[_node_arg(node, 2)]
        else:
            name, options = activation_name(op, node.args, dict(node.kwargs)) or ("", {})
            rows = lowering.activate(rows, layout, name, options)
        rows[0] = env[node]
    result = sum(rows[layout.index(alpha)] * coefficient * float(prod(factorial(a) for a in alpha)) for alpha, coefficient in symbol.items())
    lowering.patterns.add("affine-bottleneck-polynomial-transport")
    lowering.max_coefficients = max(lowering.max_coefficients, layout.size)
    return result


def compile_operator(model: nn.Module, example_args: tuple[Any, ...], *, operator: OperatorSpec,
                     argnum: int = 0, batch_ndim: int = 0, dynamic_shapes: Any = None,
                     activation_method: str = "recurrence", strategy: str = "auto",
                     kernel_lowering: str = "auto",
                     _exported: _CapturedGraph | None = None) -> OperatorProgram:
    """Compile one raw spatial operator with structural fastpaths and exact fallback.

    The selected tensor is ``(*batch, coordinates)``. Batch axes are independent
    examples, never derivative coordinates. Parameter values do not establish
    sparsity or rank. ``polynomial`` is an explicit, accuracy-sensitive opt-in
    to the original activation providers; recurrence is the default.
    """
    if not isinstance(model, nn.Module) or not isinstance(example_args, tuple):
        raise TypeError("compile_operator expects an nn.Module and tuple of primal example arguments")
    if not isinstance(operator, OperatorSpec):
        raise TypeError("operator must be an OperatorSpec")
    validate_lowering(kernel_lowering)
    for _, coefficient in operator.terms:
        try:
            static_weight = float(coefficient)
        except OverflowError as exc:
            raise JetCompileError("static operator coefficient is not representable as a finite nonzero binary64 weight") from exc
        if not isfinite(static_weight) or (coefficient and static_weight == 0):
            raise JetCompileError("static operator coefficient is not representable as a finite nonzero binary64 weight")
    if strategy not in {"auto", "reference"} or activation_method not in {"recurrence", "polynomial"}:
        raise ValueError("strategy must be auto/reference and activation_method recurrence/polynomial")
    if not isinstance(argnum, int) or isinstance(argnum, bool) or not 0 <= argnum < len(example_args):
        raise ValueError("argnum must identify one positional tensor argument")
    x = example_args[argnum]
    if not isinstance(x, Tensor) or not isinstance(batch_ndim, int) or isinstance(batch_ndim, bool) or batch_ndim < 0 or x.ndim != batch_ndim + 1:
        raise ValueError("selected input must have batch_ndim batch axes and one coordinate axis")
    _validate_tensor(x)
    dimension = operator.check_dimension(int(x.shape[-1]))
    if _exported is None:
        _reject_capture_state_aliases(model, example_args)
        try:
            exported = _CapturedGraph.from_export(torch.export.export(model, example_args, dynamic_shapes=dynamic_shapes, strict=True))
        except Exception as exc:
            raise JetCompileError(f"torch.export capture failed: {exc}") from exc
    else:
        exported = _exported
    operator.planning_budget.check_graph(len(list(exported.graph_module.graph.nodes)))
    if any(out.kind.name != "USER_OUTPUT" for out in exported.graph_signature.output_specs):
        raise JetCompileError("mutation/effect outputs are unsupported for spatial operators")
    selected_leaf = sum(len(_pytree.tree_flatten(arg)[0]) for arg in example_args[:argnum])
    graph, holder = _BoundedGraph(operator.planning_budget.max_graph_nodes), nn.Module()
    tracer = GraphAppendingTracer(graph)
    holder.add_module("model", model)
    constants = nn.Module()
    holder.add_module("constants", constants)
    env: dict[Node, Any] = {}
    expressions: dict[Node, Any] = {}
    dependencies: dict[Node, bool] = {}
    guards: list[Any] = []
    user_proxies: list[Any] = []
    outputs: list[Any] = []
    selected_node: Node | None = None
    input_specs = iter(exported.graph_signature.input_specs)
    structure_failures: list[str] = []
    for node in exported.graph_module.graph.nodes:
        value = node.meta.get("val")
        shape = tuple(value.shape) if isinstance(value, Tensor) else ()
        if node.op == "placeholder":
            info = next(input_specs)
            if info.kind.name in {"PARAMETER", "BUFFER"}:
                proxy = Proxy(graph.get_attr("model." + info.target), tracer)
            elif info.kind.name == "CONSTANT_TENSOR":
                name = f"tensor_{len(constants._buffers)}"
                constants.register_buffer(name, exported.constants[info.target])
                proxy = Proxy(graph.get_attr("constants." + name), tracer)
            elif info.kind.name == "USER_INPUT":
                proxy = Proxy(graph.placeholder(str(node.target)), tracer)
                index = len(user_proxies)
                user_proxies.append(proxy)
                dtype, guard = _shape_guard(value, exported)
                if dtype is None:
                    guard = getattr(info.arg, "value", value)
                guards.append((index, False, dtype, guard))
                if index == selected_leaf:
                    if not isinstance(shape[-1], int):
                        raise JetCompileError("operator coordinate dimension must be static; only batch dimensions may vary")
                    selected_node = node
                    dependencies[node] = True
            else:
                raise JetCompileError(f"unsupported export input kind {info.kind.name}")
            env[node] = proxy
            expressions.setdefault(node, TensorExpr(proxy, shape) if isinstance(value, Tensor) else proxy)
        elif node.op == "call_function":
            active = any(dependencies.get(parent, False) for parent in node.all_input_nodes)
            dependencies[node] = active
            try:
                _audit(node, active, batch_ndim, dependencies)
            except Exception as exc:
                raise JetCompileError(f"operator audit failed at {node.name} ({node.target}): {exc}\n{node.meta.get('stack_trace', '')}") from exc
            args = _map(node.args, lambda n: env[n] if isinstance(n, Node) else n)
            kwargs = _map(node.kwargs, lambda n: env[n] if isinstance(n, Node) else n)
            env[node] = tracer.create_proxy("call_function", node.target, args, kwargs)
        elif node.op == "output":
            outputs = list(node.args[0])
        else:
            raise JetCompileError(f"unsupported export node kind {node.op}")
    assert selected_node is not None
    if any(not isinstance(output, Node) or not isinstance(output.meta.get("val"), Tensor) for output in outputs):
        raise JetCompileError("operator outputs must be tensors")
    direction: Any = None
    input_structure = _pytree.tree_flatten(example_args)[1]
    if operator.mode == "directional":
        direction = Proxy(graph.placeholder("direction"), tracer)
        guards.append((len(user_proxies), False, guards[selected_leaf][2], guards[selected_leaf][3]))
        input_structure = _pytree.tree_flatten((*example_args, x))[1]
    kernels = KernelLowering(kernel_lowering, _OPS, tracer, holder,
                             lambda: _BoundedGraph(graph.maximum, graph.emitted))
    lowering = Lowering(operator, activation_method, kernels)
    bound_layout = JetSpec.directional(0, operator.budget)
    lowering.tensor_bytes_per_coefficient = max((_check_tensor_bound(node.meta["val"], bound_layout, exported.range_constraints)
                for node in exported.graph_module.graph.nodes if dependencies.get(node, False) and isinstance(node.meta.get("val"), Tensor)), default=0)
    lowered: list[Any] = []
    fallback_reason = None
    apolar_report = None
    reference_kernels: tuple[dict[str, Any], ...] = ()
    try:
        if strategy == "reference":
            raise NotStructured("reference strategy explicitly requested")
        for output in outputs:
            if operator.order == 0 and operator.mode != "polynomial":
                lowered.append(env[output])
                continue
            try:
                layers = _chain(output, selected_node, dependencies)
                try:
                    result = _ridge(layers, env, operator, lowering, direction)
                except NotStructured:
                    result = _transport(layers, env, operator, lowering, dimension)
                lowered.append(_restore_chain_shapes(result, output, layers[-1][0], env, tracer))
                continue
            except NotStructured:
                pass
            try:
                lowered.append(separable_derivative(output, selected_node, list(exported.graph_module.graph.nodes), env,
                                                     dependencies, operator, lowering, batch_ndim, tracer, holder))
                continue
            except NotStructured:
                pass
            if selected_node not in expressions or not isinstance(expressions[selected_node], TensorExpr) or expressions[selected_node].expressions is None:
                expressions = _scalar_ir(exported, env, selected_node, dependencies, batch_ndim, operator, structure_failures)
            ir = expressions.get(output)
            if operator.mode == "directional" or not isinstance(ir, TensorExpr) or ir.expressions is None:
                raise NotStructured("no proven compressed operator region")
            terms = operator.resolve_terms(dimension)
            values = [sum(lowering.derivative(expr, alpha) * float(coefficient) for alpha, coefficient in terms)
                      for expr in ir.expressions]
            lowered.append(torch.stack(values, dim=-1).reshape(env[output].shape))
            lowering.patterns.add("compressed-operator-polynomial")
    except (NotStructured, ValueError) as exc:
        if isinstance(exc, JetCompileError):
            raise
        fallback_reason = str(exc)
        try:
            if strategy == "reference":
                raise NotStructured("reference strategy explicitly requested")
            algebra = select_apolar(operator, dimension)
            lowering.check_state(algebra.size)
            if selected_node not in expressions or not isinstance(expressions[selected_node], TensorExpr) or expressions[selected_node].expressions is None:
                expressions = _scalar_ir(exported, env, selected_node, dependencies, batch_ndim, operator, structure_failures)
            if any(not isinstance(expressions.get(node), TensorExpr) or expressions[node].expressions is None for node in outputs):
                raise NotStructured("the graph has no complete scalar region for apolar lowering")
            memo: dict[Any, list[Any]] = {}
            lowered = [torch.stack([apolar_derivative(expr, algebra, lowering, memo) for expr in expressions[node].expressions], dim=-1).reshape(env[node].shape)
                       for node in outputs]
            apolar_report = algebra.explain()
            structure_failures.append(f"compressed region declined: {fallback_reason}")
            fallback_reason = None
        except (NotStructured, ValueError) as apolar_exc:
            if isinstance(apolar_exc, JetCompileError):
                raise
            structure_failures.append(str(apolar_exc))
    if fallback_reason is not None:
        if activation_method == "polynomial":
            # The generic coefficient engine remains recurrence based. Expose
            # this change of strategy, rather than calling it a polynomial run.
            lowering.patterns.add("polynomial-provider-fallback-to-recurrence")
        layout = operator.jet_spec(dimension)
        reference = compile_jet(model, example_args, spec=layout, argnums=(argnum,), dynamic_shapes=dynamic_shapes,
                                kernel_lowering=kernel_lowering,
                                _graph_factory=lambda: _BoundedGraph(graph.maximum, graph.emitted), _exported=exported)
        reference_kernels = reference.explain()["kernel_lowering"]["regions"]
        holder.add_module("reference", reference.graph_module)
        coordinate = env[selected_node]
        seeds = []
        for alpha in layout.indices:
            if not any(alpha):
                seeds.append(coordinate)
            elif sum(alpha) == 1:
                if operator.mode == "directional":
                    seeds.append(direction)
                else:
                    axis = alpha.index(1)
                    seeds.append(torch.stack([torch.ones_like(coordinate[..., i]) if i == axis else torch.zeros_like(coordinate[..., i]) for i in range(dimension)], dim=-1))
            else:
                seeds.append(torch.zeros_like(coordinate))
        arguments = list(user_proxies)
        arguments[selected_leaf] = torch.stack(seeds)
        reference_result: Any = tracer.create_proxy("call_module", "reference", tuple(arguments), {})
        # Export outputs are flattened; the coefficient program reconstructs
        # the original tree. Walk that tree structurally using the export spec.
        output_spec = exported.call_spec.out_spec
        operator.planning_budget.check_graph(0, output_spec.num_leaves)
        dummy_output = _pytree.tree_unflatten([None] * output_spec.num_leaves, output_spec)
        keyed_leaves, _ = _pytree.tree_flatten_with_path(dummy_output)
        lowered = []
        for path, _ in keyed_leaves:
            coefficients = reference_result
            for key in path:
                coefficients = key.get(coefficients)
            total = None
            for index, weight in operator.jet_readout(layout):
                term = coefficients[index] * float(weight)
                total = term if total is None else total + term
            lowered.append(total)
        lowering.max_coefficients = layout.size
        lowering.patterns.add("general-taylor-fallback")
    _emit_output(graph, lowered, exported.call_spec.out_spec)
    graph.lint()
    module = GraphModule(holder, graph, "OmnibiasOperatorGraph")
    module.graph.eliminate_dead_code()
    module.recompile()
    graph_nodes = sum(len(list(child.graph.nodes)) for child in module.modules() if isinstance(child, GraphModule))
    operator.planning_budget.check_graph(graph_nodes)
    # This conservative single-tensor bound includes the largest original
    # intermediate times retained scalar coefficients; workspaces/tapes differ.
    peak = lowering.max_coefficients * lowering.tensor_bytes_per_coefficient
    if peak > operator.budget.max_tensor_bytes:
        raise JetCompileError(f"operator forward state bound {peak} bytes exceeds budget {operator.budget.max_tensor_bytes}; reduce batch or width")
    report = {"backend": "torch.export+fx", "backend_version": torch.__version__, "operator": operator.explain(dimension),
              "strategy_requested": strategy, "selected_strategy": "reference" if fallback_reason else ("apolar" if apolar_report else "structured"),
              "activation_method": activation_method, "matched_patterns": tuple(sorted(lowering.patterns)),
              "kernel_lowering": {**kernels.explain(), "regions": (*kernels.explain()["regions"], *reference_kernels)},
              "fallback_reason": fallback_reason, "structural_diagnostics": tuple(structure_failures),
              "coefficient_count": lowering.max_coefficients, "graph_nodes": graph_nodes,
              "emitted_graph_nodes": graph.emitted[0],
              "apolar_algebra": apolar_report,
              "peak_single_state_bytes_bound": peak, "batch_ndim": batch_ndim,
              "budget_scope": "conservative single forward tensor; excludes backward tape and backend workspaces",
              "polynomial_accuracy_note": "explicit opt-in; high-order cancellation must be measured" if activation_method == "polynomial" else None}
    program = OperatorProgram(module, JetSpec.directional(0, operator.budget), (), guards, report, input_structure, operator=operator)
    metadata_args = _metadata_examples(example_args)
    object.__setattr__(program, "_canonical_factory", lambda: compile_operator(
        model, metadata_args, operator=operator, argnum=argnum, batch_ndim=batch_ndim,
        dynamic_shapes=dynamic_shapes, activation_method=activation_method, strategy=strategy,
        kernel_lowering="inline", _exported=exported))
    return program


__all__ = ["OperatorProgram", "compile_operator"]
