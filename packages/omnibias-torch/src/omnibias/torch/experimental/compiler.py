# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Strict export-to-FX Taylor transformation with live model parameters."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import activation, multiply, reciprocal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.fx import Graph, GraphModule, Node, Proxy
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.fx.proxy import GraphAppendingTracer
from torch.utils import _pytree

from ._kernel_lowering import KernelLowering, activation_graph, validate_lowering


class JetCompileError(ValueError):
    """An exported graph cannot be transformed under the requested contract."""


@dataclass(frozen=True)
class _CapturedGraph:
    """Only lowering metadata, without ExportedProgram's example input graph."""

    graph_module: GraphModule
    graph_signature: Any
    call_spec: Any
    range_constraints: Any
    constants: Any

    @classmethod
    def from_export(cls, exported: Any) -> _CapturedGraph:
        return cls(exported.graph_module, exported.graph_signature, exported.call_spec,
                   exported.range_constraints, exported.constants)


def _metadata_examples(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return cast(tuple[Any, ...], _pytree.tree_map(
        lambda value: torch.empty_like(value, device="meta", requires_grad=False) if isinstance(value, Tensor) else value,
        args))


def _reject_capture_state_aliases(model: nn.Module, args: tuple[Any, ...]) -> None:
    state = {id(value): name for name, value in (*model.named_parameters(), *model.named_buffers())}
    seen: set[int] = set()
    for value in _pytree.tree_leaves(args):
        if isinstance(value, Tensor) and id(value) in state:
            raise JetCompileError(f"capture example tensor aliases model state {state[id(value)]!r}; "
                                  "use an independent example tensor to keep input and parameter roles distinct")
        if isinstance(value, Tensor):
            if id(value) in seen:
                raise JetCompileError("duplicate capture example tensor; distinct example tensors identify independent "
                                      "argument roles; runtime aliases are supported")
            seen.add(id(value))


_OPS = SimpleNamespace(
    exp=torch.exp, log=torch.log, sqrt=torch.sqrt, tanh=torch.tanh,
    sigmoid=torch.sigmoid, sin=torch.sin, cos=torch.cos, erf=torch.erf,
    erfc=torch.erfc, zeros_like=torch.zeros_like, ones_like=torch.ones_like,
    logaddexp=torch.logaddexp, softplus=F.softplus,
)


def _validate_tensor(x: Tensor) -> None:
    if x.dtype not in (torch.float32, torch.float64):
        raise JetCompileError("jet inputs require float32 or float64 (half, complex and integer jets are unsupported)")


def seed_jet(
    value: Tensor, spec: JetSpec, coefficients: dict[tuple[int, ...], Tensor] | None = None,
) -> Tensor:
    """Pack a primal and optional Taylor coefficients without detaching either.

    Missing coefficients are zero. Values supplied in ``coefficients`` are
    already divided by the multi-index factorial, rather than raw derivatives.
    """
    _validate_tensor(value)
    spec.check_tensor(tuple(value.shape), value.element_size())
    supplied = coefficients or {}
    for alpha in supplied:
        spec.index(alpha)
        if not any(alpha):
            raise ValueError("the constant coefficient is supplied as value")
        coefficient = supplied[alpha]
        if coefficient.shape != value.shape or coefficient.dtype != value.dtype or coefficient.device != value.device:
            raise ValueError("every seed coefficient must match the primal shape, dtype and device")
    return torch.stack([value if not any(a) else supplied.get(a, torch.zeros_like(value)) for a in spec.indices])


def directional_seed(value: Tensor, direction: Tensor, spec: JetSpec | int) -> Tensor:
    """Seed ``value + t * direction``; the direction remains trainable."""
    if isinstance(spec, int):
        spec = JetSpec.directional(spec)
    if spec.dim != 1:
        raise ValueError("directional_seed requires a one-variable specification")
    if value.shape != direction.shape:
        raise ValueError("value and direction must have identical shapes")
    if value.dtype != direction.dtype or value.device != direction.device:
        raise ValueError("value and direction must have identical dtypes and devices")
    _validate_tensor(direction)
    return seed_jet(value, spec, {(1,): direction} if spec.order else {})


def coefficients_to_derivatives(coefficients: Tensor, spec: JetSpec) -> Tensor:
    """Multiply each leading coefficient by its multi-index factorial."""
    if coefficients.shape[0] != spec.size:
        raise ValueError("leading coefficient count does not match JetSpec")
    factors = coefficients.new_tensor(spec.factorials)
    return coefficients * factors.reshape((-1,) + (1,) * (coefficients.ndim - 1))


def extract_derivative(coefficients: Tensor, spec: JetSpec, alpha: tuple[int, ...] | int) -> Tensor:
    """Extract one raw derivative from a packed output jet."""
    if isinstance(alpha, int):
        alpha = (alpha,)
    row = spec.index(alpha)
    return coefficients[row] * float(spec.factorials[row])


def _scale(a: Sequence[Any], scale: Any) -> list[Any]:
    return [x * scale for x in a]


def _add(a: Sequence[Any], b: Sequence[Any]) -> list[Any]:
    return [x + y for x, y in zip(a, b, strict=True)]


class _ActivationVJP(torch.autograd.Function):
    @staticmethod
    def forward(packed: Tensor, forward_graph: nn.Module, slope_graph: nn.Module, pairs: Any) -> Tensor:
        return cast(Tensor, forward_graph(packed))

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
        packed, _, slope_graph, pairs = inputs
        ctx.save_for_backward(packed)
        ctx.save_for_forward(packed)
        ctx.slope_graph, ctx.pairs = slope_graph, pairs

    @staticmethod
    def backward(ctx: Any, cotangent: Tensor) -> tuple[Tensor, None, None, None]:
        (packed,) = ctx.saved_tensors
        # Recompute from the saved input, preserving the graph for double backward.
        slope = ctx.slope_graph(packed)
        rows = [torch.zeros_like(packed[0]) for _ in ctx.pairs]
        for out, pairs in enumerate(ctx.pairs):
            for i, j in pairs:
                rows[i] = rows[i] + cotangent[out] * slope[j]
        return torch.stack(rows), None, None, None


class _TransformActivationVJP(_ActivationVJP):
    """Eager functorch companion; Dynamo currently rejects custom JVP methods."""

    generate_vmap_rule = True

    @staticmethod
    def jvp(ctx: Any, tangent: Tensor, _forward: Any, _slope: Any, _pairs: Any) -> Tensor:
        (packed,) = ctx.saved_tensors
        slope = ctx.slope_graph(packed)
        rows = []
        for pairs in ctx.pairs:
            row = torch.zeros_like(packed[0])
            for i, j in pairs:
                row = row + tangent[i] * slope[j]
            rows.append(row)
        return torch.stack(rows)


def _activation_graph(spec: JetSpec, name: str, derivative: bool) -> GraphModule:
    return activation_graph(spec, name, _OPS, derivative=derivative)


class _AnalyticActivation(nn.Module):
    def __init__(self, spec: JetSpec, name: str, kernels: KernelLowering | None = None) -> None:
        super().__init__()
        self.pairs = spec.pairs
        self.forward_graph = _activation_graph(spec, name, False) if kernels is None else kernels.module(spec, name)
        self.slope_graph = _activation_graph(spec, name, True) if kernels is None else kernels.module(spec, name, True)

    def forward(self, packed: Tensor) -> Tensor:
        # Both variants use the same analytic correlation backward. The eager
        # variant additionally exposes JVP/vmap, which Dynamo does not trace.
        function = _ActivationVJP if torch.compiler.is_compiling() else _TransformActivationVJP
        return cast(Tensor, function.apply(packed, self.forward_graph, self.slope_graph, self.pairs))


@dataclass
class _Series:
    rows: list[Any]


# A custom rule receives nested arguments containing coefficient lists in place
# of tracked tensors, kwargs, spec, and the torch operations adapter. It returns
# one coefficient list (or a tuple of lists for a multiple-result primitive).
_CUSTOM_RULES: dict[Any, Callable[..., Any]] = {}


def register_jet_rule(target: Any, rule: Callable[..., Any]) -> None:
    """Register an explicit pure coefficient rule for a torch operator overload.

    Registration affects subsequently compiled programs. The caller owns the
    rule's mathematical correctness, differentiability, and dtype semantics.
    """
    if not callable(rule):
        raise TypeError("rule must be callable")
    _CUSTOM_RULES[target] = rule


def _map(value: Any, fn: Callable[[Any], Any]) -> Any:
    if isinstance(value, tuple):
        return tuple(_map(v, fn) for v in value)
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map(v, fn) for k, v in value.items()}
    return fn(value)


def _emit_output(graph: Graph, leaves: list[Any], output_spec: Any) -> None:
    """Keep registered output containers intact in the generated FX function."""
    def needs_pytree_codegen(spec: Any) -> bool:
        return not spec.is_leaf() and (spec.type not in {tuple, list, dict} or any(needs_pytree_codegen(child) for child in spec.children()))

    if needs_pytree_codegen(output_spec):
        names = [str(node.target) for node in graph.nodes if node.op == "placeholder"]
        flat_input_spec = _pytree.tree_flatten([None] * len(names))[1]
        graph.set_codegen(_PyTreeCodeGen(_PyTreeInfo(names, flat_input_spec, output_spec)))
        graph.output([leaf.node if isinstance(leaf, Proxy) else leaf for leaf in leaves])
    else:
        output = _pytree.tree_unflatten(leaves, output_spec)
        graph.output(_map(output, lambda leaf: leaf.node if isinstance(leaf, Proxy) else leaf))


def _primal(value: Any) -> Any:
    return _map(value, lambda v: v.rows[0] if isinstance(v, _Series) else v)


def _tracked(value: Any) -> bool:
    leaves: list[bool] = []
    _map(value, lambda v: leaves.append(isinstance(v, _Series)))
    return any(leaves)


def _zero(value: Any) -> Any:
    return value * 0 if isinstance(value, (int, float)) else torch.zeros_like(value)


def _rows(value: Any, spec: JetSpec) -> list[Any]:
    return value.rows if isinstance(value, _Series) else [value] + [_zero(value) for _ in range(spec.size - 1)]


_ACTIVATIONS = {
    "tanh": "tanh", "sigmoid": "sigmoid", "sin": "sin", "cos": "cos",
    "exp": "exp", "log": "log", "sqrt": "sqrt", "reciprocal": "reciprocal",
    "erf": "erf", "erfc": "erfc", "silu": "silu",
}
_LINEAR = {
    "view", "reshape", "_unsafe_view", "flatten", "squeeze", "unsqueeze",
    "permute", "transpose", "t", "numpy_T", "select", "slice", "expand", "expand_as",
    "clone", "contiguous", "alias", "sum", "mean", "neg", "constant_pad_nd", "pad",
    "repeat", "tile", "flip", "roll", "split", "split_with_sizes", "unbind",
}
_SHAPE = {"sym_size", "sym_numel", "sym_stride", "size", "numel"}
_CREATION = {"zeros", "ones", "full", "arange", "scalar_tensor", "lift_fresh_copy"}


def _rule(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any], spec: JetSpec, backward: str, tracer: GraphAppendingTracer | None = None, holder: nn.Module | None = None, kernels: KernelLowering | None = None) -> Any:
    def compose(rows: list[Any], name: str) -> list[Any]:
        return activation(rows, spec, name, _OPS) if kernels is None else kernels.activate(rows, spec, name)

    if target is operator.getitem:
        obj, key = args
        if isinstance(obj, _Series):
            return _Series([row[key] for row in obj.rows])
        return obj[key]
    if target in (operator.add, operator.sub, operator.mul, operator.floordiv, operator.mod):
        if _tracked(args):
            raise JetCompileError("Python arithmetic on tracked values must be exported as tensor operations")
        return target(*args, **kwargs)
    if target in _CUSTOM_RULES:
        def unpack(v: Any) -> Any:
            return v.rows if isinstance(v, _Series) else v
        result = _CUSTOM_RULES[target](_map(args, unpack), _map(kwargs, unpack), spec, _OPS)
        return _Series(result) if isinstance(result, list) else tuple(_Series(r) for r in result)
    qualified = str(target).split(".")
    if len(qualified) < 2 or qualified[0] != "aten":
        raise JetCompileError(f"unsupported graph operation {target}")
    op = qualified[1]
    supported = _LINEAR | _SHAPE | _CREATION | set(_ACTIVATIONS) | {
        "add", "sub", "mul", "div", "true_divide", "pow", "square", "linear",
        "mm", "matmul", "bmm", "addmm", "conv1d", "conv2d", "convolution",
        "cat", "stack", "gelu", "softplus", "softmax", "_softmax",
        "zeros_like", "ones_like", "full_like", "detach", "_to_copy",
    }
    if op not in supported:
        raise JetCompileError(f"unsupported primitive {target}; register an explicit jet rule")
    if not _tracked((args, kwargs)):
        return tracer.create_proxy("call_function", target, args, kwargs) if tracer is not None else target(*args, **kwargs)
    if op in _SHAPE:
        return target(*_primal(args), **_primal(kwargs))
    if op == "detach":
        raise JetCompileError("detach of a differentiated value would discard its derivative contract")
    if op == "_to_copy":
        dtype = kwargs.get("dtype")
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise JetCompileError("casting a tracked jet to a non-float32/float64 dtype is unsupported")
        return _Series([target(row, **kwargs) for row in _rows(args[0], spec)])
    if op in {"zeros_like", "ones_like", "full_like"}:
        primal = target(*_primal(args), **kwargs)
        return _Series([primal] + [torch.zeros_like(primal) for _ in range(spec.size - 1)])
    if op in _LINEAR:
        if any(_tracked(a) for a in args[1:]) or _tracked(kwargs):
            raise JetCompileError(f"{target} requires static shape/index arguments")
        if op == "constant_pad_nd":
            pad = args[1]
            value = args[2] if len(args) > 2 else kwargs.get("value", 0)
            return _Series([target(row, pad, value if i == 0 else 0) for i, row in enumerate(_rows(args[0], spec))])
        if op == "pad":
            pad = args[1]
            mode = args[2] if len(args) > 2 else kwargs.get("mode", "constant")
            value = args[3] if len(args) > 3 else kwargs.get("value")
            return _Series([target(row, pad, mode, value if i == 0 or mode != "constant" else 0) for i, row in enumerate(_rows(args[0], spec))])
        rows = [target(row, *args[1:], **kwargs) for row in _rows(args[0], spec)]
        return _Series(rows)
    if op in {"cat", "stack"}:
        return _Series([target([_rows(a, spec)[i] for a in args[0]], *args[1:], **kwargs) for i in range(spec.size)])
    if op in {"add", "sub"}:
        a, b = (_rows(v, spec) for v in args[:2])
        return _Series([target(x, y, *args[2:], **kwargs) for x, y in zip(a, b, strict=True)])
    if op == "mul":
        if not isinstance(args[0], _Series):
            return _Series([args[0] * r for r in _rows(args[1], spec)])
        if not isinstance(args[1], _Series):
            return _Series([r * args[1] for r in _rows(args[0], spec)])
        return _Series(multiply(args[0].rows, args[1].rows, spec, _OPS))
    if op in {"div", "true_divide"}:
        if kwargs.get("rounding_mode") is not None:
            raise JetCompileError("integer/rounded division is not differentiable")
        if not isinstance(args[1], _Series):
            return _Series([r / args[1] for r in _rows(args[0], spec)])
        return _Series(multiply(_rows(args[0], spec), reciprocal(args[1].rows, spec, _OPS), spec, _OPS))
    if op in {"pow", "square"}:
        exponent = 2 if op == "square" else args[1]
        if isinstance(exponent, float) and exponent.is_integer():
            exponent = int(exponent)
        a = _rows(args[0], spec)
        if isinstance(exponent, int):
            if exponent == 0:
                return _Series([torch.ones_like(a[0])] + [torch.zeros_like(a[0]) for _ in a[1:]])
            if exponent < 0:
                a, exponent = reciprocal(a, spec, _OPS), -exponent
            result = [torch.ones_like(a[0])] + [torch.zeros_like(a[0]) for _ in a[1:]]
            while exponent:
                if exponent & 1:
                    result = multiply(result, a, spec, _OPS)
                exponent >>= 1
                if exponent:
                    a = multiply(a, a, spec, _OPS)
            return _Series(result)
        if isinstance(exponent, (int, float)):
            return _Series(compose(_scale(compose(a, "log"), exponent), "exp"))
        raise JetCompileError("tensor-valued powers require an explicit registered rule")
    if op in {"mm", "matmul", "bmm"}:
        if not isinstance(args[0], _Series):
            return _Series([target(args[0], r, **kwargs) for r in args[1].rows])
        if not isinstance(args[1], _Series):
            return _Series([target(r, args[1], **kwargs) for r in args[0].rows])
        a, b = (_rows(v, spec) for v in args[:2])
        return _Series([sum((target(a[i], b[j], **kwargs) for i, j in pairs)) for pairs in spec.pairs])
    if op in {"linear", "conv1d", "conv2d", "convolution"}:
        a, b = (_rows(v, spec) for v in args[:2])
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        rest = args[3:]
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "bias"}
        out = []
        if op == "linear" and isinstance(args[0], _Series) and not isinstance(args[1], _Series):
            # Linear preserves every leading dimension, so all coefficients
            # can share one matrix kernel without changing broadcast semantics.
            packed = F.linear(torch.stack(a), args[1], None)
            out = [packed[row] for row in range(spec.size)]
        else:
            for row, pairs in enumerate(spec.pairs):
                if not isinstance(args[0], _Series):
                    pairs = ((0, row),)
                elif not isinstance(args[1], _Series):
                    pairs = ((row, 0),)
                terms = [target(a[i], b[j], None, *rest, **clean_kwargs) for i, j in pairs]
                out.append(sum(terms))
        if bias is not None:
            bias_rows = _rows(bias, spec)
            if op == "linear":
                if isinstance(bias, _Series):
                    out = _add(out, bias_rows)
                else:
                    out[0] = out[0] + bias
            else:
                rank = 1 if op == "conv1d" else 2
                if isinstance(bias, _Series):
                    out = [v + z.reshape((1, -1) + (1,) * rank) for v, z in zip(out, bias_rows, strict=True)]
                else:
                    out[0] = out[0] + bias.reshape((1, -1) + (1,) * rank)
        return _Series(out)
    if op == "addmm":
        bias, a, b = args[:3]
        product = _rule(torch.ops.aten.mm.default, (a, b), {}, spec, backward, tracer, holder, kernels)
        return _Series(_add(_scale(_rows(bias, spec), kwargs.get("beta", 1)), _scale(product.rows, kwargs.get("alpha", 1))))
    if op in {"softmax", "_softmax"}:
        if op == "_softmax" and len(args) > 2 and args[2]:
            raise JetCompileError("half-to-float softmax is outside float32/float64 jet semantics")
        if kwargs.get("dtype") not in (None, torch.float32, torch.float64):
            raise JetCompileError("softmax dtype must be float32 or float64")
        dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
        a = list(_rows(args[0], spec))
        if kwargs.get("dtype") is not None:
            a = [r.to(dtype=kwargs["dtype"]) for r in a]
        # A constant-in-jet shift preserves the ratio and avoids exponent overflow.
        a[0] = a[0] - torch.amax(a[0], dim=dim, keepdim=True)
        exp = compose(a, "exp")
        total = [torch.sum(r, dim=dim, keepdim=True) for r in exp]
        return _Series(multiply(exp, reciprocal(total, spec, _OPS), spec, _OPS))
    if op == "softplus":
        beta = args[1] if len(args) > 1 else kwargs.get("beta", 1)
        threshold = args[2] if len(args) > 2 else kwargs.get("threshold", 20)
        a = _rows(args[0], spec)
        nonlinear = compose(_scale(a, beta), "softplus")
        return _Series([torch.where(a[0] * beta > threshold, x, y / beta) for x, y in zip(a, nonlinear, strict=True)])
    name = _ACTIVATIONS.get(op)
    if op == "gelu":
        approximation = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
        name = "gelu" if approximation == "none" else "gelu_approx"
    if name is not None:
        a = _rows(args[0], spec)
        if backward == "analytic":
            assert holder is not None and tracer is not None
            module_name = f"jet_activation_{len(holder._modules)}"
            holder.add_module(module_name, _AnalyticActivation(spec, name, kernels))
            analytic_proxy: Any = tracer.create_proxy("call_module", module_name, (torch.stack(a),), {})
            return _Series([analytic_proxy[i] for i in range(spec.size)])
        return _Series(compose(a, name))
    raise JetCompileError(f"no coefficient implementation for {target}")


class JetProgram(nn.Module):
    """A trainable FX program consuming packed jets for selected arguments.

    Its parameters are the original model's Parameter objects. Ordinary
    ``backward`` and optimizers update those same objects. ``graph_module`` is
    generated once and can be inspected or compiled with ``torch.compile``.
    """

    def __init__(self, graph_module: GraphModule, spec: JetSpec, argnums: tuple[int, ...], guards: list[Any], report: dict[str, Any], input_structure: Any) -> None:
        super().__init__()
        self.graph_module = graph_module
        self.spec, self.argnums = spec, argnums
        self._guards, self._report = guards, report
        self._input_structure = input_structure
        children = list(input_structure.children())
        self._flat_input_arity = len(children) if input_structure.type is tuple and all(child.is_leaf() for child in children) else None

    def forward(self, *args: Any) -> Any:
        leaves: Sequence[Any]
        if self._flat_input_arity is not None:
            if len(args) != self._flat_input_arity:
                raise JetCompileError("runtime argument tree differs from the exported example")
            leaves = args
        else:
            leaves, structure = _pytree.tree_flatten(args)
            if structure != self._input_structure:
                raise JetCompileError("runtime argument tree differs from the exported example")
        symbols: dict[str, Any] = {}
        for leaf, selected, dtype, shape in self._guards:
            value = leaves[leaf]
            if dtype is None:
                if type(value) is not type(shape) or value != shape:
                    raise JetCompileError("a specialized non-tensor export argument changed")
                continue
            if not isinstance(value, Tensor):
                raise JetCompileError("runtime tensor argument differs from the exported argument tree")
            if value.dtype != dtype:
                raise JetCompileError("runtime dtype differs from the exported example")
            offset = int(selected)
            if selected:
                torch._assert(value.shape[0] == self.spec.size, "jet coefficient count mismatch")
                torch._assert(value.numel() * value.element_size() <= self.spec.budget.max_tensor_bytes, "runtime jet exceeds tensor byte budget")
            torch._assert(value.ndim == len(shape) + offset, "runtime rank differs from export")
            for axis, (key, lower, upper) in enumerate(shape):
                length = value.shape[axis + offset]
                if isinstance(key, int):
                    torch._assert(length == key, "static export dimension changed")
                else:
                    if key in symbols:
                        torch._assert(length == symbols[key], "shared dynamic export dimension differs")
                    symbols[key] = length
                    if lower is not None:
                        torch._assert(length >= lower, "dynamic dimension below exported bound")
                    if upper is not None:
                        torch._assert(length <= upper, "dynamic dimension above exported bound")
        return self.graph_module(*leaves)

    def explain(self) -> dict[str, Any]:
        """Return the captured rule and resource plan without evaluating tensors."""
        return {**self._report, "parameter_ad": {
            "eager_higher_gradients": True,
            "compiled_forward": "first backward; use compiled HVP or hybrid for higher gradients",
            "compiled_parameter_hvp": True,
            "hybrid_higher_gradients": True,
        }}

    def hybrid(self, **compile_options: Any) -> Any:
        """Compile forward and recompute native backward for higher gradients.

        The returned module shares this program's parameters. Its outer wrapper
        stays outside fullgraph compilation; options apply to the pure forward.
        """
        from ._parameter_ad import HybridProgram

        return HybridProgram(self, **compile_options)

    def compile_parameter_hvp(self, loss_fn: Callable[..., Tensor], **compile_options: Any) -> Any:
        """Return ``hvp(tangents, *args, loss_args=(), **kwargs)``.

        ``loss_fn(outputs, *loss_args)`` returns a scalar. Tangents and returned
        products use this program's unique named parameter keys. No .grad field
        is changed, and inputs, directions and buffers are fixed for the Hessian.
        """
        from ._parameter_ad import ParameterHVP

        return ParameterHVP(self, loss_fn, **compile_options)


def _check_tensor_bound(value: Tensor, spec: JetSpec, ranges: dict[Any, Any]) -> int:
    shape = []
    for dim in value.shape:
        if isinstance(dim, int):
            shape.append(dim)
            continue
        expr = dim.node.expr
        substitutions = {}
        for symbol in expr.free_symbols:
            bound = ranges.get(symbol)
            if bound is None or str(bound.upper) in {"oo", "int_oo"}:
                raise JetCompileError("dynamic jet dimensions require finite upper bounds for resource checks")
            derivative = expr.diff(symbol)
            if derivative.is_nonnegative:
                substitutions[symbol] = int(bound.upper)
            elif derivative.is_nonpositive and str(bound.lower) not in {"-oo", "-int_oo"}:
                substitutions[symbol] = int(bound.lower)
            else:
                raise JetCompileError("cannot prove a resource bound for this dynamic shape expression")
        shape.append(int(expr.subs(substitutions)))
    return spec.check_tensor(shape, value.element_size())


def compile_jet(
    model: nn.Module, example_args: tuple[Any, ...], *, spec: JetSpec,
    argnums: tuple[int, ...] = (0,), dynamic_shapes: Any = None, backward: str = "native",
    kernel_lowering: str = "auto",
    _graph_factory: Callable[[], Graph] = Graph,
    _exported: _CapturedGraph | None = None,
) -> JetProgram:
    """Export a supported model and compile its Taylor algebra to ordinary FX.

    ``example_args`` contain ordinary primal inputs. At runtime, replace each
    selected positional tensor argument with a tensor of shape
    ``(spec.size, *original_shape)``. Input and output coefficients are Taylor
    normalized. Nonselected arguments and model parameters remain runtime data.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("compile_jet expects a torch.nn.Module")
    if backward not in {"native", "analytic"}:
        raise ValueError("backward must be 'native' or 'analytic'")
    validate_lowering(kernel_lowering)
    if not isinstance(example_args, tuple):
        raise TypeError("example_args must be a tuple of positional arguments")
    argnums = tuple(argnums)
    if len(set(argnums)) != len(argnums) or any(i < 0 or i >= len(example_args) for i in argnums):
        raise ValueError("argnums must contain distinct valid positional argument indices")
    selected_leaves: set[int] = set()
    offset = 0
    for i, arg in enumerate(example_args):
        flat, _ = _pytree.tree_flatten(arg)
        if i in argnums:
            if not isinstance(arg, Tensor):
                raise JetCompileError("selected positional arguments must be tensors")
            _validate_tensor(arg)
            spec.check_tensor(tuple(arg.shape), arg.element_size())
            selected_leaves.add(offset)
        offset += len(flat)
    if _exported is None:
        _reject_capture_state_aliases(model, example_args)
        try:
            exported = _CapturedGraph.from_export(torch.export.export(model, example_args, dynamic_shapes=dynamic_shapes, strict=True))
        except Exception as exc:
            raise JetCompileError(f"torch.export capture failed: {exc}") from exc
    else:
        exported = _exported
    for output in exported.graph_signature.output_specs:
        if output.kind.name != "USER_OUTPUT":
            raise JetCompileError(f"mutation/effect output {output.kind.name} is unsupported")
    # The operator frontend supplies a graph factory that checks its shared
    # planning budget before each emitted node. Ordinary jet callers retain
    # the existing graph construction behavior.
    graph = _graph_factory()
    tracer = GraphAppendingTracer(graph)
    holder = nn.Module()
    holder.add_module("model", model)
    constant_module = nn.Module()
    holder.add_module("constants", constant_module)
    kernels = KernelLowering(kernel_lowering, _OPS, tracer, holder, _graph_factory)
    env: dict[Node, Any] = {}
    guards: list[Any] = []
    rules: list[str] = []
    user_index = 0
    estimated_peak = 0
    input_specs = iter(exported.graph_signature.input_specs)
    for node in exported.graph_module.graph.nodes:
        if node.op == "placeholder":
            inp = next(input_specs)
            if inp.kind.name in {"PARAMETER", "BUFFER"}:
                env[node] = Proxy(graph.get_attr("model." + inp.target), tracer)
            elif inp.kind.name == "CONSTANT_TENSOR":
                attr = f"tensor_{len(constant_module._buffers)}"
                constant_module.register_buffer(attr, exported.constants[inp.target])
                env[node] = Proxy(graph.get_attr("constants." + attr), tracer)
            elif inp.kind.name == "USER_INPUT":
                proxy: Any = Proxy(graph.placeholder(str(node.target)), tracer)
                selected = user_index in selected_leaves
                env[node] = _Series([proxy[i] for i in range(spec.size)]) if selected else proxy
                value = node.meta.get("val")
                if isinstance(value, Tensor):
                    dims = []
                    for dim in value.shape:
                        if isinstance(dim, int):
                            dims.append((dim, None, None))
                        else:
                            if not dim.node.expr.is_Symbol:
                                raise JetCompileError("derived symbolic input dimensions are unsupported; use independent bounded dimensions")
                            key = str(dim)
                            bound = exported.range_constraints.get(dim.node.expr)
                            lower = int(bound.lower) if bound is not None and str(bound.lower) not in {"-oo", "oo", "-int_oo", "int_oo"} else None
                            upper = int(bound.upper) if bound is not None and str(bound.upper) not in {"-oo", "oo", "-int_oo", "int_oo"} else None
                            dims.append((key, lower, upper))
                    guards.append((user_index, selected, value.dtype, dims))
                else:
                    guards.append((user_index, False, None, getattr(inp.arg, "value", value)))
                user_index += 1
            else:
                raise JetCompileError(f"unsupported export input kind {inp.kind.name}")
        elif node.op == "call_function":
            args = _map(node.args, lambda n: env[n] if isinstance(n, Node) else n)
            kwargs = _map(node.kwargs, lambda n: env[n] if isinstance(n, Node) else n)
            try:
                if _tracked((args, kwargs)):
                    value = node.meta.get("val")
                    if isinstance(value, Tensor):
                        if value.dtype not in (torch.float32, torch.float64):
                            raise JetCompileError("a tracked operation produced a non-float32/float64 tensor")
                        estimated_peak = max(estimated_peak, _check_tensor_bound(value, spec, exported.range_constraints))
                env[node] = _rule(node.target, args, kwargs, spec, backward, tracer, holder, kernels)
            except Exception as exc:
                source = node.meta.get("stack_trace", "source unavailable")
                raise JetCompileError(f"jet compilation failed at {node.name} ({node.target}): {exc}\n{source}") from exc
            rules.append(str(node.target))
        elif node.op == "output":
            raw = _map(node.args[0], lambda n: env[n] if isinstance(n, Node) else n)
            leaves = []
            for original, result in zip(node.args[0], raw, strict=True):
                tensor_output = original.meta.get("val") if isinstance(original, Node) else None
                if isinstance(tensor_output, Tensor):
                    estimated_peak = max(estimated_peak, _check_tensor_bound(tensor_output, spec, exported.range_constraints))
                if isinstance(result, _Series):
                    leaves.append(torch.stack(result.rows))
                elif isinstance(result, Proxy) and isinstance(tensor_output, Tensor):
                    leaves.append(torch.stack(_rows(result, spec)))
                else:
                    leaves.append(result)
            _emit_output(graph, leaves, exported.call_spec.out_spec)
        else:
            raise JetCompileError(f"unsupported export node kind {node.op}")
    # Constant parameters carry no higher spatial coefficients. Remove unused
    # zero-producing nodes before constructing the runtime graph.
    graph.lint()
    gm = GraphModule(holder, graph, "OmnibiasJetGraph")
    gm.graph.eliminate_dead_code()
    gm.recompile()
    report = {"backend": "torch.export+fx", "backend_version": torch.__version__,
              "backward": backward, "spec": spec.to_config(), "algebra": spec.explain(),
              "coefficient_axis": 0, "rules": tuple(rules),
              "graph_nodes": sum(len(list(child.graph.nodes)) for child in gm.modules() if isinstance(child, GraphModule)),
              "kernel_lowering": kernels.explain(), "selected_arguments": argnums,
              "peak_single_jet_bytes_bound": estimated_peak,
              "budget_scope": "one packed tensor; excludes auxiliaries, backend workspaces and backward tape"}
    program = JetProgram(gm, spec, argnums, guards, report, _pytree.tree_flatten(example_args)[1])
    metadata_args = _metadata_examples(example_args)
    object.__setattr__(program, "_canonical_factory", lambda: compile_jet(
        model, metadata_args, spec=spec, argnums=argnums, dynamic_shapes=dynamic_shapes,
        backward="native", kernel_lowering="inline", _exported=exported))
    return program
