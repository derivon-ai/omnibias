# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator-directed differentiation of ordinary Functional model inputs.

Structural proofs precede mixed-jet allocation. The fast paths are deliberately
conservative: a scalar read-once expression has disjoint coordinate supports at
every binary node, and a ridge chain has only one nonlinear layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from operator import index as integer_index
from types import SimpleNamespace
from typing import Any

from omnibias.core.apolar import apolar_algebra
from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import multiply
from omnibias.core.operator_polynomial import (
    coordinate_polynomial,
    operator_product,
    operator_product_plan,
    operator_readout,
    operator_sum,
    operator_unary,
    operator_unary_plan,
)
from omnibias.core.operator_quotient import (
    quotient_compose,
    quotient_estimate,
    quotient_multiply,
    quotient_readout,
    quotient_reciprocal,
)
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.polynomials import sigmoid_polynomial_coeffs, tanh_polynomial_coeffs
from omnibias.keras.experimental.model import (
    _CUSTOM_RULES,
    _LINEAR,
    _SMOOTH,
    _STRICT_RULES,
    _activation_name,
    _check_shape,
    _dtype,
    _evaluate,
    _nonlinear,
    _reject_batch_change,
    _Series,
    seed,
)
from omnibias.keras.experimental.precision import (
    PrecisionObjectiveMixin,
    StrictNumerics,
    strict_source_contract,
    validate_precision,
)

import keras

_BUILTIN_LAYERS = {
    keras.layers.Dense, keras.layers.Conv1D, keras.layers.Conv2D,
    keras.layers.Activation, keras.layers.Softmax, keras.layers.Add,
    keras.layers.Subtract, keras.layers.Multiply, keras.layers.Average,
    keras.layers.Reshape, keras.layers.Flatten, keras.layers.Permute,
    keras.layers.Concatenate,
}


@dataclass
class _Ref:
    index: int


@dataclass
class _Node:
    source: Any
    args: Any
    kwargs: Any
    shape: tuple[Any, ...]
    support: frozenset[int] = frozenset()
    rule: str | None = None
    coordinate: int | None = None
    activation: str | None = None


def _map(value: Any, function: Any) -> Any:
    if isinstance(value, _Ref):
        return function(value)
    if isinstance(value, tuple):
        return tuple(_map(v, function) for v in value)
    if isinstance(value, list):
        return [_map(v, function) for v in value]
    if isinstance(value, dict):
        return {k: _map(v, function) for k, v in value.items()}
    return value


def _refs(value: Any) -> list[int]:
    result: list[int] = []
    _map(value, lambda ref: result.append(ref.index))
    return result


def _capture(model: keras.Model, operator: OperatorSpec, precision: str = "source") -> tuple[list[_Node], Any]:
    """Build a source tape using public clone callbacks, never private nodes."""
    nodes = [_Node(None, (), {}, tuple(model.inputs[0].shape))]
    indices: dict[int, int] = {}

    def encode(value: Any) -> Any:
        if isinstance(value, keras.KerasTensor):
            return _Ref(indices[id(value)])
        if isinstance(value, tuple):
            return tuple(encode(v) for v in value)
        if isinstance(value, list):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, keras.Variable) and value.trainable:
            raise NotImplementedError("Standalone trainable operands require a registered layer rule.")
        return value

    def clone(original: keras.Model, inputs: Any) -> keras.Model:
        if isinstance(original, keras.Sequential):
            original = keras.Model(original.inputs[0], original.outputs[0])
        elif type(original).__name__ != "Functional":
            raise TypeError("operator_model supports built Functional/Sequential models.")

        def call(source: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(source, keras.Model):
                nested = clone(source, args[0])
                return nested.output
            custom = type(source) in _CUSTOM_RULES
            if precision == "strict":
                strict_source_contract(source, model.inputs[0].dtype)
                if custom and type(source) not in _STRICT_RULES:
                    raise NotImplementedError(f"{source.name}: strict precision requires an explicit strict_rule.")
            if not custom and type(source) not in _BUILTIN_LAYERS and not (
                isinstance(source, keras.Operation) and not isinstance(source, keras.layers.Layer)
                and type(source).__module__.startswith("keras.src.ops.")
            ):
                raise NotImplementedError(f"Custom/unsupported operation {type(source).__name__} needs a registered jet rule.")
            if not custom and type(source).__name__ == "GetItem":
                # Keras JSON round trips tuples as lists. Expanding ellipses
                # also proves which index consumes the original batch axis.
                key = args[1] if isinstance(args[1], (tuple, list)) else (args[1],)
                consumed = sum(item is not None and item is not Ellipsis for item in key)
                expanded = []
                for item in key:
                    expanded.extend([slice(None)] * (len(args[0].shape) - consumed)
                                    if item is Ellipsis else [item])
                args = (args[0], tuple(expanded), *args[2:])
            def series(value: Any) -> Any:
                if isinstance(value, keras.KerasTensor):
                    return _Series([value], value.dtype)
                if isinstance(value, (list, tuple)):
                    return type(value)(series(v) for v in value)
                if isinstance(value, dict):
                    return {key: series(v) for key, v in value.items()}
                return value
            if not custom:
                _reject_batch_change(source, series(args))
            name = "" if custom else type(source).__name__
            if isinstance(source, keras.layers.Dense) and (
                getattr(source, "lora_enabled", False) or getattr(source, "quantization_mode", None)
            ):
                raise NotImplementedError("Quantized/LoRA Dense requires a registered operator rule.")
            if name == "Pad" and tuple(source.pad_width[0]) != (0, 0):
                raise NotImplementedError("Padding must preserve the batch axis.")
            output = source(*args, **kwargs)
            if not isinstance(output, keras.KerasTensor):
                raise NotImplementedError("Multi-output primitive operations need an explicit operator rule.")
            if not output.shape or output.shape[0] != model.inputs[0].shape[0]:
                raise NotImplementedError("Operator graphs must preserve the batch axis.")
            if name in {"Add", "Subtract", "Multiply", "Divide", "TrueDivide", "Power", "Matmul", "BroadcastTo"}:
                tensors = [value for value in keras.tree.flatten(args) if isinstance(value, keras.KerasTensor)]
                if any(len(value.shape) != len(output.shape) for value in tensors):
                    raise NotImplementedError("Broadcasting must preserve the batch axis of every spatial operand.")
                if name == "Matmul" and isinstance(args[1], keras.KerasTensor) and len(args[1].shape) == 2:
                    raise NotImplementedError("Matrix multiplication may not contract a spatial batch axis.")
            if name == "Reshape":
                before, after = tuple(args[0].shape[1:]), tuple(output.shape[1:])
                if any(size is None for size in (*before, *after)) or math.prod(before) != math.prod(after):
                    raise NotImplementedError("Reshape must preserve each batch member's element count.")
            node = _Node(source, encode(args), encode(kwargs), tuple(output.shape))
            operator.planning_budget.check_graph(len(nodes), len(nodes) * int(model.inputs[0].shape[-1]))
            nodes.append(node)
            indices[id(output)] = len(nodes) - 1
            return output

        return keras.models.clone_model(original, input_tensors=inputs,
                                        clone_function=lambda layer: layer,
                                        call_function=call)

    inputs = keras.Input(batch_shape=model.inputs[0].shape, dtype=model.inputs[0].dtype)
    indices[id(inputs)] = 0
    graph = clone(model, inputs)
    return nodes, encode(graph.output)


def _name(source: Any) -> str | None:
    if type(source) in _CUSTOM_RULES:
        return None
    if isinstance(source, (keras.layers.Dense, keras.layers.Activation)):
        if source.activation is keras.activations.exponential:
            return "exp"
        return _activation_name(source.activation)
    name = type(source).__name__.lower()
    if name == "gelu" and getattr(source, "approximate", False):
        return "gelu_approx"
    return name if name in _SMOOTH else None


def _coordinate(args: Any, dimension: int) -> int | None:
    if len(args) != 2 or args[0] != _Ref(0):
        return None
    key = args[1]
    if not isinstance(key, tuple) or len(key) != 2 or key[0] != slice(None):
        return None
    if isinstance(key[1], int):
        index = key[1] + dimension if key[1] < 0 else key[1]
        return index if 0 <= index < dimension else None
    if isinstance(key[1], slice):
        indices = range(*key[1].indices(dimension))
        return indices[0] if len(indices) == 1 else None
    return None


def _read_once(nodes: list[_Node], output: Any, dimension: int) -> bool:
    if not isinstance(output, _Ref):
        return False
    nodes[0].support = frozenset(range(dimension))
    if dimension == 1:
        nodes[0].rule, nodes[0].coordinate = "coordinate", 0
    for node in nodes[1:]:
        if type(node.source) in _CUSTOM_RULES:
            return False
        refs = _refs((node.args, node.kwargs))
        name = type(node.source).__name__
        coordinate = _coordinate(node.args, dimension) if name == "GetItem" else None
        if coordinate is not None:
            node.rule, node.coordinate = "coordinate", coordinate
            node.support = frozenset((coordinate,))
            continue
        if not refs or any(nodes[index].rule is None for index in refs):
            return False
        # Scalar per example: (B,) and (B,1) are both accepted. Broadcasting
        # between them can create a BxB tensor, so binary shapes must agree.
        if node.shape[1:] not in {(), (1,)}:
            return False
        node.support = frozenset().union(*(nodes[index].support for index in refs))
        activation_name = _name(node.source)
        if len(refs) == 1 and activation_name and not isinstance(node.source, keras.layers.Dense):
            node.rule, node.activation = "unary", activation_name
        elif isinstance(node.source, keras.layers.Dense) and node.source.units == 1:
            node.rule, node.activation = "dense", activation_name
        elif name in {"Add", "Subtract", "Multiply"} and len(node.args) == 2:
            left, right = node.args
            left_support = nodes[left.index].support if isinstance(left, _Ref) else frozenset()
            right_support = nodes[right.index].support if isinstance(right, _Ref) else frozenset()
            if left_support & right_support:
                return False
            if isinstance(left, _Ref) and isinstance(right, _Ref):
                if nodes[left.index].shape != nodes[right.index].shape:
                    return False
            elif not isinstance(left if not isinstance(left, _Ref) else right, (int, float)):
                return False
            node.rule = name.lower()
        elif name == "Negative" and len(refs) == 1:
            node.rule = "negative"
        else:
            return False
    return nodes[output.index].rule is not None


def _ridge(nodes: list[_Node], output: Any) -> bool:
    if not isinstance(output, _Ref) or output.index != len(nodes) - 1:
        return False
    nonlinear = False
    for index, node in enumerate(nodes[1:], 1):
        if type(node.source) in _CUSTOM_RULES:
            return False
        if _refs((node.args, node.kwargs)) != [index - 1]:
            return False
        name = _name(node.source)
        if isinstance(node.source, keras.layers.Dense):
            if name != "linear":
                if nonlinear:
                    return False
                nonlinear = True
            node.rule, node.activation = "ridge_dense", name
        elif name and name != "linear":
            if nonlinear:
                return False
            nonlinear = True
            node.rule, node.activation = "ridge_unary", name
        elif isinstance(node.source, keras.layers.Activation) and name == "linear":
            node.rule = "ridge_identity"
        else:
            return False
    return nonlinear


def _scalar_cut(nodes: list[_Node], output: Any, dimension: int) -> tuple[int, tuple[int, ...]] | None:
    """Prove that every path into an outer scalar DAG passes one scalar node."""
    if not isinstance(output, _Ref):
        return None
    for cut in range(len(nodes) - 2, 0, -1):
        if nodes[cut].rule is None or nodes[cut].support != frozenset(range(dimension)):
            continue
        pending, outer, valid = [output.index], set(), True
        while pending:
            index = pending.pop()
            if index == cut or index in outer:
                continue
            if index < cut or nodes[index].shape != nodes[cut].shape:
                valid = False
                break
            outer.add(index)
            pending.extend(_refs((nodes[index].args, nodes[index].kwargs)))
        if valid:
            return cut, tuple(sorted(outer))
    return None


def _separable_cut(nodes: list[_Node], output: Any) -> tuple[int, tuple[int, ...]] | None:
    """Featurewise vector expressions followed by a feature-only sum/mean."""
    if not isinstance(output, _Ref):
        return None
    elementwise = {"Add", "Subtract", "Multiply", "Divide", "TrueDivide", "Negative", "Power", "Square"}
    for index, node in enumerate(nodes[1:], 1):
        if type(node.source) in _CUSTOM_RULES:
            return None
        name = type(node.source).__name__
        if name in {"Sum", "Mean"}:
            axis = node.source.axis
            if axis not in (1, -1, (1,), (-1,), [1], [-1]) or node.shape[1:] not in {(), (1,)}:
                return None
            pending, outer = [output.index], set()
            while pending:
                current = pending.pop()
                if current == index or current in outer:
                    continue
                if current < index or nodes[current].shape != node.shape:
                    return None
                outer.add(current)
                pending.extend(_refs((nodes[current].args, nodes[current].kwargs)))
            return index, tuple(sorted(outer))
        if node.shape != nodes[0].shape:
            return None
        if name not in elementwise and not (_name(node.source) and not isinstance(node.source, keras.layers.Dense)):
            return None
    return None


def _bottleneck(nodes: list[_Node], output: Any, dimension: int) -> int | None:
    if len(nodes) < 2 or type(nodes[1].source) is not keras.layers.Dense:
        return None
    dense = nodes[1].source
    if dense.units >= dimension or _refs(nodes[1].args) != [0]:
        return None
    if getattr(dense, "lora_enabled", False) or getattr(dense, "quantization_mode", None):
        return None
    pending, visited = _refs(output), set()
    while pending:
        index = pending.pop()
        if index == 1 or index in visited:
            continue
        if index == 0:
            return None
        visited.add(index)
        pending.extend(_refs((nodes[index].args, nodes[index].kwargs)))
    return int(dense.units)


def _homogeneous(dimension: int, degree: int) -> list[tuple[int, ...]]:
    if dimension == 1:
        return [(degree,)]
    return [(first, *tail) for first in range(degree + 1)
            for tail in _homogeneous(dimension - 1, degree - first)]


def _transport_steps(factors: list[list[tuple[tuple[int, ...], tuple[int, ...]]]],
                     dimension: int, operator: OperatorSpec) -> tuple[Any, ...]:
    indices = [(0,) * dimension]
    steps = []
    work = 0
    for factor in factors:
        destinations: dict[tuple[int, ...], int] = {}
        schedule = []
        for index, alpha in enumerate(indices):
            for beta, descriptor in factor:
                target = tuple(a + b for a, b in zip(alpha, beta, strict=True))
                destination = destinations.setdefault(target, len(destinations))
                schedule.append((index, destination, descriptor))
                work += 1
                operator.planning_budget.check_graph(0, work)
        indices = list(destinations)
        operator.budget.check_layout(len(indices), len(schedule))
        steps.append((len(indices), tuple(schedule)))
    return tuple(indices), tuple(steps)


def _tower(value: Any, order: int, name: str, method: str, spec: JetSpec, numerics: Any = None) -> list[Any]:
    ops = numerics or keras.ops
    if name == "linear":
        return [value] + ([ops.ones_like(value)] if order else []) + [ops.zeros_like(value)] * max(0, order - 1)
    if method == "polynomial" and name in {"tanh", "sigmoid", "softplus"}:
        base = ops.tanh(value) if name == "tanh" else ops.sigmoid(value)
        rows = [getattr(ops, name)(value)]
        for degree in range(1, order + 1):
            coefficients = (tanh_polynomial_coeffs(degree) if name == "tanh" else
                            sigmoid_polynomial_coeffs(degree - (name == "softplus")))
            row = ops.ones_like(base) * float(coefficients[-1])
            for coefficient in reversed(coefficients[:-1]):
                row = row * base + float(coefficient)
            row = row / float(math.factorial(degree))
            if not isinstance(ops, StrictNumerics) and name == "softplus" and keras.backend.backend() == "torch":
                row = ops.where(value > 20, ops.ones_like(value) if degree == 1 else ops.zeros_like(value), row)
            rows.append(row)
        return rows
    coefficients = [value] + ([ops.ones_like(value)] if order else [])
    coefficients += [ops.zeros_like(value)] * max(0, order - 1)
    return _nonlinear(coefficients, spec, name, numerics=ops)


def _source_args(node: _Node, values: list[Any]) -> tuple[Any, Any]:
    return (_map(node.args, lambda ref: values[ref.index]),
            _map(node.kwargs, lambda ref: values[ref.index]))


@keras.saving.register_keras_serializable(package="omnibias")
class _OperatorLayer(keras.layers.Layer):
    def __init__(self, source: keras.Model, operator: OperatorSpec,
                 activation_method: str = "recurrence", strategy: str = "auto",
                 precision: str = "source", **kwargs: Any) -> None:
        kwargs.setdefault("autocast", False)
        super().__init__(**kwargs)
        self.source = source
        self.operator = operator
        self.activation_method = activation_method
        self.strategy = strategy
        self.precision = validate_precision(precision)
        self.numerics = StrictNumerics(self.compute_dtype) if precision == "strict" else None
        self.primal_spec = JetSpec.directional(0, budget=operator.budget)
        self.dimension = int(source.inputs[0].shape[-1])
        operator.check_dimension(self.dimension)
        self.nodes, self.output_template = _capture(source, operator, precision)
        selected = "reference"
        self.cut: int | None = None
        self.outer_nodes: tuple[int, ...] = ()
        self.vector_cut: int | None = None
        self.latent_dimension: int | None = None
        if strategy == "auto":
            if operator.mode != "directional":
                if _read_once(self.nodes, self.output_template, self.dimension):
                    selected = "read_once"
                else:
                    cut = _scalar_cut(self.nodes, self.output_template, self.dimension)
                    if cut is not None:
                        self.cut, self.outer_nodes = cut
                        selected = "scalar_cut"
                    else:
                        cut = _separable_cut(self.nodes, self.output_template)
                        if cut is not None:
                            self.vector_cut, self.outer_nodes = cut
                            selected = "separable"
            if selected == "reference" and _ridge(self.nodes, self.output_template):
                selected = "ridge"
            if selected == "reference":
                self.latent_dimension = _bottleneck(self.nodes, self.output_template, self.dimension)
                if self.latent_dimension is not None:
                    count = math.comb(self.latent_dimension + operator.order, operator.order)
                    ambient = math.prod(order + 1 for order in operator.alpha) if operator.mode == "mixed" else None
                    if (operator.mode == "directional" or (count <= operator.budget.max_coefficients and
                                                          (ambient is None or count <= ambient))):
                        selected = "affine_bottleneck"
                    else:
                        self.latent_dimension = None
        self.selected = selected
        self.terms = (operator.resolve_terms(self.dimension)
                      if operator.mode != "directional" and not (selected in {"ridge", "affine_bottleneck"} and operator.mode == "laplacian") else ())
        self.tower_specs: dict[int, JetSpec] = {}
        self.unary_plans: dict[int, Any] = {}
        self.product_plans: dict[tuple[int, int], Any] = {}
        self.lifted = None
        self.transport: list[Any] = []
        self.algebra = None
        self.quotient_rejection: str | None = None
        if selected == "reference":
            self.spec = None
            if strategy == "auto" and operator.mode in {"polynomial", "laplacian"}:
                allowed = _LINEAR | {"Add", "Subtract", "Multiply", "Divide", "TrueDivide", "Negative", "Matmul", "Square", "Power", "Pad", "OnesLike", "ZerosLike", "FullLike"}
                eligible = all(type(node.source) not in _CUSTOM_RULES and (type(node.source).__name__ in allowed or _name(node.source)
                                or isinstance(node.source, (keras.layers.Conv1D, keras.layers.Conv2D)))
                               and (type(node.source).__name__ != "Power" or
                                    isinstance(node.args[1], int) and node.args[1] >= 0)
                               for node in self.nodes[1:])
                if eligible:
                    try:
                        candidate = apolar_algebra(dict(self.terms), max_monomials=min(512, operator.budget.max_coefficients),
                                                   max_basis=min(128, operator.budget.max_coefficients),
                                                   max_products=operator.budget.max_multiplication_terms,
                                                   max_order=operator.planning_budget.max_order,
                                                   max_coefficient_bits=operator.planning_budget.max_coefficient_bits)
                        quotient_estimate(candidate, "activation", budget=operator.budget)
                        traversal = sum(math.prod(order + 1 for order in alpha) for alpha, _ in self.terms)
                        operator.planning_budget.check_graph(len(self.nodes), traversal)
                        closure = {beta for alpha, _ in self.terms for beta in product(*(range(order + 1) for order in alpha))}
                        if candidate.size < len(closure):
                            self.algebra = candidate
                            selected = self.selected = "apolar"
                            self.tower_specs[operator.order] = JetSpec.directional(operator.order, budget=operator.budget)
                            self.quotient_layout = SimpleNamespace(size=candidate.size)
                    except ValueError as error:
                        self.quotient_rejection = str(error)
            if self.algebra is None:
                self.spec = operator.jet_spec(self.dimension)
                self.tower_specs[operator.order] = JetSpec.directional(operator.order, budget=operator.budget)
        elif selected == "affine_bottleneck":
            latent = self.latent_dimension
            self.tower_specs[operator.order] = JetSpec.directional(operator.order, budget=operator.budget)
            if operator.mode == "directional":
                self.spec = JetSpec.directional(operator.order, budget=operator.budget)
            else:
                operator.budget.check_layout(math.comb(latent + operator.order, operator.order), 0)
                self.spec = JetSpec.mixed(_homogeneous(latent, operator.order), budget=operator.budget)
                units = [tuple(int(i == j) for i in range(latent)) for j in range(latent)]
                if operator.mode == "laplacian":
                    factor = [(tuple(a + b for a, b in zip(units[i], units[j], strict=True)), (i, j))
                              for i in range(latent) for j in range(latent)]
                    indices, steps = _transport_steps([factor] * operator.power, latent, operator)
                    self.transport.append((1.0, tuple(self.spec.index(alpha) for alpha in indices), steps))
                else:
                    for alpha, coefficient in self.terms:
                        factors = [[(unit, (coordinate, latent_axis)) for latent_axis, unit in enumerate(units)]
                                   for coordinate, order in enumerate(alpha) for _ in range(order)]
                        indices, steps = _transport_steps(factors, latent, operator)
                        self.transport.append((float(coefficient), tuple(self.spec.index(alpha) for alpha in indices), steps))
        else:
            self.spec = None
            # A scalar derivative tower has order+1 entries, independent of
            # spatial dimension. This allocation follows structural selection.
            for order in range(operator.order + 1):
                self.tower_specs[order] = JetSpec.directional(order, budget=operator.budget)
                if selected in {"read_once", "scalar_cut", "separable"}:
                    self.unary_plans[order] = operator_unary_plan(order, budget=operator.budget)
            if selected in {"read_once", "scalar_cut"}:
                for alpha, _ in self.terms:
                    for node in self.nodes:
                        if node.rule == "multiply":
                            pair = tuple(sum(alpha[i] for i in self.nodes[arg.index].support)
                                         if isinstance(arg, _Ref) else 0 for arg in node.args)
                            self.product_plans[pair] = operator_product_plan(*pair, budget=operator.budget)
        schedule_work = self._schedule_work()
        operator.planning_budget.check_graph(len(self.nodes) - 1, schedule_work)
        self.report = {
            "strategy_requested": strategy, "selected_strategy": selected,
            "activation_method": activation_method, "operator": operator.to_config(),
            "precision": precision, "compute_dtype": self.compute_dtype,
            "numeric_adapter": f"{keras.backend.backend()}_native" if precision == "strict" else "keras_source",
            "matched_patterns": [] if selected == "reference" else [selected],
            "graph_nodes": len(self.nodes) - 1,
            "estimated_schedule_work": schedule_work,
            "schedule_work_scope": "Aggregate conservative estimate for owned coefficient schedules; excludes registered custom-rule internals and backend/compiler expansion inside tensor primitives.",
            "coefficient_count": self.algebra.size if self.algebra else self.spec.size if self.spec else operator.order + 1,
            "latent_dimension": self.latent_dimension,
            "fallback_reason": ("explicit reference strategy" if strategy == "reference" else
                                "no proven ridge or disjoint scalar expression") if selected == "reference" else None,
            "precision_notes": (["Strict primitives preserve declared precision and GELU variants; softplus uses its canonical smooth definition. External losses and reporting metrics keep their own precision."]
                                if precision == "strict" else ["Derivatives inherit the source backend's activation and matmul precision; float64 tensor dtype does not guarantee float64 intermediate arithmetic."]),
            "activation_method_note": "Polynomial selection applies to tanh, sigmoid and softplus; other smooth primitives retain their source-aware recurrence.",
            "apolar_rejection": self.quotient_rejection,
            "budget_note": "Symbolic dimensions require a bounded caller batch under dynamic compilation.",
        }

    def build(self, input_shape: Any) -> None:
        # The supplied source is already built; all engine children reuse its
        # existing variables. There are no delayed weights in this wrapper.
        self.built = True

    def _activation_tower(self, value: Any, order: int, name: str) -> list[Any]:
        spec = self.tower_specs[order]
        _check_shape(spec, value.shape, value.dtype)
        return _tower(value, order, name, self.activation_method, spec, self.numerics)

    def _primal(self, node: _Node, values: list[Any]) -> Any:
        if self.numerics is None:
            args, kwargs = _source_args(node, values)
            return node.source(*args, **kwargs)
        def decode(ref: _Ref) -> _Series:
            return _Series([values[ref.index]], self.compute_dtype)
        args, kwargs = _map(node.args, decode), _map(node.kwargs, decode)
        name = _name(node.source) or "linear"
        value = _evaluate(node.source, args, kwargs, self.primal_spec,
                          activation_name=name, numerics=self.numerics)[0]
        return self.numerics.check(value)

    def _schedule_work(self) -> int:
        """Conservative scalar coefficient schedule cost, excluding opaque ops."""
        def series(node: _Node, size: int, pairs: int, order: int) -> int:
            name = type(node.source).__name__
            if type(node.source) in _CUSTOM_RULES:
                return 0  # The registered rule owns its internal resource contract.
            if _name(node.source) not in {None, "linear"} or name == "Softmax":
                return 128 * pairs + 8 * size
            if name in {"Power", "Divide", "TrueDivide"}:
                return 128 * pairs + 8 * size
            return 8 * pairs if name in {"Multiply", "Matmul"} else 8 * size

        if self.selected == "apolar":
            work = quotient_estimate(self.algebra, "activation", budget=self.operator.budget)["multiplication_terms"]
            return 4 * work * max(1, len(self.nodes) - 1)
        if self.selected in {"reference", "affine_bottleneck"}:
            pairs = sum(len(row) for row in self.spec.pairs)
            work = sum(series(node, self.spec.size, pairs, self.spec.order) for node in self.nodes[1:])
            work += sum(len(schedule) for _, _, steps in self.transport for _, schedule in steps)
            return work
        order = self.operator.order
        if self.selected == "ridge":
            pairs = (order + 1) * (order + 2) // 2
            return sum(series(node, order + 1, pairs, order) for node in self.nodes[1:]) + order * max(1, len(self.terms))
        if self.selected == "separable":
            work = 0
            for alpha, _ in self.terms:
                local = max(alpha)
                pairs = (local + 1) * (local + 2) // 2
                work += sum(series(node, local + 1, pairs, local) for node in self.nodes[1:self.vector_cut])
                work += 8 * self.dimension * ((order + 1) * (local + 1) + (local + 1)**3)
        else:
            work = 0
            for alpha, _ in self.terms:
                stop = self.cut + 1 if self.cut is not None else len(self.nodes)
                for node in self.nodes[:stop]:
                    local = sum(alpha[i] for i in node.support)
                    if node.rule in {"unary", "dense"}:
                        work += 128 * (local + 1) * (local + 2) // 2 + 8 * (local + 1)**3
                    else:
                        work += 8 * (local + 1)**2
        work += sum(series(self.nodes[index], order + 1, (order + 1) * (order + 2) // 2, order)
                    for index in self.outer_nodes)
        return work

    def _check_slots(self, slots: int, value: Any) -> None:
        dimensions = []
        for size in value.shape:
            try:
                dimensions.append(integer_index(size))
            except TypeError:
                dimensions.append(1)
        needed = slots * math.prod(dimensions) * (8 if _dtype(value.dtype) == "float64" else 4)
        if needed > self.operator.budget.max_tensor_bytes:
            raise ValueError(f"Operator state needs {needed} bytes; budget is {self.operator.budget.max_tensor_bytes}.")

    def _series_activation(self, rows: list[Any], spec: JetSpec, name: str) -> list[Any]:
        ops = self.numerics or keras.ops
        if self.activation_method != "polynomial" or name not in {"tanh", "sigmoid", "softplus"}:
            return _nonlinear(rows, spec, name, numerics=ops)
        tower = self._activation_tower(rows[0], spec.order, name)
        zero = ops.zeros_like(rows[0])
        delta = [zero, *rows[1:]]
        powers = [ops.ones_like(rows[0])] + [zero] * (spec.size - 1)
        result = [tower[0]] + [zero] * (spec.size - 1)
        for degree in range(1, spec.order + 1):
            powers = multiply(powers, delta, spec, ops)
            result = [value + tower[degree] * power for value, power in zip(result, powers, strict=True)]
        return result

    def _propagate_towers(self, initial: dict[int, list[Any]], indices: Any,
                          values: list[Any], spec: JetSpec) -> dict[int, list[Any]]:
        ops = self.numerics or keras.ops
        towers = dict(initial)
        for index in indices:
            node = self.nodes[index]
            def decode(ref: _Ref, node: _Node = node) -> _Series:
                dtype = _dtype(values[ref.index].dtype)
                coefficients = towers[ref.index]
                if isinstance(node.source, keras.layers.Layer) and node.source.autocast:
                    dtype = node.source.compute_dtype
                    coefficients = [ops.cast(value, dtype) for value in coefficients]
                return _Series(coefficients, dtype)
            args, kwargs = _map(node.args, decode), _map(node.kwargs, decode)
            name = _name(node.source)
            if type(node.source) in {keras.layers.Dense, keras.layers.Conv1D, keras.layers.Conv2D}:
                coefficients = _evaluate(node.source, args, kwargs, spec, "native", "linear", self.numerics)
                coefficients = self._series_activation(coefficients, spec, name or _activation_name(node.source.activation))
            elif name:
                coefficients = self._series_activation(args[0].values, spec, name)
            else:
                coefficients = _evaluate(node.source, args, kwargs, spec, numerics=self.numerics)
            coefficients[0] = values[index]
            _check_shape(spec, values[index].shape, values[index].dtype)
            towers[index] = coefficients
        return towers

    def _outer_tower(self, cut: int, values: list[Any]) -> list[Any]:
        ops = self.numerics or keras.ops
        spec = self.tower_specs[self.operator.order]
        value = values[cut]
        unit = ([value, ops.ones_like(value)] + [ops.zeros_like(value)] * max(0, spec.order - 1))[:spec.size]
        towers = self._propagate_towers({cut: unit}, self.outer_nodes, values, spec)
        return towers[self.output_template.index]

    def _separable(self, points: Any) -> Any:
        ops = self.numerics or keras.ops
        values = [points]
        for node in self.nodes[1:]:
            values.append(self._primal(node, values))
        cut = self.vector_cut
        outer = self._outer_tower(cut, values)
        result = ops.zeros_like(values[self.output_template.index])
        argument = _refs(self.nodes[cut].args)[0]
        keepdims = len(values[cut].shape) == 2
        scale = 1 / self.dimension if type(self.nodes[cut].source).__name__ == "Mean" else 1
        for alpha, coefficient in self.terms:
            spec = self.tower_specs[max(alpha)]
            _check_shape(spec, points.shape, points.dtype)
            unit = ([points, ops.ones_like(points)] + [ops.zeros_like(points)] * max(0, spec.order - 1))[:spec.size]
            prefix = self._propagate_towers({0: unit}, range(1, cut), values, spec)[argument]
            polynomial = [ops.ones_like(values[cut])]
            for coordinate, order in enumerate(alpha):
                local = [(entry[:, coordinate:coordinate + 1] if keepdims else entry[:, coordinate]) * scale
                         for entry in prefix[:order + 1]]
                coordinate_rule = coordinate_polynomial(order, one=ops.ones_like(local[0]), budget=self.operator.budget)
                contribution = operator_unary(coordinate_rule, local, budget=self.operator.budget,
                                              plan=self.unary_plans[order])
                polynomial = operator_sum(polynomial, contribution, budget=self.operator.budget)
            result = result + float(coefficient) * operator_readout(polynomial, outer, budget=self.operator.budget)
        return result

    def _affine_bottleneck(self, points: Any, direction: Any) -> Any:
        ops = self.numerics or keras.ops
        dense = self.nodes[1].source
        argument = ops.matmul(ops.cast(points, dense.compute_dtype), dense.kernel)
        if dense.use_bias:
            argument = argument + dense.bias
        spec = self.spec
        _check_shape(spec, argument.shape, argument.dtype)
        if direction is not None:
            tangent = ops.matmul(ops.cast(direction, dense.compute_dtype), dense.kernel)
            coefficients = [argument] + ([tangent] if spec.order else []) + [ops.zeros_like(argument)] * max(0, spec.order - 1)
        else:
            basis = ops.eye(self.latent_dimension, dtype=argument.dtype)
            coefficients = [argument] + [ops.ones_like(argument) * basis[alpha.index(1)]
                            if sum(alpha) == 1 else ops.zeros_like(argument) for alpha in spec.indices[1:]]
        coefficients = self._series_activation(coefficients, spec, _name(dense))
        values = [points]
        for node in self.nodes[1:]:
            values.append(self._primal(node, values))
        coefficients[0] = values[1]
        towers = self._propagate_towers({1: coefficients}, range(2, len(self.nodes)), values, spec)
        gram = ops.matmul(ops.transpose(dense.kernel), dense.kernel) if self.operator.mode == "laplacian" else None

        def readout(ref: _Ref) -> Any:
            rows = towers[ref.index]
            if direction is not None:
                return rows[-1] * float(math.factorial(spec.order)) + rows[0] * 0
            result = rows[0] * 0
            for scale, indices, steps in self.transport:
                weights = [ops.ones_like(dense.kernel[0, 0])]
                for size, schedule in steps:
                    next_weights = [ops.zeros_like(weights[0]) for _ in range(size)]
                    for source, target, descriptor in schedule:
                        weight = gram[descriptor] if gram is not None else dense.kernel[descriptor]
                        next_weights[target] = next_weights[target] + weights[source] * weight
                    weights = next_weights
                for index, weight in zip(indices, weights, strict=True):
                    result = result + scale * float(spec.factorials[index]) * rows[index] * weight
            return result

        return _map(self.output_template, readout)

    def _apolar(self, points: Any) -> Any:
        ops = self.numerics or keras.ops
        algebra, budget = self.algebra, self.operator.budget
        self._check_slots(algebra.size, points)
        generators = [[float(row[index]) for row in algebra.generators] for index in range(1, algebra.size)]
        initial = [points] + [ops.ones_like(points) * ops.convert_to_tensor(row, dtype=points.dtype) for row in generators]
        towers, values = {0: initial}, [points]

        def lift_constant(value: Any, dtype: str) -> list[Any]:
            if isinstance(value, _Series):
                return value.values
            constant = (ops.convert_to_tensor(value, dtype=dtype) if self.precision == "strict"
                        else ops.cast(ops.convert_to_tensor(value), dtype))
            return [constant] + [ops.zeros_like(constant)] * (algebra.size - 1)

        def compose(rows: list[Any], name: str) -> list[Any]:
            if name == "linear":
                return rows
            tower = self._activation_tower(rows[0], algebra.order, name)
            return quotient_compose(rows, tower, algebra, ops, budget=budget)

        for index, node in enumerate(self.nodes[1:], 1):
            primal = self._primal(node, values)
            values.append(primal)
            self._check_slots(algebra.size, primal)
            def decode(ref: _Ref, node: _Node = node) -> _Series:
                rows = towers[ref.index]
                dtype = _dtype(values[ref.index].dtype)
                if isinstance(node.source, keras.layers.Layer) and node.source.autocast:
                    dtype = node.source.compute_dtype
                    rows = [ops.cast(row, dtype) for row in rows]
                return _Series(rows, dtype)
            args, kwargs = _map(node.args, decode), _map(node.kwargs, decode)
            source, name = node.source, type(node.source).__name__
            activation_name = _name(source)
            if isinstance(source, (keras.layers.Dense, keras.layers.Conv1D, keras.layers.Conv2D)):
                rows = _evaluate(source, args, kwargs, self.quotient_layout, "native", "linear", self.numerics)
                rows = compose(rows, activation_name or _activation_name(source.activation))
            elif activation_name:
                rows = compose(args[0].values, activation_name)
            elif name in {"Add", "Subtract", "Multiply", "Divide", "TrueDivide", "Matmul"}:
                left, right = (lift_constant(value, primal.dtype) for value in args[:2])
                if name == "Add":
                    rows = [ops.add(a, b) for a, b in zip(left, right, strict=True)]
                elif name == "Subtract":
                    rows = [ops.subtract(a, b) for a, b in zip(left, right, strict=True)]
                else:
                    if name in {"Divide", "TrueDivide"}:
                        right = quotient_reciprocal(right, algebra, ops, budget=budget)
                    options = {"product": ops.matmul if name == "Matmul" else ops.multiply}
                    rows = quotient_multiply(left, right, algebra, ops=ops, budget=budget, **options)
            elif name == "Negative":
                rows = [-value for value in args[0].values]
            elif name in {"Square", "Power"}:
                exponent = 2 if name == "Square" else args[1]
                if not isinstance(exponent, int) or exponent < 0:
                    raise NotImplementedError("Apolar powers require a nonnegative integer exponent.")
                rows = [ops.ones_like(args[0].values[0])] + [ops.zeros_like(args[0].values[0])] * (algebra.size - 1)
                factor = args[0].values
                while exponent:
                    if exponent & 1:
                        rows = quotient_multiply(rows, factor, algebra, product=ops.multiply, ops=ops, budget=budget)
                    exponent //= 2
                    if exponent:
                        factor = quotient_multiply(factor, factor, algebra, product=ops.multiply, ops=ops, budget=budget)
            else:
                rows = _evaluate(source, args, kwargs, self.quotient_layout, numerics=self.numerics)
            rows[0] = primal
            towers[index] = rows
        return _map(self.output_template, lambda ref: quotient_readout(towers[ref.index], algebra, budget=budget))

    def _read_once(self, points: Any) -> Any:
        ops = self.numerics or keras.ops
        values = [points]
        for node in self.nodes[1:]:
            values.append(self._primal(node, values))
        output = values[self.output_template.index]
        result = ops.zeros_like(output)
        outer_tower = None
        if self.cut is not None:
            outer_tower = self._outer_tower(self.cut, values)
        for alpha, coefficient in self.terms:
            if any(power and coordinate not in self.nodes[self.output_template.index].support
                   for coordinate, power in enumerate(alpha)) and self.cut is None:
                continue
            polynomials: list[list[Any]] = []
            stop = self.cut + 1 if self.cut is not None else len(self.nodes)
            for index, node in enumerate(self.nodes[:stop]):
                value = values[index]
                order = sum(alpha[coordinate] for coordinate in node.support)
                slots = ((order + 1) * (order + 2) // 2 + order + 1
                         if node.rule in {"unary", "dense"} else 2 * order + 3)
                self._check_slots(slots, value)
                one = ops.ones_like(value)
                if node.rule == "coordinate":
                    poly = coordinate_polynomial(alpha[node.coordinate], one=one, budget=self.operator.budget)
                elif index == 0:
                    poly = [one]
                elif node.rule in {"add", "subtract", "multiply"}:
                    left, right = node.args
                    left_value = values[left.index] if isinstance(left, _Ref) else left
                    right_value = values[right.index] if isinstance(right, _Ref) else right
                    left_poly = polynomials[left.index] if isinstance(left, _Ref) else [one]
                    right_poly = polynomials[right.index] if isinstance(right, _Ref) else [one]
                    if node.rule == "subtract":
                        right_poly = [entry * (-1) ** degree for degree, entry in enumerate(right_poly)]
                    if node.rule in {"add", "subtract"}:
                        poly = operator_sum(left_poly, right_poly, budget=self.operator.budget)
                    else:
                        poly = operator_product(left_poly, right_poly, left_value, right_value,
                                                budget=self.operator.budget,
                                                plan=self.product_plans[(len(left_poly) - 1, len(right_poly) - 1)])
                else:
                    parent = _refs(node.args)[0]
                    incoming = polynomials[parent]
                    if node.rule == "negative":
                        poly = [entry * (-1) ** degree for degree, entry in enumerate(incoming)]
                    elif node.rule == "dense":
                        kernel = node.source.kernel[0, 0]
                        incoming = [entry * kernel**degree for degree, entry in enumerate(incoming)]
                        argument = ops.matmul(ops.cast(values[parent], node.source.compute_dtype), node.source.kernel)
                        if node.source.use_bias:
                            argument = argument + node.source.bias
                        tower = self._activation_tower(argument, order, node.activation)
                        poly = operator_unary(incoming, tower, budget=self.operator.budget, plan=self.unary_plans[order])
                    else:
                        argument = values[parent]
                        if isinstance(node.source, keras.layers.Layer) and node.source.autocast:
                            argument = ops.cast(argument, node.source.compute_dtype)
                        tower = self._activation_tower(argument, order, node.activation)
                        poly = operator_unary(incoming, tower, budget=self.operator.budget, plan=self.unary_plans[order])
                polynomials.append(poly)
            poly = polynomials[self.cut if self.cut is not None else self.output_template.index]
            identity = outer_tower
            if identity is None:
                identity = [output] + ([ops.ones_like(output)] if len(poly) > 1 else [])
                identity += [ops.zeros_like(output)] * max(0, len(poly) - 2)
            result = result + float(coefficient) * operator_readout(poly, identity, budget=self.operator.budget)
        return result

    def _ridge(self, points: Any, direction: Any) -> Any:
        ops = self.numerics or keras.ops
        value = points
        slopes = direction if self.operator.mode == "directional" else None
        result = None
        constant = (1.0 if self.operator.order == 0 else 0.0) if self.operator.mode == "directional" else sum(
            float(coefficient) for alpha, coefficient in self.terms if not any(alpha))
        if self.operator.mode == "laplacian" and self.operator.power == 0:
            constant = 1.0
        for node in self.nodes[1:]:
            source = node.source
            argument = ops.cast(value, source.compute_dtype) if isinstance(source, keras.layers.Layer) and source.autocast else value
            if isinstance(source, keras.layers.Dense):
                argument = ops.matmul(argument, source.kernel)
                if source.use_bias:
                    argument = argument + source.bias
                if result is None:
                    slopes = source.kernel if slopes is None else ops.matmul(slopes, source.kernel)
                else:
                    result = ops.matmul(result, source.kernel)
                    if source.use_bias:
                        result = result + constant * source.bias
            if node.activation and node.activation != "linear":
                tower = self._activation_tower(argument, self.operator.order, node.activation)
                if self.operator.mode == "directional":
                    order = self.operator.order
                    result = tower[order] * float(math.factorial(order)) * slopes**order
                elif self.operator.mode == "laplacian":
                    order = self.operator.order
                    contraction = (ops.ones_like(argument) if slopes is None else
                                   ops.sum(slopes * slopes, axis=0)**self.operator.power)
                    result = tower[order] * float(math.factorial(order)) * contraction
                else:
                    result = ops.zeros_like(tower[0])
                    for alpha, coefficient in self.terms:
                        contraction = ops.ones_like(argument) if slopes is None else ops.ones_like(slopes[0])
                        for coordinate, power in enumerate(alpha):
                            if power:
                                row = (ops.cast(ops.arange(self.dimension) == coordinate, argument.dtype)
                                       if slopes is None else slopes[coordinate])
                                contraction = contraction * row**power
                        order = sum(alpha)
                        result = result + float(coefficient) * float(math.factorial(order)) * tower[order] * contraction
            value = (tower[0] if node.activation and node.activation != "linear" else argument) if self.precision == "strict" else source(value)
        return result

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        ops = self.numerics or keras.ops
        points, direction = inputs if self.operator.mode == "directional" else (inputs, None)
        _dtype(points.dtype)
        if direction is not None:
            if tuple(direction.shape) != tuple(points.shape):
                raise ValueError("Directions must have exactly the input point shape.")
            if _dtype(direction.dtype) != _dtype(points.dtype):
                raise TypeError("Directions must have exactly the input point dtype.")
        if self.selected in {"read_once", "scalar_cut"}:
            return self._read_once(points)
        if self.selected == "separable":
            return self._separable(points)
        if self.selected == "affine_bottleneck":
            return self._affine_bottleneck(points, direction)
        if self.selected == "apolar":
            return self._apolar(points)
        if self.selected == "ridge":
            return self._ridge(points, direction)
        if direction is None:
            basis = ops.eye(self.dimension, dtype=points.dtype)
            directions = [ops.ones_like(points) * basis[index] for index in range(self.dimension)]
        else:
            directions = direction
        packed = seed(points, directions, spec=self.spec)
        values = [points]
        for node in self.nodes[1:]:
            values.append(self._primal(node, values))
        towers = self._propagate_towers({0: [packed[:, index] for index in range(self.spec.size)]},
                                       range(1, len(self.nodes)), values, self.spec)

        def readout(ref: _Ref) -> Any:
            output = towers[ref.index]
            if direction is not None:
                return output[-1] * float(math.factorial(self.operator.order)) + output[0] * 0
            result = output[0] * 0
            for alpha, coefficient in self.terms:
                index = self.spec.index(alpha)
                result = result + float(coefficient) * float(self.spec.factorials[index]) * output[index]
            return result

        return _map(self.output_template, readout)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(source=keras.saving.serialize_keras_object(self.source),
                      operator=self.operator.to_config(), activation_method=self.activation_method,
                      strategy=self.strategy, precision=self.precision)
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _OperatorLayer:
        config = dict(config)
        config["source"] = keras.saving.deserialize_keras_object(config["source"])
        config["operator"] = OperatorSpec.from_config(config["operator"])
        return cls(**config)


@keras.saving.register_keras_serializable(package="omnibias")
class OperatorModel(PrecisionObjectiveMixin, keras.Model):
    """Normal Keras training/serialization around an operator-directed graph."""

    def __init__(self, graph: keras.Model, operator: OperatorSpec, report: dict[str, Any],
                 precision: str = "source", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.operator = operator
        self._report = report
        self.precision = validate_precision(precision)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        if self.operator.mode == "directional":
            if _dtype(inputs[0].dtype) != _dtype(inputs[1].dtype):
                raise TypeError("Directions must have exactly the input point dtype.")
        return self.graph(inputs, training=training)

    def explain(self) -> dict[str, Any]:
        return dict(self._report)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(graph=keras.saving.serialize_keras_object(self.graph),
                      operator=self.operator.to_config(), report=self._report, precision=self.precision)
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OperatorModel:
        config = dict(config)
        config["graph"] = keras.saving.deserialize_keras_object(config["graph"])
        config["operator"] = OperatorSpec.from_config(config["operator"])
        return cls(**config)


def operator_model(model: keras.Model, *, operator: OperatorSpec,
                   activation_method: str = "recurrence", strategy: str = "auto",
                   precision: str = "source") -> OperatorModel:
    """Apply a raw spatial derivative operator to an ordinary model.

The source must have one input of shape ``(batch, dimension)``. Coordinate and
Laplacian operators accept ordinary points. Directional operators accept
``[points, directions]`` with equal shape and dtype. Original variables are
reused; ``compile``, optimizers and ``.keras`` round trips remain available.
``precision='strict'`` preserves declared compute precision for supported
mathematics. The default ``'source'`` retains the source backend's numerical
behavior. Use PrecisionMeanSquaredError for an explicitly typed objective.
"""
    validate_precision(precision)
    if not isinstance(operator, OperatorSpec):
        raise TypeError("operator must be an OperatorSpec.")
    if strategy not in {"auto", "reference"}:
        raise ValueError("strategy must be 'auto' or 'reference'.")
    if activation_method not in {"recurrence", "polynomial"}:
        raise ValueError("activation_method must be 'recurrence' or 'polynomial'.")
    for _, coefficient in operator.terms:
        try:
            converted = float(coefficient)
        except OverflowError as error:
            raise ValueError("Operator coefficient exceeds the floating-point range.") from error
        if not math.isfinite(converted) or (coefficient and not converted):
            raise ValueError("Operator coefficient overflows or underflows the floating-point range.")
    if not isinstance(model, keras.Model) or not getattr(model, "inputs", None):
        raise TypeError("operator_model requires a built Functional/Sequential model.")
    if len(model.inputs) != 1 or len(model.inputs[0].shape) != 2 or model.inputs[0].shape[-1] is None:
        raise ValueError("operator_model requires one (batch, dimension) input with static dimension.")
    dtype = _dtype(model.inputs[0].dtype)
    layer = _OperatorLayer(model, operator, activation_method, strategy, precision=precision, dtype=dtype,
                           name=f"{model.name}_operator_rule")
    points = keras.Input(batch_shape=model.inputs[0].shape, dtype=dtype, name="points")
    inputs = [points, keras.Input(batch_shape=points.shape, dtype=dtype, name="directions")] if operator.mode == "directional" else points
    graph = keras.Model(inputs, layer(inputs), name=f"{model.name}_operator_graph")
    return OperatorModel(graph, operator, layer.report, precision=precision, dtype=dtype, autocast=False,
                         name=f"{model.name}_operator")


__all__ = ["OperatorModel", "OperatorSpec", "operator_model"]
