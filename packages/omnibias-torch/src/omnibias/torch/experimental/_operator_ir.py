# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Structural scalar expressions used only while planning spatial operators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from math import prod
from typing import Any


class NotStructured(Exception):
    """A valid tensor program needs the general Taylor fallback."""


@dataclass(eq=False)
class Expr:
    kind: str
    value: Any
    children: tuple[Expr, ...] = ()
    support: frozenset[int] = frozenset()
    affine: dict[int, Any] | None = None
    coordinate: int | None = None
    name: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    dominators: frozenset[Expr] = field(init=False)

    def __post_init__(self) -> None:
        active = [child.dominators for child in self.children if child.support]
        common = frozenset.intersection(*active) if active else frozenset()
        self.dominators = common | {self} if self.support else frozenset()


def constant(value: Any) -> Expr:
    return Expr("constant", value, affine={})


def variable(value: Any, coordinate: int) -> Expr:
    return Expr("variable", value, support=frozenset({coordinate}), affine={coordinate: 1}, coordinate=coordinate)


def add(left: Expr, right: Expr, value: Any = None) -> Expr:
    affine = None
    if left.affine is not None and right.affine is not None:
        affine = dict(left.affine)
        for axis, coefficient in right.affine.items():
            affine[axis] = affine.get(axis, 0) + coefficient
    return Expr("add", left.value + right.value if value is None else value, (left, right),
                left.support | right.support, affine=affine)


def multiply(left: Expr, right: Expr, value: Any = None) -> Expr:
    affine = None
    if not left.support and right.affine is not None:
        affine = {axis: left.value * c for axis, c in right.affine.items()}
    elif not right.support and left.affine is not None:
        affine = {axis: right.value * c for axis, c in left.affine.items()}
    return Expr("multiply", left.value * right.value if value is None else value, (left, right),
                left.support | right.support, affine=affine)


def unary(child: Expr, value: Any, name: str, options: dict[str, Any] | None = None) -> Expr:
    return Expr("unary", value, (child,), child.support, name=name, options=options or {})


@dataclass
class TensorExpr:
    value: Any
    shape: tuple[Any, ...]
    features: tuple[int, ...] | None = None
    expressions: tuple[Expr, ...] | None = None

    @property
    def tracked(self) -> bool:
        return self.expressions is not None


def coordinates(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    return list(product(*(range(n) for n in shape)))


def select(value: Any, index: tuple[int, ...]) -> Any:
    return value[(Ellipsis, *index)] if index else value


def at(value: TensorExpr | Any, index: tuple[int, ...]) -> Expr:
    if not isinstance(value, TensorExpr):
        return constant(value)
    if value.expressions is None:
        count = min(len(value.shape), len(index))
        key = tuple(0 if value.shape[-count + i] == 1 else axis for i, axis in enumerate(index[-count:])) if count else ()
        return constant(select(value.value, key))
    assert value.features is not None
    key = index[-len(value.features):] if value.features else ()
    key = tuple(0 if size == 1 else i for i, size in zip(key, value.features, strict=True))
    offset = 0
    for axis, size in zip(key, value.features, strict=True):
        offset = offset * size + axis
    return value.expressions[offset]


def feature_shape(shape: tuple[Any, ...], batch_ndim: int) -> tuple[int, ...]:
    features = shape[batch_ndim:]
    if any(not isinstance(n, int) or n < 1 for n in features):
        raise NotStructured("feature dimensions must be static and nonempty")
    return tuple(features)


def build_tensor_rule(op: str, args: tuple[Any, ...], kwargs: dict[str, Any], value: Any,
                      shape: tuple[Any, ...], batch_ndim: int, activation_name: Callable[[str, tuple[Any, ...], dict[str, Any]], tuple[str, dict[str, Any]] | None]) -> TensorExpr:
    """Conservative feature scalarization; batch axes are never scalarized."""
    features = feature_shape(shape, batch_ndim)
    indices = coordinates(features)
    first = args[0] if args else None
    result: list[Expr]
    active = activation_name(op, args, kwargs)
    if active is not None and isinstance(first, TensorExpr) and first.tracked:
        name, options = active
        result = [unary(at(first, index), select(value, index), name, options) for index in indices]
    elif op in {"add", "sub", "mul", "div", "true_divide"}:
        result = []
        for index in indices:
            a, b = at(args[0], index), at(args[1], index)
            if op in {"add", "sub"}:
                scale = kwargs.get("alpha", 1) * (-1 if op == "sub" else 1)
                b = multiply(b, constant(scale)) if scale != 1 else b
                result.append(add(a, b, select(value, index)))
            elif op == "mul":
                result.append(multiply(a, b, select(value, index)))
            else:
                inverse = unary(b, 1 / b.value, "reciprocal") if b.support else constant(1 / b.value)
                result.append(multiply(a, inverse, select(value, index)))
    elif op == "neg":
        result = [multiply(constant(-1), at(first, index), select(value, index)) for index in indices]
    elif op in {"square", "pow"}:
        exponent = 2 if op == "square" else args[1]
        if not isinstance(exponent, (int, float)):
            raise NotStructured("tensor exponent needs general rules")
        result = [unary(at(first, index), select(value, index), "power", {"exponent": exponent}) for index in indices]
    elif op == "linear":
        if not isinstance(first, TensorExpr) or first.features is None or len(first.features) != 1:
            raise NotStructured("linear source must have one feature axis")
        weight = args[1]
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        if not isinstance(weight, TensorExpr) or weight.tracked or len(weight.shape) != 2:
            raise NotStructured("spatially varying matrix")
        result = []
        for row in range(features[0]):
            term = constant(0)
            for col in range(first.features[0]):
                term = add(term, multiply(at(first, (col,)), constant(weight.value[row, col])))
            if bias is not None:
                term = add(term, at(bias, (row,)))
            term.value = value[..., row]
            result.append(term)
    elif op in {"sum", "mean"}:
        if not isinstance(first, TensorExpr) or first.features is None:
            raise NotStructured("reduction input is unstructured")
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        dims = tuple(range(len(first.shape))) if dim is None else ((dim,) if isinstance(dim, int) else tuple(dim))
        dims = tuple(d % len(first.shape) for d in dims)
        if any(d < batch_ndim for d in dims):
            raise NotStructured("a reduction mixes independent examples")
        reduced = tuple(d - batch_ndim for d in dims)
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        result = []
        for index in indices:
            term = constant(0)
            for original in coordinates(first.features):
                projected = tuple(0 if axis in reduced else item for axis, item in enumerate(original)) if keepdim else tuple(item for axis, item in enumerate(original) if axis not in reduced)
                if projected == index:
                    term = add(term, at(first, original))
            if op == "mean":
                term = multiply(term, constant(1 / prod(first.features[d] for d in reduced)))
            term.value = select(value, index)
            result.append(term)
    elif op in {"view", "reshape", "_unsafe_view", "flatten", "squeeze", "unsqueeze", "clone", "contiguous", "alias"}:
        if not isinstance(first, TensorExpr) or first.expressions is None or len(first.expressions) != prod(features):
            raise NotStructured("shape operation changes spatial feature count")
        result = list(first.expressions)
    elif op in {"select", "slice"}:
        if not isinstance(first, TensorExpr) or first.features is None:
            raise NotStructured("unstructured indexing")
        axis = int(args[1]) % len(first.shape)
        if axis < batch_ndim:
            raise NotStructured("indexing a batch axis")
        axis -= batch_ndim
        result = []
        for index in indices:
            if op == "select":
                key = (*index[:axis], int(args[2]) % first.features[axis], *index[axis:])
            else:
                start = int(args[2] or 0)
                step = int(args[4]) if len(args) > 4 else 1
                key = (*index[:axis], start + index[axis] * step, *index[axis + 1:])
            result.append(at(first, key))
    elif op in {"cat", "stack"}:
        sources = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        axis = int(dim) % len(shape) - batch_ndim
        if axis < 0:
            raise NotStructured("joining batch axes")
        result = []
        for index in indices:
            if op == "stack":
                result.append(at(sources[index[axis]], (*index[:axis], *index[axis + 1:])))
            else:
                offset = index[axis]
                for source in sources:
                    size = source.shape[batch_ndim + axis]
                    if offset < size:
                        result.append(at(source, (*index[:axis], offset, *index[axis + 1:])))
                        break
                    offset -= size
    else:
        raise NotStructured(f"{op} has no structural scalar rule")
    return TensorExpr(value, shape, features, tuple(result))
