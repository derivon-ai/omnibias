# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Accuracy-first activation providers and compressed scalar-region lowering."""

from __future__ import annotations

from math import factorial
from typing import Any

from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import activation, multiply, reciprocal
from omnibias.core.operator_polynomial import (
    coordinate_polynomial,
    operator_product,
    operator_unary,
)
from omnibias.core.operator_spec import OperatorSpec
from omnibias.torch.activations import get_activation

import torch

from ._kernel_lowering import KernelLowering
from ._operator_ir import Expr, NotStructured
from .compiler import _OPS


def activation_name(op: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if op in {"tanh", "sigmoid", "sin", "cos", "exp", "log", "sqrt", "reciprocal", "erf", "erfc", "silu"}:
        return op, {}
    if op == "gelu":
        approximation = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
        return ("gelu" if approximation == "none" else "gelu_approx"), {}
    if op == "softplus":
        return "softplus", {"beta": args[1] if len(args) > 1 else kwargs.get("beta", 1),
                            "threshold": args[2] if len(args) > 2 else kwargs.get("threshold", 20)}
    return None


class Lowering:
    def __init__(self, operator: OperatorSpec, method: str, kernels: KernelLowering | None = None) -> None:
        self.operator, self.method = operator, method
        self.kernels = kernels
        self.layouts: dict[int, JetSpec] = {}
        self.packets: dict[tuple[Expr, tuple[int, ...]], list[Any]] = {}
        self.packet_supports: dict[tuple[Expr, tuple[int, ...]], frozenset[int]] = {}
        self.jets: dict[tuple[Expr, Expr, int], list[Any]] = {}
        self.towers: dict[tuple[Expr, int], list[Any]] = {}
        self.patterns: set[str] = set()
        self.max_coefficients = 1
        self.tensor_bytes_per_coefficient = 0

    def check_state(self, size: int) -> None:
        self.operator.budget.check_layout(size, 0)
        estimated = size * self.tensor_bytes_per_coefficient
        if estimated > self.operator.budget.max_tensor_bytes:
            raise ValueError(f"operator state bound {estimated} bytes exceeds budget {self.operator.budget.max_tensor_bytes}; reduce batch or width")

    def layout(self, order: int) -> JetSpec:
        self.check_state(order + 1)
        if order not in self.layouts:
            self.layouts[order] = JetSpec.directional(order, self.operator.budget)
        self.max_coefficients = max(self.max_coefficients, order + 1)
        return self.layouts[order]

    def activate(self, rows: list[Any], layout: JetSpec, name: str, options: dict[str, Any] | None = None) -> list[Any]:
        def compose(values: list[Any], function: str) -> list[Any]:
            return activation(values, layout, function, _OPS) if self.kernels is None else self.kernels.activate(values, layout, function)

        options = options or {}
        if self.method == "polynomial" and name != "power":
            tower = self.tower(rows[0], layout.order, name, options)
            delta = [torch.zeros_like(rows[0]), *rows[1:]]
            power = [torch.ones_like(rows[0])] + [torch.zeros_like(rows[0])] * (layout.size - 1)
            result = [torch.zeros_like(rows[0])] * layout.size
            for k, coefficient in enumerate(tower):
                result = [a + coefficient * b for a, b in zip(result, power, strict=True)]
                if k < layout.order:
                    power = multiply(power, delta, layout, _OPS)
            return result
        if name == "power":
            exponent = options["exponent"]
            if isinstance(exponent, float) and exponent.is_integer():
                exponent = int(exponent)
            if isinstance(exponent, int):
                base = reciprocal(rows, layout, _OPS) if exponent < 0 else rows
                exponent = abs(exponent)
                result = [torch.ones_like(rows[0])] + [torch.zeros_like(rows[0])] * (layout.size - 1)
                while exponent:
                    if exponent & 1:
                        result = multiply(result, base, layout, _OPS)
                    exponent >>= 1
                    if exponent:
                        base = multiply(base, base, layout, _OPS)
                return result
            inner = compose(rows, "log")
            return compose([v * exponent for v in inner], "exp")
        if name == "softplus":
            beta, threshold = options.get("beta", 1), options.get("threshold", 20)
            nonlinear = compose([v * beta for v in rows], name)
            return [torch.where(rows[0] * beta > threshold, a, b / beta) for a, b in zip(rows, nonlinear, strict=True)]
        return compose(rows, name)

    def tower(self, value: Any, order: int, name: str, options: dict[str, Any] | None = None) -> list[Any]:
        """Normalized derivative tower of an atomic activation; no sampled AD."""
        options = options or {}
        self.check_state(order + 1)
        self.max_coefficients = max(self.max_coefficients, order + 1)
        if name in {"sin", "cos"} and self.method == "recurrence":
            sine, cosine = torch.sin(value), torch.cos(value)
            cycle = (sine, cosine, -sine, -cosine) if name == "sin" else (cosine, -sine, -cosine, sine)
            return [cycle[n % 4] / float(factorial(n)) for n in range(order + 1)]
        if self.method == "polynomial" and name != "power":
            if name == "gelu_approx":
                raise NotStructured("the registry has no matching approximate GELU polynomial provider")
            spec = get_activation(name)
            if spec.fastpath is None:
                raise NotStructured(f"no polynomial derivative provider for {name}")
            if name == "softplus":
                beta, threshold = options.get("beta", 1), options.get("threshold", 20)
                rows = [torch.nn.functional.softplus(value, beta=beta, threshold=threshold)]
                for n in range(1, order + 1):
                    nonlinear = spec.fastpath(value * beta, n) * (beta ** (n - 1))
                    linear = torch.ones_like(value) if n == 1 else torch.zeros_like(value)
                    rows.append(torch.where(value * beta > threshold, linear, nonlinear) / float(factorial(n)))
                return rows
            return [spec.forward(value)] + [spec.fastpath(value, n) / float(factorial(n)) for n in range(1, order + 1)]
        layout = self.layout(order)
        rows = [value] + ([torch.ones_like(value)] if order else []) + [torch.zeros_like(value)] * max(0, order - 1)
        return self.activate(rows, layout, name, options)

    def jet(self, expr: Expr, cut: Expr, order: int) -> list[Any]:
        key = expr, cut, order
        if key in self.jets:
            return self.jets[key]
        layout = self.layout(order)
        if expr is cut:
            rows = [expr.value] + ([torch.ones_like(expr.value)] if order else []) + [torch.zeros_like(expr.value)] * max(0, order - 1)
        elif not expr.support:
            rows = [expr.value] + [expr.value * 0] * order
        elif expr.kind == "variable":
            raise NotStructured("a spatial dependency bypasses the scalar cut")
        elif expr.kind in {"add", "multiply"}:
            a, b = (self.jet(child, cut, order) for child in expr.children)
            rows = [x + y for x, y in zip(a, b, strict=True)] if expr.kind == "add" else multiply(a, b, layout, _OPS)
        elif expr.kind == "unary":
            rows = (self.tower(cut.value, order, expr.name or "", expr.options) if expr.children[0] is cut else
                    self.activate(self.jet(expr.children[0], cut, order), layout, expr.name or "", expr.options))
        else:
            raise NotStructured("unsupported scalar region")
        rows[0] = expr.value
        self.jets[key] = rows
        return rows

    def _cut(self, expr: Expr) -> Expr | None:
        candidates = [cut for cut in expr.dominators if cut is not expr and cut.support == expr.support]
        # The earliest dominating scalar keeps the largest downstream region
        # in one univariate jet. No numerical zeros or ranks are consulted.
        return min(candidates, key=lambda cut: len(cut.dominators), default=None)

    def packet(self, expr: Expr, alpha: tuple[int, ...]) -> list[Any]:
        restricted = tuple(a if i in expr.support else 0 for i, a in enumerate(alpha))
        key = expr, restricted
        if key in self.packets:
            return self.packets[key]
        order = sum(restricted)
        self.check_state(order + 1)
        one = expr.value * 0 + 1
        if order == 0:
            result = [one]
            nonzero = frozenset({0})
        elif expr.affine is not None:
            scale = one
            for i, n in enumerate(restricted):
                if n:
                    scale = scale * expr.affine[i] ** n
            result = coordinate_polynomial(order, scale, budget=self.operator.budget)
            nonzero = frozenset({order})
            self.patterns.add("affine-input")
        elif expr.kind == "unary":
            child = expr.children[0]
            inner = self.packet(child, restricted)
            tower = self.tower(child.value, order, expr.name or "", expr.options)
            result = [0, inner[1] * tower[1]] if order == 1 else operator_unary(inner, tower, budget=self.operator.budget)
            nonzero = frozenset(range(1, order + 1))
        elif expr.kind in {"add", "multiply"} and not (expr.children[0].support & expr.children[1].support & {i for i, a in enumerate(restricted) if a}):
            a, b = expr.children
            left, right = self.packet(a, restricted), self.packet(b, restricted)
            if expr.kind == "add":
                left_indices = self.packet_supports[(a, tuple(n if i in a.support else 0 for i, n in enumerate(restricted)))]
                right_indices = self.packet_supports[(b, tuple(n if i in b.support else 0 for i, n in enumerate(restricted)))]
                nonzero = frozenset(i + j for i in left_indices for j in right_indices)
                if len(left) == 1:
                    result = right
                elif len(right) == 1:
                    result = left
                else:
                    result = [0] * (len(left) + len(right) - 1)
                    for i in left_indices:
                        for j in right_indices:
                            result[i + j] = result[i + j] + left[i] * right[j]
            else:
                result = operator_product(left, right, a.value, b.value, budget=self.operator.budget)
                nonzero = frozenset(range(1, order + 1))
            self.patterns.add("disjoint-support-" + expr.kind)
        else:
            cut = self._cut(expr)
            if cut is None:
                raise NotStructured("overlapping spatial supports have no dominating scalar cut")
            result = operator_unary(self.packet(cut, restricted), self.jet(expr, cut, order), budget=self.operator.budget)
            nonzero = frozenset(range(1, order + 1))
            self.patterns.add("scalar-cut")
        self.max_coefficients = max(self.max_coefficients, len(result))
        self.packets[key] = result
        self.packet_supports[key] = nonzero
        return result

    def readout(self, expr: Expr, alpha: tuple[int, ...], tower: list[Any]) -> Any:
        """Read only coefficient positions proved nonzero by structural rules."""
        packet = self.packet(expr, alpha)
        restricted = tuple(n if i in expr.support else 0 for i, n in enumerate(alpha))
        return sum(packet[k] * tower[k] * float(factorial(k)) for k in self.packet_supports[(expr, restricted)])

    def derivative(self, expr: Expr, alpha: tuple[int, ...]) -> Any:
        if not any(alpha):
            return expr.value
        if any(n and i not in expr.support for i, n in enumerate(alpha)):
            return expr.value * 0
        if expr.kind == "add":
            return sum(self.derivative(child, alpha) for child in expr.children)
        if expr.kind == "multiply" and any(not child.support for child in expr.children):
            a, b = expr.children
            return a.value * self.derivative(b, alpha) if not a.support else self.derivative(a, alpha) * b.value
        order = sum(alpha)
        cut = self._cut(expr)
        if cut is not None:
            self.patterns.add("scalar-cut")
            return self.readout(cut, alpha, self.jet(expr, cut, order))
        if expr.kind == "unary":
            child = expr.children[0]
            return self.readout(child, alpha, self.tower(child.value, order, expr.name or "", expr.options))
        packet = self.packet(expr, alpha)
        return packet[1] if len(packet) > 1 else expr.value
