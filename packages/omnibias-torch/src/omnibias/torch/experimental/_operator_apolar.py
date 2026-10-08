# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded exact operator algebras for captured scalar expression regions."""

from __future__ import annotations

from itertools import product
from math import prod
from typing import Any

from omnibias.core.apolar import ApolarAlgebra, apolar_algebra
from omnibias.core.operator_quotient import (
    quotient_compose,
    quotient_estimate,
    quotient_multiply,
    quotient_readout,
    quotient_seed,
)
from omnibias.core.operator_spec import OperatorSpec

from ._operator_ir import Expr, NotStructured
from ._operator_lowering import Lowering


def select_apolar(request: OperatorSpec, dimension: int) -> ApolarAlgebra:
    """Select only a bounded homogeneous quotient smaller than a mixed box.

    Never enumerate the ambient mixed closure to make this decision. The
    largest target box is a lower bound on the union-of-boxes reference state.
    Construction may refuse independently of whether another strategy works.
    """
    if request.mode == "directional" or not request.order:
        raise NotStructured("a directional/primal query does not need apolar compression")
    terms = request.resolve_terms(dimension)
    if len({sum(alpha) for alpha, _ in terms}) != 1:
        raise NotStructured("the bounded apolar strategy requires a homogeneous operator")
    largest_box = max(prod(a + 1 for a in alpha) for alpha, _ in terms)
    # A union may be substantially larger than every box; enumerate at most
    # 512 terms to distinguish cases such as a Laplacian, never a huge layout.
    closure: set[tuple[int, ...]] = set()
    for alpha, _ in terms:
        if prod(a + 1 for a in alpha) > 512:
            break
        for beta in product(*(range(a + 1) for a in alpha)):
            closure.add(beta)
            if len(closure) > 512:
                break
        if len(closure) > 512:
            break
    lower_bound = max(largest_box, len(closure))
    try:
        algebra = apolar_algebra(
            dict(terms), max_monomials=min(512, request.planning_budget.max_planning_work),
            max_basis=min(128, request.budget.max_coefficients),
            max_products=request.budget.max_multiplication_terms,
            max_order=min(32, request.planning_budget.max_order),
            max_coefficient_bits=request.planning_budget.max_coefficient_bits,
        )
    except ValueError as exc:
        raise NotStructured(f"bounded apolar planning declined: {exc}") from exc
    if algebra.size >= lower_bound:
        raise NotStructured("apolar state is not smaller than the selected mixed layout")
    quotient_estimate(algebra, "compose", budget=request.budget)
    return algebra


def apolar_derivative(expr: Expr, algebra: ApolarAlgebra, lowering: Lowering,
                      memo: dict[Expr, list[Any]] | None = None) -> Any:
    """Emit a live-parameter quotient program and its raw differential readout."""
    cache = {} if memo is None else memo
    budget = lowering.operator.budget

    def lift(node: Expr) -> list[Any]:
        if node in cache:
            return cache[node]
        zero = node.value * 0
        if not node.support:
            result = [node.value] + [zero] * (algebra.size - 1)
        elif node.kind == "variable":
            assert node.coordinate is not None
            result = quotient_seed(node.value, node.coordinate, algebra, budget=budget)
        elif node.kind in {"add", "multiply"}:
            a, b = (lift(child) for child in node.children)
            result = ([x + y for x, y in zip(a, b, strict=True)]
                      if node.kind == "add" else quotient_multiply(a, b, algebra, budget=budget))
        elif node.kind == "unary":
            rows = lift(node.children[0])
            tower = lowering.tower(rows[0], algebra.order, node.name or "", node.options)
            result = quotient_compose(rows, tower, algebra, budget=budget)
        else:
            raise NotStructured(f"no apolar lift for scalar region {node.kind}")
        # Keep the captured primitive's primal semantics (for example exact
        # versus approximate GELU), including floating-point evaluation order.
        result[0] = node.value
        cache[node] = result
        return result

    rows = lift(expr)
    lowering.max_coefficients = max(lowering.max_coefficients, algebra.size)
    lowering.patterns.add("apolar-quotient")
    return quotient_readout(rows, algebra, budget=budget)
