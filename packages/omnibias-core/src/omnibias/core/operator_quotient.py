# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded tensor-independent lifting in an exact apolar quotient algebra.

Coordinates are in ``ApolarAlgebra.indices`` order, with the unit first.
Multiplication weights and the raw operator readout come from exact host
metadata. Only their use with a numerical backend converts rational weights
to floating point. No tensor-pair array, detached parameter, or nested spatial
autodiff is needed. Native reverse AD remains applicable to every operation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from itertools import chain
from math import isfinite
from operator import mul
from typing import Any

from omnibias.core.apolar import ApolarAlgebra
from omnibias.core.jet_algebra import JetBudget, JetSpec
from omnibias.core.jet_series import activation, activation_derivative


def _zero(value: Any, ops: Any) -> Any:
    return value * 0 if ops is None else ops.zeros_like(value)


def _one(value: Any, ops: Any) -> Any:
    return value * 0 + 1 if ops is None else ops.ones_like(value)


def _scale(value: Any, weight: Fraction) -> Any:
    return value * (weight if isinstance(value, (int, Fraction)) else float(weight))


def _validate_weights(algebra: ApolarAlgebra, sample: Any) -> None:
    if isinstance(sample, (int, Fraction)):
        return
    weights = chain((weight for _, _, _, weight in algebra.products), algebra.readout,
                    chain.from_iterable(algebra.generators))
    for weight in weights:
        try:
            converted = float(weight)
        except OverflowError as exc:
            raise ValueError("apolar coefficient exceeds the binary64 metadata range") from exc
        if not isfinite(converted) or (weight and not converted):
            raise ValueError("apolar coefficient overflows or underflows the binary64 metadata range")


def _validate(a: Sequence[Any], algebra: ApolarAlgebra) -> None:
    if len(a) != algebra.size:
        raise ValueError(f"quotient element has {len(a)} coordinates; expected {algebra.size}")


def quotient_estimate(algebra: ApolarAlgebra, operation: str = "compose", *,
                      budget: JetBudget | None = None) -> dict[str, int]:
    """Check forward coefficient slots and arithmetic before executing a kernel.

    Estimates cover live coefficient arrays and scalar product contributions,
    including a conservative allowance for builtin activation tower rules.
    Tensor shapes, framework workspaces and reverse tapes are frontend duties.
    No plan is cached, and no coordinate-pair tensor is allocated.
    """
    size, order, products = algebra.size, algebra.order, len(algebra.products)
    if operation == "multiply":
        slots, work = size, products
    elif operation in {"compose", "reciprocal"}:
        slots, work = 3 * size + order + 1, order * products + (order + 1) * size
    elif operation in {"activation", "activation_derivative"}:
        slots = 3 * size + 32 * (order + 1)
        work = order * products + (order + 1) * size + 64 * (order + 1) * (order + 2) // 2
    elif operation in {"seed", "readout", "transpose"}:
        slots, work = size, products if operation == "transpose" else size
    else:
        raise ValueError(f"unknown quotient operation {operation!r}")
    (budget or JetBudget()).check_layout(slots, work)
    return {"coefficients": size, "live_coefficient_slots": slots, "multiplication_terms": work}


def quotient_seed(value: Any, coordinate: int, algebra: ApolarAlgebra, ops: Any = None, *,
                  budget: JetBudget | None = None) -> list[Any]:
    """Lift one input coordinate, retaining its ordinary primal tensor graph."""
    quotient_estimate(algebra, "seed", budget=budget)
    if isinstance(coordinate, bool) or not isinstance(coordinate, int) or not 0 <= coordinate < algebra.dim:
        raise ValueError(f"coordinate must lie in [0, {algebra.dim})")
    _validate_weights(algebra, value)
    one = _one(value, ops)
    return [value] + [_scale(one, c) for c in algebra.generators[coordinate][1:]]


def _multiply(a: Sequence[Any], b: Sequence[Any], algebra: ApolarAlgebra,
              product: Callable[[Any, Any], Any], ops: Any) -> list[Any]:
    # Use an actual product to establish the result shape (matmul/conv may
    # change it), rather than allocating an array of every coefficient pair.
    primal = product(a[0], b[0])
    zero = _zero(primal, ops)
    result = [zero] * algebra.size
    for i, j, k, coefficient in algebra.products:
        term = primal if i == 0 and j == 0 else product(a[i], b[j])
        result[k] = result[k] + _scale(term, coefficient)
    return result


def quotient_multiply(a: Sequence[Any], b: Sequence[Any], algebra: ApolarAlgebra,
                      product: Callable[[Any, Any], Any] = mul, ops: Any = None, *,
                      budget: JetBudget | None = None) -> list[Any]:
    """Weighted Cauchy product; ``product`` may be a backend matmul or convolution."""
    _validate(a, algebra)
    _validate(b, algebra)
    quotient_estimate(algebra, "multiply", budget=budget)
    _validate_weights(algebra, a[0])
    _validate_weights(algebra, b[0])
    return _multiply(a, b, algebra, product, ops)


def _compose(a: Sequence[Any], normalized_tower: Sequence[Any], algebra: ApolarAlgebra,
             ops: Any) -> list[Any]:
    zero = _zero(a[0], ops)
    delta = [zero, *a[1:]]
    result = [normalized_tower[0]] + [zero] * (algebra.size - 1)
    power = delta
    for degree in range(1, algebra.order + 1):
        if degree > 1:
            power = _multiply(power, delta, algebra, mul, ops)
        for i in range(1, algebra.size):
            result[i] = result[i] + normalized_tower[degree] * power[i]
    return result


def quotient_compose(a: Sequence[Any], normalized_tower: Sequence[Any], algebra: ApolarAlgebra,
                     ops: Any = None, *, budget: JetBudget | None = None) -> list[Any]:
    """Evaluate sum(phi**(k)(a0)/k! * (a-a0)**k), exactly modulo the ideal.

    The supplied tower must use the same primal ``a[0]`` and include orders
    zero through ``algebra.order``. This finite formula does not require an
    Euler grading recurrence and is valid for the homogeneous apolar bases
    currently constructed by core.
    """
    _validate(a, algebra)
    if len(normalized_tower) < algebra.order + 1:
        raise ValueError(f"quotient composition needs a tower through order {algebra.order}")
    quotient_estimate(algebra, "compose", budget=budget)
    _validate_weights(algebra, a[0])
    return _compose(a, normalized_tower, algebra, ops)


def quotient_reciprocal(a: Sequence[Any], algebra: ApolarAlgebra, ops: Any = None, *,
                        budget: JetBudget | None = None) -> list[Any]:
    """Reciprocal in the local algebra; its scalar primal must be nonzero."""
    _validate(a, algebra)
    quotient_estimate(algebra, "reciprocal", budget=budget)
    _validate_weights(algebra, a[0])
    inverse = _one(a[0], ops) / a[0]
    tower = [inverse]
    for _ in range(algebra.order):
        tower.append(-tower[-1] * inverse)
    return _compose(a, tower, algebra, ops)


def _activation(a: Sequence[Any], algebra: ApolarAlgebra, name: str, ops: Any,
                budget: JetBudget | None, directional_spec: JetSpec | None, derivative: bool) -> list[Any]:
    _validate(a, algebra)
    quotient_estimate(algebra, "activation_derivative" if derivative else "activation", budget=budget)
    _validate_weights(algebra, a[0])
    spec = directional_spec or JetSpec.directional(algebra.order, budget)
    if spec.mode != "directional" or spec.order != algebra.order:
        raise ValueError("activation tower layout must be directional with the quotient order")
    path = [a[0]]
    if algebra.order:
        path += [_one(a[0], ops)] + [_zero(a[0], ops)] * (algebra.order - 1)
    tower = (activation_derivative if derivative else activation)(path, spec, name, ops)
    return _compose(a, tower, algebra, ops)


def quotient_activation(a: Sequence[Any], algebra: ApolarAlgebra, name: str, ops: Any, *,
                        budget: JetBudget | None = None, directional_spec: JetSpec | None = None) -> list[Any]:
    """Lift a builtin smooth activation using its existing stable normalized tower.

    Frontends may prepare ``directional_spec`` outside framework tracing.
    Activation aliases and exact-versus-approximate semantics match jet_series.
    """
    return _activation(a, algebra, name, ops, budget, directional_spec, False)


def quotient_activation_derivative(a: Sequence[Any], algebra: ApolarAlgebra, name: str, ops: Any, *,
                                   budget: JetBudget | None = None,
                                   directional_spec: JetSpec | None = None) -> list[Any]:
    """Lift sigma'(a), including the extra smoothness needed by its pullback."""
    return _activation(a, algebra, name, ops, budget, directional_spec, True)


def quotient_readout(a: Sequence[Any], algebra: ApolarAlgebra, *, budget: JetBudget | None = None) -> Any:
    """Return the raw constant-coefficient differential operator value."""
    _validate(a, algebra)
    quotient_estimate(algebra, "readout", budget=budget)
    _validate_weights(algebra, a[0])
    total = a[0] * 0
    for value, coefficient in zip(a, algebra.readout, strict=True):
        total = total + _scale(value, coefficient)
    return total


def quotient_multiply_transpose(cotangent: Sequence[Any], multiplier: Sequence[Any], algebra: ApolarAlgebra,
                                ops: Any = None, *, budget: JetBudget | None = None) -> list[Any]:
    """Transpose of scalar multiplication by an algebra element, for analytic VJPs.

    This is the Euclidean coefficient transpose, without assuming an
    orthonormal or Frobenius-normalized basis. Matrix-product transposes and
    broadcasting reductions remain frontend responsibilities.
    """
    _validate(cotangent, algebra)
    _validate(multiplier, algebra)
    quotient_estimate(algebra, "transpose", budget=budget)
    _validate_weights(algebra, cotangent[0])
    _validate_weights(algebra, multiplier[0])
    zero = _zero(cotangent[0] * multiplier[0], ops)
    result = [zero] * algebra.size
    for i, j, k, coefficient in algebra.products:
        result[i] = result[i] + _scale(cotangent[k] * multiplier[j], coefficient)
    return result


__all__ = ["quotient_activation", "quotient_activation_derivative", "quotient_compose", "quotient_estimate",
           "quotient_multiply", "quotient_multiply_transpose", "quotient_readout", "quotient_reciprocal",
           "quotient_seed"]
