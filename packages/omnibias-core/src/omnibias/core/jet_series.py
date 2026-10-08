# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-independent Taylor rules on lists of ordinary tensor coefficients.

``ops`` supplies elementary value functions and ``zeros_like``/``ones_like``.
No arrays are detached or converted to host values. Polynomial differential
systems use the Euler identity ``E y = P(u,y) E u`` and cache each monomial
coefficient once. For fixed systems their work is linear in the layout's
Cauchy-pair count (quadratic in directional order).
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import pi, sqrt
from typing import Any

from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_systems import SYSTEMS, DifferentialSystemSpec, get_system

MonomialPlan = tuple[tuple[tuple[int, int], ...], tuple[tuple[tuple[Fraction, int], ...], ...]]


def _zero(value: Any, ops: Any) -> Any:
    return ops.zeros_like(value) if ops is not None else value * 0


def _one(value: Any, ops: Any) -> Any:
    return ops.ones_like(value) if ops is not None else value * 0 + 1


def _validate(a: Sequence[Any], spec: JetSpec) -> None:
    if len(a) != spec.size:
        raise ValueError(f"jet has {len(a)} coefficients; layout requires {spec.size}")


def _scale(value: Any, coefficient: Fraction) -> Any:
    # Integer arithmetic also preserves exact Fraction-valued oracle tests.
    if coefficient == 1:
        return value
    result = value * coefficient.numerator
    return result if coefficient.denominator == 1 else result / coefficient.denominator


def multiply(a: Sequence[Any], b: Sequence[Any], spec: JetSpec, ops: Any = None) -> list[Any]:
    """Truncated Cauchy product in the selected downward-closed layout."""
    _validate(a, spec)
    _validate(b, spec)
    zero = _zero(a[0] * b[0], ops)
    result = []
    for pairs in spec.pairs:
        total = zero
        for i, j in pairs:
            total = total + a[i] * b[j]
        result.append(total)
    return result


def reciprocal(a: Sequence[Any], spec: JetSpec, ops: Any = None) -> list[Any]:
    """Reciprocal jet, defined when the primal coefficient is nonzero."""
    _validate(a, spec)
    result = [_one(a[0], ops) / a[0]]
    zero = _zero(a[0], ops)
    for pairs in spec.pairs[1:]:
        total = zero
        for i, j in pairs:
            if i:
                total = total + a[i] * result[j]
        result.append(-total / a[0])
    return result


@lru_cache(maxsize=128)
def _monomial_plan(system: DifferentialSystemSpec) -> MonomialPlan:
    variables = len(system.states) + 1
    known = {(0,) * variables: variables}
    for i in range(variables):
        known[tuple(int(j == i) for j in range(variables))] = i
    nodes: list[tuple[int, int]] = []

    def node(powers: tuple[int, ...]) -> int:
        if powers not in known:
            axis = next(i for i, degree in enumerate(powers) if degree)
            rest = tuple(degree - int(i == axis) for i, degree in enumerate(powers))
            left = node(rest)
            known[powers] = variables + 1 + len(nodes)
            nodes.append((left, axis))
        return known[powers]

    polynomials = tuple(tuple((coefficient, node(powers)) for coefficient, powers in poly)
                        for poly in (*system.rhs, system.output))
    return tuple(nodes), polynomials


def differential_system(a: Sequence[Any], spec: JetSpec, system: DifferentialSystemSpec,
                        initial: Sequence[Any], ops: Any = None, *,
                        plan: MonomialPlan | None = None,
                        initial_rhs: Sequence[Any] | None = None) -> list[Any]:
    """Integrate a polynomial ODE along an arbitrary incoming Taylor path.

    ``initial`` contains each auxiliary state's value at ``a[0]``. These values
    must remain connected to the primal when parameter gradients are needed.
    The state equations and input path, not sampled differences, set every
    subsequent coefficient. This is a formal local rule, not an ODE timestep.
    Builtin rules pass an immutable precomputed ``plan`` so traced execution
    never enters a Python cache wrapper. Custom systems may omit this argument.
    ``initial_rhs`` optionally supplies algebraically equivalent, numerically
    stable state derivatives at the primal. It replaces only RHS degree zero;
    all higher coefficients still follow the declared polynomial equations.
    """
    _validate(a, spec)
    if len(initial) != len(system.states):
        raise ValueError("initial values must match the differential-system states")
    if initial_rhs is not None and len(initial_rhs) != len(system.states):
        raise ValueError("initial RHS values must match the differential-system states")
    nodes, polynomials = _monomial_plan(system) if plan is None else plan
    zero, one = _zero(a[0], ops), _one(a[0], ops)
    states = [[value] + [zero] * (spec.size - 1) for value in initial]
    values = [list(a), *states, [one] + [zero] * (spec.size - 1)]
    values.extend([zero] * spec.size for _ in nodes)
    derivatives = [[zero] * spec.size for _ in states]
    output = [zero] * spec.size
    first_product = len(states) + 2

    def polynomial(poly: tuple[tuple[Fraction, int], ...], k: int) -> Any:
        total = zero
        for coefficient, source in poly:
            total = total + _scale(values[source][k], coefficient)
        return total

    for k, degree in enumerate(spec.degrees):
        if k:
            for state, rhs in zip(states, derivatives, strict=True):
                total = zero
                for i, j in spec.pairs[k]:
                    if i:
                        total = total + (a[i] * spec.degrees[i]) * rhs[j]
                state[k] = total / degree
        # Products may involve multiple auxiliary states. All coefficients of
        # this degree are solved before advancing the shared monomial DAG.
        for offset, (left, right) in enumerate(nodes):
            total = zero
            for i, j in spec.pairs[k]:
                total = total + values[left][i] * values[right][j]
            values[first_product + offset][k] = total
        for state_index, (rhs, poly) in enumerate(zip(derivatives, polynomials[:-1], strict=True)):
            rhs[k] = initial_rhs[state_index] if k == 0 and initial_rhs is not None else polynomial(poly, k)
        output[k] = polynomial(polynomials[-1], k)
    return output


def _initial(name: str, value: Any, ops: Any) -> Any:
    if name == "reciprocal":
        return _one(value, ops) / value
    if name == "normal_cdf":
        # Avoid 1 + erf(z/sqrt(2)), which rounds to zero in the negative tail.
        return ops.erfc(-value / sqrt(2)) / 2
    if name in {"gaussian", "gaussian_normalized", "gaussian_rbf", "erf_density"}:
        if name in {"gaussian_rbf", "erf_density"}:
            result = ops.exp(-(value * value))
        else:
            result = ops.exp(-(value * value) / 2)
        if name == "gaussian_normalized":
            result = result / sqrt(2 * pi)
        elif name == "erf_density":
            result = result * (2 / sqrt(pi))
        return result
    if name == "softplus":
        if hasattr(ops, "logaddexp"):
            return ops.logaddexp(_zero(value, ops), value)
        if hasattr(ops, "softplus"):
            return ops.softplus(value)
        return ops.log(_one(value, ops) + ops.exp(value))
    return getattr(ops, name)(value)


_ALIASES = {"gelu_exact": "gelu", "gelu_tanh": "gelu_approx", "swish": "silu",
            "normal_pdf": "gaussian_normalized", "rbf": "gaussian_rbf"}


def _stable_initial_rhs(name: str, value: Any, initial: Sequence[Any], ops: Any) -> list[Any] | None:
    # Avoid cancellation in 1-tanh(z)^2 and sigmoid(z)-sigmoid(z)^2
    # when the rounded primal has saturated. Minimal custom ops without a
    # sigmoid retain the original polynomial rule and its precision behavior.
    if not hasattr(ops, "sigmoid"):
        return None
    if name == "tanh":
        return [4 * ops.sigmoid(2 * value) * ops.sigmoid(-2 * value)]
    if name in {"sigmoid", "softplus"}:
        slope = ops.sigmoid(value) * ops.sigmoid(-value)
        return [initial[1], slope] if name == "softplus" else [slope]
    return None


def activation(a: Sequence[Any], spec: JetSpec, name: str, ops: Any) -> list[Any]:
    """Elementwise smooth composition using native backend value operations.

    Names include tanh, sigmoid, exp, sin, cos, erf, erfc, log, sqrt,
    reciprocal, softplus, silu, gelu (exact), gelu_approx (tanh approximation),
    gaussian = exp(-x*x/2), gaussian_normalized = gaussian/sqrt(2*pi), and
    gaussian_rbf = exp(-x*x). The approximation in gelu_approx is its declared
    primal function; its jet coefficients differentiate that same function.
    """
    _validate(a, spec)
    name = _ALIASES.get(name, name)
    if name in {"identity", "linear"}:
        return list(a)
    if name == "reciprocal":
        return reciprocal(a, spec, ops)
    if name == "silu":
        return multiply(a, activation(a, spec, "sigmoid", ops), spec, ops)
    if name == "gelu_approx":
        square = multiply(a, a, spec, ops)
        cube = multiply(square, a, spec, ops)
        inner = [(x + y * 0.044715) * sqrt(2 / pi) for x, y in zip(a, cube, strict=True)]
        if hasattr(ops, "sigmoid"):
            # (1 + tanh(h))/2 = sigmoid(2h), without subtractive
            # cancellation when tanh(h) rounds to -1 in the negative tail.
            blend = activation([2*x for x in inner], spec, "sigmoid", ops)
            return multiply(a, blend, spec, ops)
        blend = activation(inner, spec, "tanh", ops)
        blend[0] = blend[0] + _one(a[0], ops)
        return [x / 2 for x in multiply(a, blend, spec, ops)]
    system = get_system(name)
    initial = [_initial(state, a[0], ops) for state in system.states]
    return differential_system(a, spec, system, initial, ops, plan=_BUILTIN_PLANS[name],
                               initial_rhs=_stable_initial_rhs(name, a[0], initial, ops))


@lru_cache(maxsize=128)
def _derivative_system(system: DifferentialSystemSpec) -> DifferentialSystemSpec:
    terms = []
    constant = ((Fraction(1), (0,) * (len(system.states) + 1)),)
    for coefficient, powers in system.output:
        for axis, exponent in enumerate(powers):
            if not exponent:
                continue
            rest = tuple(p - int(i == axis) for i, p in enumerate(powers))
            for rhs_coefficient, rhs_powers in (constant if axis == 0 else system.rhs[axis - 1]):
                terms.append((coefficient * exponent * rhs_coefficient,
                              tuple(p + q for p, q in zip(rest, rhs_powers, strict=True))))
    return DifferentialSystemSpec(system.name + "_derivative", system.states, system.rhs,
                                  tuple(terms), system.domain)


# Construct immutable schedules outside every traced tensor program. Keep the
# bounded metadata caches for custom systems, without invoking them at runtime
# for standard activations or their analytic transpose rules.
_BUILTIN_PLANS = {name: _monomial_plan(system) for name, system in SYSTEMS.items()}
_DERIVATIVE_SYSTEMS = {name: _derivative_system(system) for name, system in SYSTEMS.items()}
_DERIVATIVE_PLANS = {name: _monomial_plan(system) for name, system in _DERIVATIVE_SYSTEMS.items()}


def activation_derivative(a: Sequence[Any], spec: JetSpec, name: str, ops: Any) -> list[Any]:
    """Jet of ``sigma'(u)`` for transposing an activation's coefficient map.

    The output has the same layout/order as ``a``. It includes the additional
    derivative information needed to backpropagate through the base value.
    """
    _validate(a, spec)
    name = _ALIASES.get(name, name)
    zero, one = _zero(a[0], ops), _one(a[0], ops)
    identity = [one] + [zero] * (spec.size - 1)
    if name in {"identity", "linear"}:
        return identity
    if name == "reciprocal":
        inv = reciprocal(a, spec, ops)
        return [-x for x in multiply(inv, inv, spec, ops)]
    if name == "silu":
        sig = activation(a, spec, "sigmoid", ops)
        correction = multiply(a, activation_derivative(a, spec, "sigmoid", ops), spec, ops)
        return [s + c for s, c in zip(sig, correction, strict=True)]
    if name == "gelu_approx":
        square = multiply(a, a, spec, ops)
        cube = multiply(square, a, spec, ops)
        inner = [(x + y * 0.044715) * sqrt(2 / pi) for x, y in zip(a, cube, strict=True)]
        inner_prime = [(i + s * (3 * 0.044715)) * sqrt(2 / pi)
                       for i, s in zip(identity, square, strict=True)]
        if hasattr(ops, "sigmoid"):
            doubled = [2*x for x in inner]
            blend = activation(doubled, spec, "sigmoid", ops)
            correction = multiply(a, multiply(activation_derivative(doubled, spec, "sigmoid", ops),
                                               [2*x for x in inner_prime], spec, ops), spec, ops)
            return [s + c for s, c in zip(blend, correction, strict=True)]
        tanh = activation(inner, spec, "tanh", ops)
        correction = multiply(a, multiply(activation_derivative(inner, spec, "tanh", ops),
                                           inner_prime, spec, ops), spec, ops)
        return [(i + t + c) / 2 for i, t, c in zip(identity, tanh, correction, strict=True)]
    system = get_system(name)
    initial = [_initial(state, a[0], ops) for state in system.states]
    rhs = _stable_initial_rhs(name, a[0], initial, ops)
    result = differential_system(a, spec, _DERIVATIVE_SYSTEMS[name], initial, ops,
                                 plan=_DERIVATIVE_PLANS[name], initial_rhs=rhs)
    if name in {"tanh", "sigmoid"} and rhs is not None:
        result[0] = rhs[0]
    return result


__all__ = ["activation", "activation_derivative", "differential_system", "multiply", "reciprocal"]
