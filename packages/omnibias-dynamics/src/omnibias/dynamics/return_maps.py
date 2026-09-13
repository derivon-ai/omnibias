# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Source-bound finite-time first-hit maps and their parameter derivatives.

All fields, initial embeddings, and events use exact ``SparsePolynomial``
sources. Interval Taylor integration encloses both the flow and its internally
derived first/second variational equations. A certificate requires a transverse
bracket, no earlier positive target zero, and exclusion of every competing
event. A singular or undecided passage is unresolved, never silently skipped.

This is a finite-time regular-passage tool, not a singular Dulac-map theorem or
a global periodic-orbit certificate. Replay with ``verify_stopped_event`` when
consuming a stored or otherwise untrusted result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.ode import (
    TaylorSeries,
    VectorField,
    _apriori_enclosure,
    _solution_coeffs,
)

Box = tuple[Interval, ...]
Matrix = tuple[Box, ...]
Tensor = tuple[Matrix, ...]
Status = Literal["certified", "excluded", "unresolved"]


def polynomial_interval(poly: SparsePolynomial, box: Sequence[Interval]) -> Interval:
    """Evaluate the exact rational polynomial with outward interval arithmetic."""
    if len(box) != poly.nvars:
        raise ValueError("polynomial interval argument has wrong dimension")
    out = Interval.point(0.0)
    for powers, coefficient in poly.terms:
        term = Interval.from_rational(coefficient)
        for value, power in zip(box, powers, strict=True):
            if power:
                term = term * value.pow_int(power)
        out = out + term
    return out


def _series_eval(poly: SparsePolynomial, values: Sequence[TaylorSeries]) -> TaylorSeries:
    order = values[0].order
    out = TaylorSeries.constant(0, order)
    for powers, coefficient in poly.terms:
        term = TaylorSeries.constant(coefficient, order)
        for value, power in zip(values, powers, strict=True):
            factor = value
            while power:
                if power & 1:
                    term = term * factor
                power //= 2
                if power:
                    factor = factor * factor
        out = out + term
    return out


@dataclass(frozen=True)
class PolynomialFlow:
    """Physical components in variables ``(state, parameters, clock)``.

    There is one component per physical state coordinate. Parameters have
    generated zero velocity and the last, explicit clock has velocity one.
    """

    components: tuple[SparsePolynomial, ...]
    parameter_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        if type(self.parameter_count) is not int or self.parameter_count < 0:
            raise ValueError("parameter_count must be a nonnegative integer")
        if not self.components or any(p.nvars != self.dimension for p in self.components):
            raise ValueError("field polynomials need state + parameter + clock variables")

    @property
    def state_count(self) -> int:
        return len(self.components)

    @property
    def dimension(self) -> int:
        return self.state_count + self.parameter_count + 1

    def extended_components(self) -> tuple[SparsePolynomial, ...]:
        d = self.dimension
        return self.components + (SparsePolynomial.constant(d, 0),) * self.parameter_count + (
            SparsePolynomial.constant(d, 1),
        )


@dataclass(frozen=True)
class PolynomialEvent:
    """Zero section in ``(state, parameters, clock)``; direction is -1, 0, or 1.

    ``guard`` restricts the event to its strictly positive set, for example
    ``x=0, y>0``. Without a guard every section zero is eligible. The target's
    direction restricts the first eligible positive-time zero. Earlier eligible
    zeros in the opposite direction are not skipped. Competitors stop on every
    eligible zero, regardless of their direction setting.
    """

    polynomial: SparsePolynomial
    direction: int = 0
    name: str = "target"
    guard: SparsePolynomial | None = None

    def __post_init__(self) -> None:
        if type(self.direction) is not int or self.direction not in (-1, 0, 1):
            raise ValueError("event direction must be -1, 0, or 1")
        if not self.name:
            raise ValueError("event name must be nonempty")
        if self.guard is not None and self.guard.nvars != self.polynomial.nvars:
            raise ValueError("event guard dimension differs from section dimension")


@dataclass(frozen=True)
class StoppedEventRequest:
    """Complete replayable input; derivatives are with respect to ``parameters``.

    ``initial`` is a polynomial embedding of the parameter box into physical
    state space. An initial target identity is inferred by exact substitution;
    a strictly signed departure is then required before a positive return.
    ``step`` denotes its exact binary floating-point value. The clock starts
    at zero; explicit time dependence belongs in the final polynomial axis.
    """

    flow: PolynomialFlow
    initial: tuple[SparsePolynomial, ...]
    parameters: Box
    target: PolynomialEvent
    competitors: tuple[PolynomialEvent, ...] = ()
    step: float = 0.125
    max_steps: int = 128
    order: int = 10
    derivative_order: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial", tuple(self.initial))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "competitors", tuple(self.competitors))
        m = self.flow.parameter_count
        if len(self.initial) != self.flow.state_count or any(p.nvars != m for p in self.initial):
            raise ValueError("initial embedding must have one polynomial in parameters per state")
        if len(self.parameters) != m or not _finite(self.parameters):
            raise ValueError("finite parameter box of matching dimension required")
        if any(e.polynomial.nvars != self.flow.dimension for e in self.events):
            raise ValueError("event polynomial dimension differs from field dimension")
        if type(self.step) not in (int, float):
            raise TypeError("step must specify an exact binary float (or integer)")
        object.__setattr__(self, "step", float(self.step))
        if not math.isfinite(self.step) or self.step <= 0:
            raise ValueError("step must be finite and positive")
        if any(type(v) is not int or v < 1 for v in (self.max_steps, self.order)):
            raise ValueError("max_steps and order must be positive integers")
        if type(self.derivative_order) is not int or self.derivative_order not in (0, 1, 2):
            raise ValueError("supported derivative orders are 0, 1, and 2")

    @property
    def events(self) -> tuple[PolynomialEvent, ...]:
        return (self.target, *self.competitors)

    @property
    def fingerprint(self) -> str:
        body = {
            "schema": "polynomial-stopped-event-v1",
            "field": [p.to_payload() for p in self.flow.components],
            "parameter_count": self.flow.parameter_count,
            "initial": [p.to_payload() for p in self.initial],
            "parameters": [(float(x.lo).hex(), float(x.hi).hex()) for x in self.parameters],
            "events": [(e.name, e.direction, e.polynomial.to_payload(),
                        e.guard.to_payload() if e.guard is not None else None) for e in self.events],
            "step": float(self.step).hex(),
            "max_steps": self.max_steps,
            "order": self.order,
            "derivative_order": self.derivative_order,
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class EventSlab:
    """Generated event bounds on one complete physical trajectory slab."""

    time: Interval
    state_tube: Box
    start_values: Box
    end_values: Box
    tube_values: Box
    normal_velocities: Box
    guard_values: Box
    target_action: str


@dataclass(frozen=True)
class StoppedEventResult:
    """Finite-time certificate or explicit unresolved/excluded outcome.

    ``excluded`` excludes positive hits on the checked finite horizon only.
    A certified event is unique and first for every parameter in the source
    box. State derivatives have indices ``[state][parameter]`` and
    ``[state][parameter_a][parameter_b]``. They include the event-time terms.
    """

    request: StoppedEventRequest
    source_fingerprint: str
    status: Status
    reason: str
    slabs: tuple[EventSlab, ...]
    time_bracket: Interval | None = None
    return_box: Box | None = None
    normal_velocity: Interval | None = None
    event_time_gradient: Box | None = None
    event_time_hessian: Matrix | None = None
    return_jacobian: Matrix | None = None
    return_hessian: Tensor | None = None

    @property
    def certified(self) -> bool:
        """Outcome of this run; untrusted results additionally require replay."""
        return self.status == "certified"


def _finite(box: Sequence[Interval]) -> bool:
    return all(math.isfinite(x.lo) and math.isfinite(x.hi) for x in box)


def _sign(value: Interval) -> int:
    return 1 if value.lo > 0 else -1 if value.hi < 0 else 0


def _dot(a: Sequence[Interval], b: Sequence[Interval]) -> Interval:
    out = Interval.point(0.0)
    for x, y in zip(a, b, strict=True):
        out = out + x * y
    return out


def _initial_event(request: StoppedEventRequest, event: PolynomialEvent) -> SparsePolynomial:
    m = request.flow.parameter_count
    values = request.initial + tuple(SparsePolynomial.variable(m, i) for i in range(m)) + (
        SparsePolynomial.constant(m, 0),
    )
    out = SparsePolynomial.constant(m, 0, budget=event.polynomial.budget)
    for powers, coefficient in event.polynomial.terms:
        term = SparsePolynomial.constant(m, coefficient, budget=event.polynomial.budget)
        for value, power in zip(values, powers, strict=True):
            if power:
                term = term * value**power
        out = out + term
    return out


class _JetSystem:
    """Internally derived polynomial flow and variational ODE, without callbacks."""

    def __init__(self, request: StoppedEventRequest) -> None:
        self.request = request
        self.d = d = request.flow.dimension
        self.m = m = request.flow.parameter_count
        self.level = request.derivative_order
        self.f = request.flow.extended_components()
        self.df = tuple(tuple(p.derivative(j) for j in range(d)) for p in self.f)
        self.ddf = tuple(tuple(tuple(p.derivative(k) for k in range(d)) for p in row)
                         for row in self.df)
        self.size = d * (1 + (m if self.level >= 1 else 0) + (m * m if self.level == 2 else 0))

    def si(self, i: int, a: int) -> int:
        return self.d + i * self.m + a

    def qi(self, i: int, a: int, b: int) -> int:
        return self.d * (1 + self.m) + (i * self.m + a) * self.m + b

    def initial(self) -> Box:
        q = self.request
        values = list(polynomial_interval(p, q.parameters) for p in q.initial)
        values.extend(q.parameters)
        values.append(Interval.point(0.0))
        initial = q.initial + tuple(SparsePolynomial.variable(self.m, a) for a in range(self.m)) + (
            SparsePolynomial.constant(self.m, 0),
        )
        if self.level >= 1:
            values.extend(polynomial_interval(p.derivative(a), q.parameters)
                          for p in initial for a in range(self.m))
        if self.level == 2:
            values.extend(polynomial_interval(p.derivative(a).derivative(b), q.parameters)
                          for p in initial for a in range(self.m) for b in range(self.m))
        return tuple(values)

    def field(self, values: list[TaylorSeries]) -> list[TaylorSeries]:
        d, m = self.d, self.m
        w = values[:d]
        out = [_series_eval(p, w) for p in self.f]
        if self.level == 0 or m == 0:
            return out
        jac = [[_series_eval(p, w) for p in row] for row in self.df]
        zero = TaylorSeries.constant(0, values[0].order)
        for i in range(d):
            for a in range(m):
                v = zero
                for j in range(d):
                    if self.df[i][j].terms:
                        v = v + jac[i][j] * values[self.si(j, a)]
                out.append(v)
        if self.level == 2:
            hess = [[[ _series_eval(p, w) for p in row] for row in mat] for mat in self.ddf]
            for i in range(d):
                for a in range(m):
                    for b in range(m):
                        v = zero
                        for j in range(d):
                            if self.df[i][j].terms:
                                v = v + jac[i][j] * values[self.qi(j, a, b)]
                            for k in range(d):
                                if self.ddf[i][j][k].terms:
                                    v = v + hess[i][j][k] * values[self.si(j, a)] * values[self.si(k, b)]
                        out.append(v)
        return out

    def clamp_constants(self, box: Sequence[Interval], time: Interval) -> Box:
        """Exact frozen-parameter and clock identities reduce interval wrapping."""
        out = list(box)
        n = self.request.flow.state_count
        for a, value in enumerate(self.request.parameters):
            out[n + a] = out[n + a].intersect(value)
        out[self.d - 1] = out[self.d - 1].intersect(time)
        for i in range(n, self.d):
            if self.level >= 1:
                for a in range(self.m):
                    out[self.si(i, a)] = out[self.si(i, a)].intersect(Interval.point(float(i == n + a)))
            if self.level == 2:
                for a in range(self.m):
                    for b in range(self.m):
                        out[self.qi(i, a, b)] = out[self.qi(i, a, b)].intersect(Interval.point(0.0))
        return tuple(out)


def _time(request: StoppedEventRequest, index: int) -> Interval:
    return Interval.from_rational(Fraction(request.step) * index)


def _advance(system: _JetSystem, start: Box, index: int) -> tuple[Box, Box]:
    q = system.request
    field: VectorField = system.field
    whole_time = Interval.hull(_time(q, index), _time(q, index + 1))
    tube = system.clamp_constants(_apriori_enclosure(field, start, q.step), whole_time)
    if not _finite(tube):
        raise ArithmeticError("nonfinite flow enclosure")
    coefficients = _solution_coeffs(field, start, q.order)
    remainder = _solution_coeffs(field, tube, q.order + 1)
    powers = [Interval.point(q.step).pow_int(k) for k in range(q.order + 2)]
    end = []
    for i in range(system.size):
        v = _dot(coefficients[i], powers[:q.order + 1])
        end.append(v + remainder[i][-1] * powers[-1])
    result = system.clamp_constants(end, _time(q, index + 1))
    if not _finite(result):
        raise ArithmeticError("nonfinite flow endpoint")
    return tube, result


def _lie(event: PolynomialEvent, flow: PolynomialFlow) -> SparsePolynomial:
    out = SparsePolynomial.constant(flow.dimension, 0, budget=event.polynomial.budget)
    for j, f in enumerate(flow.extended_components()):
        out = out + event.polynomial.derivative(j) * f
    return out


def _no_zero(start: Interval, end: Interval, tube: Interval, velocity: Interval) -> bool:
    return _sign(tube) != 0 or (_sign(velocity) != 0 and _sign(start) == _sign(end) != 0)


def _contract_event(event: PolynomialEvent, box: Box, state_count: int) -> Box:
    # Coordinate interval Newton: every event zero in the input box survives.
    out = list(box)
    for axis in range(state_count):
        slope = polynomial_interval(event.polynomial.derivative(axis), out)
        if _sign(slope):
            mid = out[axis].mid
            point = list(out)
            point[axis] = Interval.point(mid)
            out[axis] = out[axis].intersect(mid - polynomial_interval(event.polynomial, point) / slope)
    return tuple(out)


def _event_derivatives(
    system: _JetSystem, tube: Box, velocity: Interval,
) -> tuple[Box | None, Matrix | None, Matrix | None, Tensor | None]:
    if system.level == 0:
        return None, None, None, None
    q, d, m = system.request, system.d, system.m
    w = tube[:d]
    f = tuple(polynomial_interval(p, w) for p in system.f)
    jac = tuple(tuple(polynomial_interval(p, w) for p in row) for row in system.df)
    g = q.target.polynomial
    grad = tuple(polynomial_interval(g.derivative(j), w) for j in range(d))
    s = tuple(tuple(tube[system.si(i, a)] for i in range(d)) for a in range(m))
    time_grad = tuple(-_dot(grad, sa) / velocity for sa in s)
    return_jac = tuple(tuple(s[a][i] + f[i] * time_grad[a] for a in range(m))
                       for i in range(q.flow.state_count))
    if system.level == 1:
        return time_grad, None, return_jac, None
    hess = tuple(tuple(polynomial_interval(g.derivative(j).derivative(k), w)
                       for k in range(d)) for j in range(d))
    lie_grad = tuple(polynomial_interval(_lie(q.target, q.flow).derivative(j), w) for j in range(d))
    etheta_t = tuple(_dot(lie_grad, sa) for sa in s)
    ett = _dot(lie_grad, f)
    jf = tuple(_dot(row, f) for row in jac)
    js = tuple(tuple(_dot(row, sa) for row in jac) for sa in s)
    time_hess = []
    for a in range(m):
        row = []
        for b in range(m):
            qab = tuple(tube[system.qi(i, a, b)] for i in range(d))
            eab = _dot(grad, qab) + _dot(s[a], tuple(_dot(hr, s[b]) for hr in hess))
            row.append(-(eab + etheta_t[a] * time_grad[b] + etheta_t[b] * time_grad[a]
                         + ett * time_grad[a] * time_grad[b]) / velocity)
        time_hess.append(tuple(row))
    return_hess = tuple(tuple(tuple(
        tube[system.qi(i, a, b)] + js[a][i] * time_grad[b] + js[b][i] * time_grad[a]
        + jf[i] * time_grad[a] * time_grad[b] + f[i] * time_hess[a][b]
        for b in range(m)) for a in range(m)) for i in range(q.flow.state_count))
    return time_grad, tuple(time_hess), return_jac, return_hess


def certify_stopped_event(request: StoppedEventRequest) -> StoppedEventResult:
    """Prove a regular first positive event uniformly on the exact source box.

    The algorithm never advances past an uncertified possible earlier event.
    A partial transverse bracket can span consecutive slabs, but its normal
    velocity must keep one strict sign throughout. Competing events must be
    absent on the whole bracket, a conservative sufficient stopping rule.
    """
    fingerprint = request.fingerprint
    slabs: list[EventSlab] = []

    def finish(status: Status, reason: str) -> StoppedEventResult:
        return StoppedEventResult(request, fingerprint, status, reason, tuple(slabs))

    try:
        system = _JetSystem(request)
        state = system.initial()
        initial_events = tuple(_initial_event(request, e) for e in request.events)
        values = tuple(polynomial_interval(p, request.parameters) if p.terms else Interval.point(0.0)
                       for p in initial_events)
        departure = not initial_events[0].terms
        if not departure and _sign(values[0]) == 0:
            return finish("unresolved", "initial target incidence is neither excluded nor an exact identity")
        initial_guards = tuple(
            polynomial_interval(_initial_event(request, PolynomialEvent(e.guard)), request.parameters)
            if e.guard is not None else Interval.point(1.0) for e in request.events
        )
        if any(_sign(v) == 0 and guard.hi > 0
               for v, guard in zip(values[1:], initial_guards[1:], strict=True)):
            return finish("unresolved", "a competing event is not excluded at the initial source")
        lies = tuple(_lie(e, request.flow) for e in request.events)
        pending: Box | None = None
        pending_index = 0
        pending_sign = 0
        pending_velocity: Interval | None = None
        for index in range(request.max_steps):
            tube, end = _advance(system, state, index)
            w = tube[:system.d]
            normal = tuple(polynomial_interval(p, w) for p in lies)
            ends = tuple(polynomial_interval(e.polynomial, end[:system.d]).intersect(
                old + Interval.point(request.step) * v)
                for e, old, v in zip(request.events, values, normal, strict=True))
            ranges = tuple(polynomial_interval(e.polynomial, w).intersect(
                old + Interval(0.0, request.step) * v)
                for e, old, v in zip(request.events, values, normal, strict=True))
            guards = tuple(polynomial_interval(e.guard, w) if e.guard is not None else Interval.point(1.0)
                           for e in request.events)
            action = "excluded"
            competing = any(guard.hi > 0 and not _no_zero(a, b, c, v) for a, b, c, v, guard in
                            zip(values[1:], ends[1:], ranges[1:], normal[1:], guards[1:], strict=True))
            if competing:
                action = "competing_event_unresolved"
            elif pending is not None:
                action = "bracket"
            elif guards[0].hi <= 0:
                action = "guard_exclusion"
            elif departure:
                action = "initial_departure" if _sign(normal[0]) else "departure_unresolved"
            elif not _no_zero(values[0], ends[0], ranges[0], normal[0]):
                action = "bracket"
                pending_index, pending_sign = index, _sign(values[0])
                pending = tube
                pending_velocity = normal[0]
            slab_time = Interval.hull(_time(request, index), _time(request, index + 1))
            slabs.append(EventSlab(slab_time, tuple(w[:request.flow.state_count]), values,
                                   ends, ranges, normal, guards, action))
            if competing:
                return finish("unresolved", "a competing event may occur before the target")
            if action == "departure_unresolved":
                return finish("unresolved", "initial target departure is not strictly transverse")
            if action == "bracket":
                if pending is None or pending_velocity is None:
                    raise ArithmeticError("missing bracket state")
                if guards[0].lo <= 0:
                    return finish("unresolved", "target guard is not strictly positive on the possible first event")
                pending = tuple(Interval.hull(a, b) for a, b in zip(pending, tube, strict=True))
                pending_velocity = Interval.hull(pending_velocity, normal[0])
                direction = _sign(pending_velocity)
                if direction == 0 or direction != -pending_sign:
                    return finish("unresolved", "possible first event is not certified transverse and unique")
                if request.target.direction and direction != request.target.direction:
                    return finish("unresolved", "possible first target zero has the opposite requested direction")
                if _sign(ends[0]) == -pending_sign:
                    contracted = _contract_event(request.target, pending[:system.d], request.flow.state_count)
                    event_tube = contracted + pending[system.d:]
                    velocity = polynomial_interval(lies[0], contracted).intersect(pending_velocity)
                    dt, ddt, dy, ddy = _event_derivatives(system, event_tube, velocity)
                    bracket = Interval.hull(_time(request, pending_index), _time(request, index + 1))
                    return StoppedEventResult(
                        request, fingerprint, "certified", "unique transverse first positive event; competitors excluded",
                        tuple(slabs), bracket, contracted[:request.flow.state_count], velocity, dt, ddt, dy, ddy,
                    )
            departure = False
            state, values = end, ends
        if pending is not None:
            return finish("unresolved", "finite horizon ended before a uniform opposite endpoint sign")
        return finish("excluded", "all positive target zeros excluded on the checked finite horizon")
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        return finish("unresolved", f"validated computation stopped: {exc}")


def verify_stopped_event(result: StoppedEventResult) -> bool:
    """Recompute from exact sources and compare all bounds; reject tampered claims."""
    if not result.certified or result.source_fingerprint != result.request.fingerprint:
        return False
    return certify_stopped_event(result.request) == result


__all__ = [
    "EventSlab",
    "PolynomialEvent",
    "PolynomialFlow",
    "StoppedEventRequest",
    "StoppedEventResult",
    "certify_stopped_event",
    "polynomial_interval",
    "verify_stopped_event",
]
