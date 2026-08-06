# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hard Dirichlet / Neumann / Robin / initial conditions for the JAX backend.

JAX twin of :mod:`omnibias.pinn.torch.cage.constrained`. Both cages import the
switching coefficients from :mod:`omnibias.pinn._core.constrained` and walk the
same recursion in the same axis order, so the two backends agree to round-off by
construction rather than by testing.

The field is pytree-registered: the base field's leaves carry the trainable
parameters, and the constraint plan travels in the (hashable) auxiliary tree-def
so ``jax.jit`` can cache on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.constrained import (
    AxisPlan,
    CornerPair,
    HardCondition,
    apply_constraint,
    certify_support_matrix,
    compatibility_sample,
    corner_pairs,
    face_point,
    group_hard_conditions,
    is_relative,
    projection_cost,
    switching_derivative_coeffs,
)
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.cage.incompressible import _CageFieldBase
from omnibias.pinn.jax.fields.base import FieldBase

_CACHE_KEY = "_constrained_cache"

Pins = tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class ConstrainedExpressionField(_CageFieldBase):
    r"""Cage enforcing linear boundary / initial conditions by construction.

    See :class:`omnibias.pinn.torch.cage.constrained.ConstrainedExpressionField`
    for the construction, the two preconditions it checks, and when to prefer it
    over the distance-function :class:`HardBoundaryField`. Build one with
    :func:`make_constrained_expression_field`.
    """

    base: FieldBase
    plans: tuple[tuple[str, tuple[AxisPlan, ...]], ...]
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    bounds: tuple[tuple[float, float], ...]
    compatibility_tol: float
    certified: bool

    # The constrained expression is *affine* in the base readout, so forwarding
    # the parameter arrays lets every caller that treats the readout as the
    # linear unknown keep working when a cage is wrapped around the ansatz.

    @property
    def W(self) -> Array:
        """The base field's hidden weights."""
        return self.base.W

    @property
    def beta(self) -> Array:
        """The base field's hidden biases."""
        return self.base.beta

    @property
    def c(self) -> Array:
        """The base field's readout weights."""
        return self.base.c

    @property
    def b(self) -> Array:
        """The base field's readout biases."""
        return self.base.b

    def steps_for(self, name: str) -> tuple[AxisPlan, ...]:
        """The axis recursion steps for one constrained component."""
        for key, steps in self.plans:
            if key == name:
                return steps
        raise KeyError(f"{name!r} carries no hard conditions")

    def support_certificates(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Sealed certificates that every constrained axis admits an exact ansatz."""
        return {
            name: tuple(
                certify_support_matrix(
                    step.constraints,
                    claim=(
                        f"hard conditions on component {name!r}, axis "
                        f"{step.constraints.axis}, admit an exact constrained "
                        "expression"
                    ),
                )
                for step in steps
            )
            for name, steps in self.plans
        }

    # ----- per-evaluation caching ------------------------------------

    def _pinned_state(self, inner: FieldState, pins: Pins) -> FieldState:
        """The base field evaluated with the given axes pinned to face values.

        Cached on the state because a projected :class:`FieldState` caches only
        ``sigma^(n)(z)``, which does not depend on the readout.
        """
        if not pins:
            return inner
        states = inner.extra.setdefault(_CACHE_KEY, {})
        hit = states.get(pins)
        if hit is None:
            coords = inner.coords
            for axis, value in pins:
                coords = coords.at[:, axis].set(value)
            hit = self.base.evaluate(coords)
            states[pins] = hit
        return hit  # type: ignore[return-value]

    def _base_partial(
        self, inner: FieldState, pins: Pins, name: str, orders: tuple[int, ...]
    ) -> Array:
        state = self._pinned_state(inner, pins)
        axes = tuple(a for a, o in enumerate(orders) if o)
        active = tuple(o for o in orders if o)
        if not axes:
            return state.ops.value(state, name)
        if len(axes) == 1:
            return state.ops.derivative(state, name, axis=axes[0], order=active[0])
        return state.ops.mixed_partial(state, name, axes, active)

    # ----- targets ----------------------------------------------------

    def _target_partial(
        self, coords: Array, target: Any, pins: Pins, orders: tuple[int, ...]
    ) -> Array:
        zeros = jnp.zeros((coords.shape[0],), dtype=coords.dtype)
        if not callable(target):
            if any(orders):
                return zeros
            return zeros + float(target)

        pinned = coords
        for axis, value in pins:
            pinned = pinned.at[:, axis].set(value)
        if not any(orders):
            return jnp.asarray(target(pinned), dtype=coords.dtype) + zeros

        def one(c: Array) -> Array:
            return jnp.asarray(target(c[None, :]), dtype=coords.dtype)[0]

        cur = one
        for axis, order in enumerate(orders):
            for _ in range(order):
                cur = (
                    lambda prev=cur, ax=axis: (lambda c: jax.grad(prev)(c)[ax])
                )()
        return jax.vmap(cur)(pinned)

    # ----- the recursion ----------------------------------------------

    def _level(
        self,
        inner: FieldState,
        name: str,
        depth: int,
        pins: Pins,
        orders: tuple[int, ...],
        memo: dict[Any, Array],
    ) -> Array:
        """``D^orders`` of the expression after the first ``depth`` axis steps.

        ``memo`` is scoped to one op call rather than to the state, because the
        values depend on the readout while a state may be reused across several.
        """
        if depth == 0:
            return self._base_partial(inner, pins, name, orders)
        key = (name, depth, pins, orders)
        hit = memo.get(key)
        if hit is not None:
            return hit

        step = self.steps_for(name)[depth - 1]
        axis = step.constraints.axis
        out = self._level(inner, name, depth - 1, pins, orders, memo)
        rest = tuple(0 if a == axis else o for a, o in enumerate(orders))
        xi = self._normalized(inner, step, pins)

        for i, constraint in enumerate(step.constraints.constraints):
            phi = self._switching(step, i, orders[axis], xi)
            if phi is None:
                continue
            point = face_point(constraint)
            face: Pins = pins if point is None else _with_pin(pins, axis, point)
            rho = self._target_partial(inner.coords, step.targets[i], face, rest)
            rho = rho - apply_constraint(
                constraint,
                lambda point, order, _rest=rest: self._level(  # type: ignore[misc]
                    inner,
                    name,
                    depth - 1,
                    _with_pin(pins, axis, point),
                    tuple(order if a == axis else o for a, o in enumerate(_rest)),
                    memo,
                ),
            )
            out = out + phi * rho

        memo[key] = out
        return out

    def _normalized(self, inner: FieldState, step: AxisPlan, pins: Pins) -> Array:
        axis = step.constraints.axis
        coords = inner.coords
        pinned = dict(pins).get(axis)
        c = step.constraints
        if pinned is not None:
            base = jnp.full((coords.shape[0],), pinned, dtype=coords.dtype)
        else:
            base = coords[:, axis]
        return (base - c.lo) / c.length

    def _switching(
        self, step: AxisPlan, index: int, order: int, xi: Array
    ) -> Array | None:
        coeffs = switching_derivative_coeffs(
            step.constraints.switching[index], order, step.constraints.length
        )
        if not coeffs:
            return None
        out = jnp.zeros_like(xi) + coeffs[0]
        power = jnp.ones_like(xi)
        for c in coeffs[1:]:
            power = power * xi
            out = out + c * power
        return out

    # ----- _CageFieldBase hooks ---------------------------------------

    def _velocity_value(self, inner: FieldState, name: str) -> Array:
        zero = (0,) * self.coordinate_spec.ndim
        return self._level(inner, name, len(self.steps_for(name)), (), zero, {})

    def _velocity_derivative(
        self, inner: FieldState, name: str, *, axis: int, order: int
    ) -> Array:
        orders = tuple(
            order if a == axis else 0 for a in range(self.coordinate_spec.ndim)
        )
        return self._level(inner, name, len(self.steps_for(name)), (), orders, {})

    def _velocity_mixed(
        self,
        inner: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        folded = [0] * self.coordinate_spec.ndim
        for a, o in zip(axes, orders, strict=False):
            folded[a] += int(o)
        return self._level(
            inner, name, len(self.steps_for(name)), (), tuple(folded), {}
        )

    # ----- the cross-axis data gate -----------------------------------

    @property
    def corner_pairs(self) -> tuple[CornerPair, ...]:
        """Every pair of conditions on different axes whose data must agree."""
        return corner_pairs(dict(self.plans))

    @property
    def projection_cost(self) -> int:
        """Base-field evaluations one forward pass costs.

        The product over constrained axes of ``1 + #projection points``, shared
        across components. See
        :func:`omnibias.pinn._core.constrained.projection_cost`.
        """
        return projection_cost(dict(self.plans))

    def _sample(self, coords: Array | None) -> Array:
        if coords is not None:
            return jnp.asarray(coords)
        return jnp.asarray(compatibility_sample(self.bounds), dtype=self.base.c.dtype)

    def _corner_gap(self, coords: Array, pair: CornerPair) -> Array:
        ndim = self.coordinate_spec.ndim
        pin_a = face_point(pair.constraint_a)
        pin_b = face_point(pair.constraint_b)
        base_a: Pins = () if pin_a is None else ((pair.axis_a, pin_a),)
        base_b: Pins = () if pin_b is None else ((pair.axis_b, pin_b),)
        lhs = apply_constraint(
            pair.constraint_a,
            lambda point, order: self._target_partial(
                coords,
                pair.target_b,
                _with_pin(base_b, pair.axis_a, point),
                tuple(order if a == pair.axis_a else 0 for a in range(ndim)),
            ),
        )
        rhs = apply_constraint(
            pair.constraint_b,
            lambda point, order: self._target_partial(
                coords,
                pair.target_a,
                _with_pin(base_a, pair.axis_b, point),
                tuple(order if a == pair.axis_b else 0 for a in range(ndim)),
            ),
        )
        return lhs - rhs

    def worst_corner(self, coords: Array | None = None) -> tuple[float, CornerPair | None]:
        """The largest cross-axis data disagreement, and which pair carries it."""
        sample = self._sample(coords)
        worst = 0.0
        culprit: CornerPair | None = None
        for pair in self.corner_pairs:
            gap = float(jnp.max(jnp.abs(self._corner_gap(sample, pair))))
            if gap >= worst:
                worst, culprit = gap, pair
        return worst, culprit

    def compatibility_residual(self, coords: Array | None = None) -> float:
        """How badly the condition *data* disagrees where two axes meet.

        Order one when the disagreement is real, round-off when it is not; see
        the torch twin for what the quantity means.
        """
        return self.worst_corner(coords)[0]

    def check_compatibility(self, coords: Array | None = None) -> None:
        """Raise when the condition data is inconsistent across axes."""
        residual, pair = self.worst_corner(coords)
        if residual > self.compatibility_tol:
            where = f" ({pair.label})" if pair is not None else ""
            raise ValueError(
                f"hard conditions are mutually inconsistent: residual {residual:.3e} "
                f"exceeds {self.compatibility_tol:.3e}{where}. The conditions on "
                "different axes disagree where those axes meet (typically the "
                "initial state does not satisfy the boundary condition at t0); no "
                "ansatz can satisfy both, so the data must be reconciled first"
            )

    def condition_residual(self, coords: Array) -> float:
        """How far the built expression is from the conditions it claims to enforce.

        The falsifier for the construction itself; see the torch twin.
        """
        worst = 0.0
        state = self.base.evaluate(jnp.asarray(coords))
        memo: dict[Any, Array] = {}
        ndim = self.coordinate_spec.ndim
        for name, steps in self.plans:
            depth = len(steps)
            for step in steps:
                axis = step.constraints.axis
                for i, constraint in enumerate(step.constraints.constraints):
                    got = apply_constraint(
                        constraint,
                        lambda point, order, _a=axis, _n=name, _d=depth: self._level(  # type: ignore[misc]
                            state,
                            _n,
                            _d,
                            ((_a, point),),
                            tuple(order if a == _a else 0 for a in range(ndim)),
                            memo,
                        ),
                    )
                    point = face_point(constraint)
                    face: Pins = () if point is None else ((axis, point),)
                    want = self._target_partial(
                        state.coords, step.targets[i], face, (0,) * ndim
                    )
                    worst = max(worst, float(jnp.max(jnp.abs(got - want))))
        return worst


def _with_pin(pins: Pins, axis: int, value: float) -> Pins:
    merged = {a: v for a, v in pins}
    merged[axis] = value
    return tuple(sorted(merged.items()))


def make_constrained_expression_field(
    *,
    base: FieldBase,
    conditions: Sequence[HardCondition],
    bounds: Sequence[tuple[float, float]] | None = None,
    passthrough_names: tuple[str, ...] = (),
    groups: dict[str, tuple[str, ...]] | None = None,
    certify: bool = True,
    check_data: bool = True,
    compatibility_tol: float = 1e-8,
) -> ConstrainedExpressionField:
    """Build a JAX constrained-expression cage around ``base``."""
    if not conditions:
        raise ValueError(
            "ConstrainedExpressionField needs at least one condition; an "
            "unconstrained field should be used directly"
        )
    resolved = bounds if bounds is not None else base.coordinate_spec.domain
    if resolved is None:
        raise ValueError(
            "hard conditions need per-axis (lo, hi) bounds: pass `bounds=` or "
            "build the base field with a CoordinateSpec carrying `domain=`"
        )
    for cond in conditions:
        if is_relative(cond.constraint) and callable(cond.target):
            raise ValueError(
                f"relative constraint {cond.constraint.label!r} ties several "
                "points together, so its target belongs to no single face and "
                "must be a constant"
            )
    plans = group_hard_conditions(conditions, tuple(resolved))
    constrained = tuple(plans)
    overlap = set(constrained) & set(passthrough_names)
    if overlap:
        raise ValueError(
            f"components {sorted(overlap)!r} are both constrained and passthrough"
        )
    field = ConstrainedExpressionField(
        base=base,
        plans=tuple((name, plans[name]) for name in constrained),
        velocity_names=constrained,
        passthrough_names=tuple(passthrough_names),
        coordinate_spec=base.coordinate_spec,
        components=ComponentSpec(
            constrained + tuple(passthrough_names),
            groups=groups if groups is not None else {"constrained": constrained},
        ),
        bounds=tuple((float(lo), float(hi)) for lo, hi in resolved),
        compatibility_tol=float(compatibility_tol),
        certified=bool(certify),
    )
    if certify:
        field.support_certificates()
    if check_data and field.corner_pairs:
        field.check_compatibility()
    return field


def _ce_flatten(f: ConstrainedExpressionField):  # type: ignore[no-untyped-def]
    return (f.base,), (
        f.plans,
        f.velocity_names,
        f.passthrough_names,
        f.coordinate_spec,
        f.components,
        f.bounds,
        f.compatibility_tol,
        f.certified,
    )


def _ce_unflatten(aux, leaves):  # type: ignore[no-untyped-def]
    (base,) = leaves
    (
        plans,
        velocity_names,
        passthrough_names,
        coordinate_spec,
        components,
        bounds,
        compatibility_tol,
        certified,
    ) = aux
    obj = ConstrainedExpressionField.__new__(ConstrainedExpressionField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "plans", plans)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "bounds", bounds)
    object.__setattr__(obj, "compatibility_tol", compatibility_tol)
    object.__setattr__(obj, "certified", certified)
    return obj


jax.tree_util.register_pytree_node(
    ConstrainedExpressionField, _ce_flatten, _ce_unflatten
)


__all__ = [
    "ConstrainedExpressionField",
    "make_constrained_expression_field",
]
