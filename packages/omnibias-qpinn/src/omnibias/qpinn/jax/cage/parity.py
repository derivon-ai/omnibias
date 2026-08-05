# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hard parity-projection cage (JAX backend).

JAX twin of :mod:`omnibias.qpinn.torch.cage.parity`. See the torch
docstring for the underlying math; here we only describe the
pytree-registration plumbing.

A :class:`ParityProjectedField` is a frozen dataclass with **one**
pytree leaf -- ``base`` -- and static metadata
``(parity_sign_value, mirror_axis, velocity_names,
passthrough_names, coordinate_spec, components)`` in the
auxiliary tree-def. This lets ``jax.grad`` / ``jax.jit`` see the
trainable base parameters through the cage transparently while keeping
the discrete parity choice static (changing parity at trace time would
require a re-compile, which is appropriate behaviour).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.cage.incompressible import _CageFieldBase
from omnibias.pinn.jax.fields.base import FieldBase, _import_jax_ops

_INV_SQRT2: float = 0.7071067811865475244008443621048


def _parity_sign(parity: str) -> float:
    p = parity.lower()
    if p in ("+", "even", "symmetric", "+1"):
        return +1.0
    if p in ("-", "odd", "antisymmetric", "-1"):
        return -1.0
    raise ValueError(f"unknown parity {parity!r}; use 'even' or 'odd'")


@dataclass(frozen=True)
class ParityProjectedField(_CageFieldBase):
    r"""Hard parity-projection cage for symmetric / antisymmetric eigenstates (JAX).

    See :class:`omnibias.qpinn.torch.cage.parity.ParityProjectedField`
    for the full mathematical description and motivation.
    """

    base: FieldBase
    parity_sign_value: float
    mirror_axis: int
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec

    def _mirror_coords(self, coords: Array) -> Array:
        sign = jnp.ones_like(coords)
        sign = sign.at[..., self.mirror_axis].set(-1.0)
        return coords * sign

    def evaluate(self, coords: Array) -> FieldState[Array]:
        coords = jnp.asarray(coords)
        if coords.ndim != 2:
            raise ValueError(
                f"coords must be 2D (B, D), got shape {tuple(coords.shape)}"
            )
        if coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"coords last dim {coords.shape[-1]} != "
                f"coordinate_spec.ndim {self.coordinate_spec.ndim}"
            )
        state_pos = self.base.evaluate(coords)
        state_neg = self.base.evaluate(self._mirror_coords(coords))
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_jax_ops(),
            sigma_cache=state_pos.sigma_cache,
            extra={
                "_parity_state_pos": state_pos,
                "_parity_state_neg": state_neg,
            },
        )

    def _is_projected(self, name: str) -> bool:
        return name in self.velocity_names

    def _is_passthrough(self, name: str) -> bool:
        return name in self.passthrough_names

    def value_component(self, state: FieldState, name: str) -> Array:
        s_pos = state.extra["_parity_state_pos"]
        s_neg = state.extra["_parity_state_neg"]
        v_pos = s_pos.ops.value(s_pos, name)
        if self._is_projected(name):
            v_neg = s_neg.ops.value(s_neg, name)
            return _INV_SQRT2 * (v_pos + self.parity_sign_value * v_neg)
        if self._is_passthrough(name):
            return v_pos
        raise KeyError(
            f"{name!r} is neither a projected component "
            f"{self.velocity_names!r} nor a passthrough "
            f"{self.passthrough_names!r}"
        )

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Array:
        if order == 0:
            return self.value_component(state, name)
        s_pos = state.extra["_parity_state_pos"]
        s_neg = state.extra["_parity_state_neg"]
        d_pos = s_pos.ops.derivative(s_pos, name, axis=axis, order=order)
        if self._is_projected(name):
            d_neg = s_neg.ops.derivative(s_neg, name, axis=axis, order=order)
            sign_neg = self.parity_sign_value
            if axis == self.mirror_axis and (order % 2) == 1:
                sign_neg = -sign_neg
            return _INV_SQRT2 * (d_pos + sign_neg * d_neg)
        if self._is_passthrough(name):
            return d_pos
        raise KeyError(name)

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        s_pos = state.extra["_parity_state_pos"]
        s_neg = state.extra["_parity_state_neg"]
        d_pos = s_pos.ops.mixed_partial(s_pos, name, axes, orders)
        if self._is_projected(name):
            d_neg = s_neg.ops.mixed_partial(s_neg, name, axes, orders)
            mirror_order = 0
            for a, o in zip(axes, orders, strict=False):
                if a == self.mirror_axis:
                    mirror_order += int(o)
            sign_neg = self.parity_sign_value
            if (mirror_order % 2) == 1:
                sign_neg = -sign_neg
            return _INV_SQRT2 * (d_pos + sign_neg * d_neg)
        if self._is_passthrough(name):
            return d_pos
        raise KeyError(name)


def make_parity_projected_field(
    *,
    base: FieldBase,
    parity: str,
    mirror_axis: int = 0,
    projected_names: Sequence[str] | None = None,
) -> ParityProjectedField:
    r"""Functional builder for :class:`ParityProjectedField` (JAX twin)."""
    if mirror_axis < 0 or mirror_axis >= base.coordinate_spec.ndim:
        raise ValueError(
            f"mirror_axis={mirror_axis} out of range for "
            f"coordinate_spec.ndim={base.coordinate_spec.ndim}"
        )
    if projected_names is None:
        projected_names = tuple(base.components.names)
    else:
        projected_names = tuple(projected_names)
        for n in projected_names:
            if not base.components.is_component(n):
                raise ValueError(
                    f"projected_name {n!r} not in base components "
                    f"{base.components.names!r}"
                )
    passthrough = tuple(
        n for n in base.components.names if n not in projected_names
    )
    return ParityProjectedField(
        base=base,
        parity_sign_value=_parity_sign(parity),
        mirror_axis=int(mirror_axis),
        velocity_names=tuple(projected_names),
        passthrough_names=passthrough,
        coordinate_spec=base.coordinate_spec,
        components=base.components,
    )


def _parity_cage_flatten(f: ParityProjectedField):
    leaves = (f.base,)
    aux = (
        f.parity_sign_value,
        f.mirror_axis,
        f.velocity_names,
        f.passthrough_names,
        f.coordinate_spec,
        f.components,
    )
    return leaves, aux


def _parity_cage_unflatten(aux, leaves):
    (base,) = leaves
    (
        parity_sign_value,
        mirror_axis,
        velocity_names,
        passthrough_names,
        coordinate_spec,
        components,
    ) = aux
    obj = ParityProjectedField.__new__(ParityProjectedField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "parity_sign_value", parity_sign_value)
    object.__setattr__(obj, "mirror_axis", mirror_axis)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    return obj


jax.tree_util.register_pytree_node(
    ParityProjectedField, _parity_cage_flatten, _parity_cage_unflatten,
)


__all__ = [
    "ParityProjectedField",
    "make_parity_projected_field",
]
