# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Conservation-law cage layers for the JAX backend.

JAX twins of the torch :mod:`omnibias.pinn.torch.cage.conservation`
helpers and fields.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.cage.incompressible import (
    VectorPotentialField,
    _CageFieldBase,
    make_vector_potential_field,
)
from omnibias.pinn.jax.fields.base import FieldBase

if TYPE_CHECKING:  # pragma: no cover
    pass


# ----- skew-symmetric advection ------------------------------------


def energy_conserving_advection(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
) -> Array:
    sa = state.coordinate_spec.spatial_axes
    if len(velocity) != len(sa):
        raise ValueError(
            f"energy_conserving_advection: velocity length {len(velocity)} "
            f"!= n_spatial {len(sa)}"
        )
    ops = state.ops
    standard = ops.advection(state, velocity=velocity)
    div_u = ops.divergence(state, velocity)
    u_i = ops.stack_components(state, velocity)
    return standard + 0.5 * div_u[..., None] * u_i


def enstrophy_conserving_advection(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    vorticity: str,
) -> Array:
    ops = state.ops
    standard = ops.advection(state, velocity=velocity, scalar=vorticity)
    div_u = ops.divergence(state, velocity)
    omega = ops.value(state, vorticity)
    return standard + 0.5 * div_u * omega


# ----- Hard boundary cage --------------------------------------------


@dataclass(frozen=True)
class HardBoundaryField(_CageFieldBase):
    """JAX hard-boundary cage. ``u(x, t) = g(x, t) + d(x) f_NN(x, t)``."""

    base: FieldBase
    distance_fn: Callable[[Array], Array]
    boundary_value_fn: Callable[[Array], dict[str, Array]] | None
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    max_derivative_order: int = 4

    def _g_value(self, name: str, coords: Array) -> Array:
        if self.boundary_value_fn is None:
            return jnp.zeros((coords.shape[0],), dtype=coords.dtype)
        g = self.boundary_value_fn(coords)
        return g.get(name, jnp.zeros((coords.shape[0],), dtype=coords.dtype))

    def _g_partial(
        self, name: str, coords: Array, axis: int, order: int = 1,
    ) -> Array:
        if self.boundary_value_fn is None:
            return jnp.zeros((coords.shape[0],), dtype=coords.dtype)

        def f_one(c_one):
            g = self.boundary_value_fn(c_one[None, :])
            v = g.get(name)
            if v is None:
                return jnp.float64(0.0)
            return v[0]

        cur = f_one
        for _ in range(order):
            prev = cur
            cur = (lambda prev_fn=prev, ax=axis:
                   (lambda c: jax.grad(prev_fn)(c)[ax]))()
        return jax.vmap(cur)(coords)

    def _d_value(self, coords: Array) -> Array:
        return self.distance_fn(coords)

    def _d_partial(
        self, coords: Array, axis: int, order: int = 1,
    ) -> Array:
        def f_one(c_one):
            return self.distance_fn(c_one[None, :])[0]
        cur = f_one
        for _ in range(order):
            prev = cur
            cur = (lambda prev_fn=prev, ax=axis:
                   (lambda c: jax.grad(prev_fn)(c)[ax]))()
        return jax.vmap(cur)(coords)

    def _d_mixed(
        self, coords: Array, axes_orders: dict[int, int],
    ) -> Array:
        def f_one(c_one):
            return self.distance_fn(c_one[None, :])[0]
        cur = f_one
        for ax, ord_ in axes_orders.items():
            for _ in range(ord_):
                prev = cur
                cur = (lambda prev_fn=prev, axx=ax:
                       (lambda c: jax.grad(prev_fn)(c)[axx]))()
        return jax.vmap(cur)(coords)

    def _g_mixed(
        self, name: str, coords: Array, axes_orders: dict[int, int],
    ) -> Array:
        if self.boundary_value_fn is None:
            return jnp.zeros((coords.shape[0],), dtype=coords.dtype)

        def f_one(c_one):
            g = self.boundary_value_fn(c_one[None, :])
            v = g.get(name)
            if v is None:
                return jnp.float64(0.0)
            return v[0]
        cur = f_one
        for ax, ord_ in axes_orders.items():
            for _ in range(ord_):
                prev = cur
                cur = (lambda prev_fn=prev, axx=ax:
                       (lambda c: jax.grad(prev_fn)(c)[axx]))()
        return jax.vmap(cur)(coords)

    def _velocity_value(self, inner: FieldState, name: str) -> Array:
        coords = inner.coords
        g = self._g_value(name, coords)
        d = self._d_value(coords)
        f = inner.ops.value(inner, name)
        return g + d * f

    def _velocity_derivative(
        self, inner: FieldState, name: str, *, axis: int, order: int,
    ) -> Array:
        coords = inner.coords
        out = self._g_partial(name, coords, axis=axis, order=order)
        for j in range(order + 1):
            d_j = (
                self._d_value(coords) if j == 0 else
                self._d_partial(coords, axis=axis, order=j)
            )
            f_kj = (
                inner.ops.value(inner, name) if (order - j) == 0 else
                inner.ops.derivative(
                    inner, name, axis=axis, order=order - j,
                )
            )
            out = out + comb(order, j) * d_j * f_kj
        return out

    def _velocity_mixed(
        self,
        inner: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        folded: dict[int, int] = {}
        for a, o in zip(axes, orders, strict=False):
            folded[a] = folded.get(a, 0) + int(o)
        coords = inner.coords
        out_g = self._g_mixed(name, coords, folded)
        terms: list[tuple[dict[int, int], dict[int, int], int]] = [
            ({}, {}, 1),
        ]
        for ax, total in folded.items():
            new_terms: list[tuple[dict[int, int], dict[int, int], int]] = []
            for d_o, f_o, mult in terms:
                for j in range(total + 1):
                    new_d_o = dict(d_o)
                    new_f_o = dict(f_o)
                    if j > 0:
                        new_d_o[ax] = new_d_o.get(ax, 0) + j
                    if (total - j) > 0:
                        new_f_o[ax] = new_f_o.get(ax, 0) + (total - j)
                    new_terms.append((new_d_o, new_f_o, mult * comb(total, j)))
            terms = new_terms
        out_df = jnp.zeros_like(out_g)
        for d_o, f_o, mult in terms:
            d_part = self._d_value(coords) if not d_o else self._d_mixed(coords, d_o)
            if not f_o:
                f_part = inner.ops.value(inner, name)
            else:
                ax_t = tuple(f_o.keys())
                og_t = tuple(f_o[a] for a in ax_t)
                f_part = inner.ops.mixed_partial(inner, name, ax_t, og_t)
            out_df = out_df + mult * d_part * f_part
        return out_g + out_df


def make_hard_boundary_field(
    *,
    base: FieldBase,
    distance_fn: Callable[[Array], Array],
    boundary_value_fn: Callable[[Array], dict[str, Array]] | None = None,
    bounded_names: tuple[str, ...] | None = None,
    passthrough_names: tuple[str, ...] = (),
    groups: dict[str, tuple[str, ...]] | None = None,
    max_derivative_order: int = 4,
) -> HardBoundaryField:
    if bounded_names is None:
        bounded_names = tuple(
            n for n in base.components.names if n not in passthrough_names
        )
    if groups is None:
        groups = {}
    component_names = tuple(bounded_names) + tuple(passthrough_names)
    components = ComponentSpec(component_names, groups=groups or None)
    return HardBoundaryField(
        base=base,
        distance_fn=distance_fn,
        boundary_value_fn=boundary_value_fn,
        velocity_names=tuple(bounded_names),
        passthrough_names=tuple(passthrough_names),
        coordinate_spec=base.coordinate_spec,
        components=components,
        max_derivative_order=int(max_derivative_order),
    )


def _hb_flatten(f: HardBoundaryField):
    return (f.base,), (
        f.distance_fn,
        f.boundary_value_fn,
        f.velocity_names,
        f.passthrough_names,
        f.coordinate_spec,
        f.components,
        f.max_derivative_order,
    )


def _hb_unflatten(aux, leaves):
    base, = leaves
    (
        distance_fn, boundary_value_fn, velocity_names, passthrough_names,
        coordinate_spec, components, max_derivative_order,
    ) = aux
    obj = HardBoundaryField.__new__(HardBoundaryField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "distance_fn", distance_fn)
    object.__setattr__(obj, "boundary_value_fn", boundary_value_fn)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "max_derivative_order", max_derivative_order)
    return obj


jax.tree_util.register_pytree_node(
    HardBoundaryField, _hb_flatten, _hb_unflatten,
)


# ----- mass-flux potential alias ------------------------------------


def make_mass_flux_potential_field(
    *,
    base: FieldBase,
    Psi_components: tuple[str, str, str] = ("Psi1", "Psi2", "Psi3"),
    flux_names: tuple[str, str, str] = ("rhou", "rhov", "rhow"),
    passthrough_names: tuple[str, ...] = ("rho",),
    spatial_axes: tuple[str, str, str] = ("x", "y", "z"),
) -> VectorPotentialField:
    """Build a :class:`VectorPotentialField` configured for mass-flux."""
    return make_vector_potential_field(
        base=base,
        A_components=Psi_components,
        velocity_names=flux_names,
        passthrough_names=passthrough_names,
        spatial_axes=spatial_axes,
    )


__all__ = [
    "HardBoundaryField",
    "energy_conserving_advection",
    "enstrophy_conserving_advection",
    "make_hard_boundary_field",
    "make_mass_flux_potential_field",
]
