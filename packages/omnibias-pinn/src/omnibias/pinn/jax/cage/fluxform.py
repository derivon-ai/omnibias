# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Flux-form (finite-volume) conservation cage for the JAX backend.

JAX twin of :mod:`omnibias.pinn.torch.cage.fluxform`; see that module for the
mathematics. An antisymmetric potential ``A^{ij} = -A^{ji}`` yields
``G^i = sum_j d_j A^{ij}`` with ``div G = 0`` identically, in any dimension,
because ``d_i d_j`` is symmetric where ``A^{ij}`` is antisymmetric. With a time
axis among the axes this is a conservation law ``d_t rho + div F = 0`` held
exactly rather than penalised.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.fluxform import antisymmetric_pairs, potential_table
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.cage.incompressible import _CageFieldBase
from omnibias.pinn.jax.fields.base import FieldBase


@dataclass(frozen=True)
class FluxFormField(_CageFieldBase):
    """JAX divergence-form conservation cage; build it with the factory below.

    ``potential`` maps an ordered ``(i, j)`` axis pair to the potential
    component name and its sign, covering both orders so ``A^{ji} = -A^{ij}``
    needs no branching at call time.
    """

    base: FieldBase
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    potential_names: tuple[str, ...]
    flux_axes: tuple[str, ...]
    axis_index: tuple[int, ...]
    potential: dict[tuple[int, int], tuple[str, float]]

    def _terms(self, name: str) -> list[tuple[str, float, int]]:
        """``G^i = sum_j d_j A^{ij}`` as ``(potential, sign, axis)`` triples."""
        i = self.velocity_names.index(name)
        out: list[tuple[str, float, int]] = []
        for j in range(len(self.flux_axes)):
            entry = self.potential.get((i, j))
            if entry is None:  # j == i: A^{ii} = 0
                continue
            potential, sign = entry
            out.append((potential, sign, self.axis_index[j]))
        return out

    def _velocity_value(self, inner: FieldState, name: str) -> Array:
        total: Array | None = None
        for potential, sign, axis in self._terms(name):
            d = inner.ops.derivative(inner, potential, axis=axis, order=1)
            term = sign * d
            total = term if total is None else total + term
        assert total is not None
        return total

    def _velocity_derivative(
        self, inner: FieldState, name: str, *, axis: int, order: int
    ) -> Array:
        total: Array | None = None
        for potential, sign, p_axis in self._terms(name):
            folded = (
                {p_axis: order + 1} if p_axis == axis else {p_axis: 1, axis: order}
            )
            term = sign * _potential_partial(inner, potential, folded)
            total = term if total is None else total + term
        assert total is not None
        return total

    def _velocity_mixed(
        self,
        inner: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        total: Array | None = None
        for potential, sign, p_axis in self._terms(name):
            folded: dict[int, int] = {p_axis: 1}
            for a, o in zip(axes, orders, strict=True):
                folded[a] = folded.get(a, 0) + int(o)
            term = sign * _potential_partial(inner, potential, folded)
            total = term if total is None else total + term
        assert total is not None
        return total


def _potential_partial(
    inner: FieldState, potential: str, folded: dict[int, int]
) -> Array:
    """``D^alpha`` of one potential, given ``alpha`` as ``{axis: order}``."""
    wanted = {a: o for a, o in folded.items() if o > 0}
    if len(wanted) == 1:
        ((axis, order),) = wanted.items()
        return inner.ops.derivative(inner, potential, axis=axis, order=order)
    axes = tuple(wanted)
    return inner.ops.mixed_partial(
        inner, potential, axes, tuple(wanted[a] for a in axes)
    )


def make_flux_form_field(
    *,
    base: FieldBase,
    potential_names: tuple[str, ...],
    flux_names: tuple[str, ...],
    axes: tuple[str, ...] | None = None,
    passthrough_names: tuple[str, ...] = (),
) -> FluxFormField:
    """Build a :class:`FluxFormField`.

    See :class:`omnibias.pinn.torch.cage.FluxFormField` for the argument
    semantics; this factory validates identically.
    """
    axis_names = tuple(base.coordinate_spec.axes) if axes is None else tuple(axes)
    if len(flux_names) != len(axis_names):
        raise ValueError(
            f"need one flux name per axis: {len(axis_names)} axes {axis_names!r} "
            f"but {len(flux_names)} names {flux_names!r}"
        )
    potential = potential_table(len(axis_names), tuple(potential_names))
    for name in potential_names:
        if not base.components.is_component(name):
            raise ValueError(
                f"potential component {name!r} not in base components "
                f"{base.components.names!r}"
            )
    for name in passthrough_names:
        if not base.components.is_component(name):
            raise ValueError(f"passthrough {name!r} not in base components")
    if len(set(flux_names)) != len(flux_names):
        raise ValueError(f"flux names must be unique, got {flux_names!r}")

    return FluxFormField(
        base=base,
        velocity_names=tuple(flux_names),
        passthrough_names=tuple(passthrough_names),
        coordinate_spec=base.coordinate_spec,
        components=ComponentSpec(
            tuple(flux_names) + tuple(passthrough_names),
            groups={"flux": tuple(flux_names)},
        ),
        potential_names=tuple(potential_names),
        flux_axes=axis_names,
        axis_index=tuple(base.coordinate_spec.axis_index(a) for a in axis_names),
        potential=potential,
    )


def _ff_flatten(f: FluxFormField) -> tuple[tuple[object, ...], tuple[object, ...]]:
    return (f.base,), (
        f.velocity_names,
        f.passthrough_names,
        f.coordinate_spec,
        f.components,
        f.potential_names,
        f.flux_axes,
        f.axis_index,
        tuple(sorted(f.potential.items())),
    )


def _ff_unflatten(
    aux: tuple[object, ...], leaves: tuple[object, ...]
) -> FluxFormField:
    (base,) = leaves
    (
        velocity_names,
        passthrough_names,
        coordinate_spec,
        components,
        potential_names,
        flux_axes,
        axis_index,
        potential_items,
    ) = aux
    obj = FluxFormField.__new__(FluxFormField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "potential_names", potential_names)
    object.__setattr__(obj, "flux_axes", flux_axes)
    object.__setattr__(obj, "axis_index", axis_index)
    object.__setattr__(obj, "potential", dict(potential_items))
    return obj


jax.tree_util.register_pytree_node(FluxFormField, _ff_flatten, _ff_unflatten)


__all__ = [
    "FluxFormField",
    "antisymmetric_pairs",
    "make_flux_form_field",
]
