# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Incompressible-flow cage fields for the torch backend.

Two backends:

- :class:`StreamfunctionField` (2D): the user trains a scalar
  streamfunction :math:`\\psi`; the cage exposes velocity components
  ``(u, v)`` with :math:`u = \\partial_y \\psi, v = -\\partial_x
  \\psi`. By construction :math:`\\nabla \\cdot u = \\partial_x \\partial_y
  \\psi - \\partial_y \\partial_x \\psi = 0` to machine precision.

- :class:`VectorPotentialField` (3D): the user trains a 3-vector
  potential :math:`A`; the cage exposes velocity components
  ``(u, v, w)`` with :math:`u = \\nabla \\times A`. By construction
  :math:`\\nabla \\cdot (\\nabla \\times A) = 0` exactly.

- :class:`HelmholtzProjectionField` (1D / 2D / 3D): the user trains a
  predicted velocity ``u_pred`` plus a scalar potential :math:`\\phi`;
  the cage exposes velocity components with ``u = u_pred - grad(phi)``.
  Combined with a Coulomb-gauge constraint :math:`\\Delta \\phi =
  \\nabla \\cdot u_{pred}` (added to the loss), this gives a soft
  divergence-free projection.

Cage fields wrap an underlying base field. They are themselves
:class:`FieldBase` subclasses, so :meth:`__call__` returns a
:class:`FieldState` and the attribute DSL works the same way the user
expects. The ops dispatch recognises these fields via the
``_is_cage(state)`` predicate and routes value / derivative /
``mixed_partial`` calls into the cage's projection logic.

Pass-through components (e.g. pressure ``p`` in NS) are forwarded to
the base field unchanged. The cage's :class:`ComponentSpec` is the
union of velocity (caged) and pass-through (raw) components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.sigma_cache import SigmaCache
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.fields.base import FieldBase, _import_torch_ops
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    pass


def _resolve_axis(coordinate_spec: CoordinateSpec, axis: int | str) -> int:
    return coordinate_spec.axis_index(axis)


# ----------------- shared base for cage fields ----------------------


class _CageFieldBase(FieldBase):
    """Common machinery for cage fields wrapping a single base field.

    Subclasses must implement:

    - :attr:`velocity_names`: tuple of caged velocity component names.
    - :meth:`_velocity_value`: ``f(state, vname) -> Tensor`` for the
      caged value.
    - :meth:`_velocity_derivative`: ``f(state, vname, axis, order) ->
      Tensor`` for caged derivative.
    - :meth:`_velocity_mixed`: ``f(state, vname, axes, orders) -> Tensor``
      for caged mixed partial.
    """

    base: FieldBase
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]

    def __init__(
        self,
        *,
        base: FieldBase,
        velocity_names: tuple[str, ...],
        passthrough_names: tuple[str, ...] = (),
        groups: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        if not isinstance(base, FieldBase):
            raise TypeError(
                f"base must be a FieldBase, got {type(base).__name__}"
            )
        coordinate_spec = base.coordinate_spec
        # Default group: "velocity" -> velocity_names if not supplied.
        if groups is None:
            groups = {"velocity": tuple(velocity_names)}
        components = ComponentSpec(
            tuple(velocity_names) + tuple(passthrough_names),
            groups=groups,
        )
        super().__init__(
            coordinate_spec=coordinate_spec,
            components=components,
        )
        # Wire the base as a sub-module so its parameters travel through
        # ``state_dict`` / ``model.parameters()``.
        self.base = base
        self.velocity_names = tuple(velocity_names)
        self.passthrough_names = tuple(passthrough_names)

    # ----- delegate the sigma cache to the base -----------------------

    def _make_sigma_cache(self, coords: Tensor) -> SigmaCache[Tensor]:
        return self.base._make_sigma_cache(coords)

    def _pre_activations(self, coords: Tensor) -> Tensor | None:
        return self.base._pre_activations(coords)

    def evaluate(self, coords: Tensor) -> FieldState[Tensor]:
        if coords.dim() != 2:
            raise ValueError(
                f"coords must be 2D (B, D), got shape {tuple(coords.shape)}"
            )
        if coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"coords last dim {coords.shape[-1]} != "
                f"coordinate_spec.ndim {self.coordinate_spec.ndim}"
            )
        # Build the inner state via the base and reuse its sigma cache so
        # caged derivatives don't recompute sigma^(n) values.
        inner_state = self.base.evaluate(coords)
        cage_state = FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_torch_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={"_cage_inner_state": inner_state},
        )
        return cage_state

    # ----- ops dispatch hooks (called by basic.py via _is_cage) -------

    def value_component(self, state: FieldState, name: str) -> Tensor:
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return self.base.ops_value(inner, name)
        if name in self.velocity_names:
            return self._velocity_value(inner, name)
        raise KeyError(
            f"{name!r} is neither a velocity {self.velocity_names!r} nor "
            f"a passthrough {self.passthrough_names!r} component"
        )

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Tensor:
        if order == 0:
            return self.value_component(state, name)
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return self.base.ops_derivative(inner, name, axis=axis, order=order)
        if name in self.velocity_names:
            return self._velocity_derivative(inner, name, axis=axis, order=order)
        raise KeyError(name)

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return self.base.ops_mixed(inner, name, axes, orders)
        if name in self.velocity_names:
            return self._velocity_mixed(inner, name, axes, orders)
        raise KeyError(name)

    # Subclasses implement these:

    def _velocity_value(self, inner_state: FieldState, name: str) -> Tensor:
        raise NotImplementedError

    def _velocity_derivative(
        self, inner_state: FieldState, name: str, *, axis: int, order: int,
    ) -> Tensor:
        raise NotImplementedError

    def _velocity_mixed(
        self,
        inner_state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        raise NotImplementedError


# Helper accessors that the base field exposes regardless of architecture.
def _ops_for(field: FieldBase) -> any:
    return _import_torch_ops()


# Monkey-attach thin accessors so cage code can write
# ``self.base.ops_value(state, name)`` instead of importing the dispatch
# repeatedly. Prefer one source of truth -- read off the state's ops.
def _value_via(state: FieldState, name: str) -> Tensor:
    return state.ops.value(state, name)


def _derivative_via(
    state: FieldState, name: str, *, axis: int, order: int,
) -> Tensor:
    return state.ops.derivative(state, name, axis=axis, order=order)


def _mixed_via(
    state: FieldState,
    name: str,
    axes: tuple[int, ...],
    orders: tuple[int, ...],
) -> Tensor:
    return state.ops.mixed_partial(state, name, axes, orders)


# Patch the base classes once, so any FieldBase subclass has these.
FieldBase.ops_value = staticmethod(_value_via)             # type: ignore[attr-defined]
FieldBase.ops_derivative = staticmethod(_derivative_via)    # type: ignore[attr-defined]
FieldBase.ops_mixed = staticmethod(_mixed_via)             # type: ignore[attr-defined]


# ----------------- 2D streamfunction --------------------------------


class StreamfunctionField(_CageFieldBase):
    """2D incompressible cage: ``u = d_y psi, v = -d_x psi``.

    Parameters
    ----------
    base
        Underlying field. Its :class:`ComponentSpec` must contain
        ``psi`` (the streamfunction component name, configurable) and
        any pass-through components.
    psi
        Name of the streamfunction component on the base. Default
        ``"psi"``.
    velocity_names
        Names for the caged velocity components. Default ``("u", "v")``.
    passthrough_names
        Names of components that are passed through unchanged
        (e.g. ``("p",)`` for pressure).
    spatial_axes
        Names of the two spatial axes. Default ``("x", "y")``. The
        first axis is differentiated negative for ``v``; the second is
        the ``y`` (used for ``u``).
    """

    def __init__(
        self,
        *,
        base: FieldBase,
        psi: str = "psi",
        velocity_names: tuple[str, str] = ("u", "v"),
        passthrough_names: tuple[str, ...] = (),
        spatial_axes: tuple[str, str] = ("x", "y"),
    ) -> None:
        if base.coordinate_spec.n_spatial != 2:
            raise ValueError(
                f"StreamfunctionField requires a 2D spatial domain; got "
                f"{base.coordinate_spec.n_spatial}"
            )
        if not base.components.is_component(psi):
            raise ValueError(
                f"streamfunction component {psi!r} not in base "
                f"components {base.components.names!r}"
            )
        for n in passthrough_names:
            if not base.components.is_component(n):
                raise ValueError(
                    f"passthrough {n!r} not in base components"
                )
        if len(velocity_names) != 2:
            raise ValueError(
                f"StreamfunctionField requires exactly 2 velocity names, "
                f"got {velocity_names!r}"
            )
        super().__init__(
            base=base,
            velocity_names=velocity_names,
            passthrough_names=passthrough_names,
        )
        self.psi = psi
        self.spatial_axes = tuple(spatial_axes)
        self._x_idx = base.coordinate_spec.axis_index(spatial_axes[0])
        self._y_idx = base.coordinate_spec.axis_index(spatial_axes[1])

    def _vsign_and_axis(self, name: str) -> tuple[float, int]:
        """For velocity component ``name`` return ``(sign, partial_axis)``.

        ``u = +d_y psi`` -> sign=+1, axis=y. ``v = -d_x psi`` -> sign=-1, axis=x.
        """
        if name == self.velocity_names[0]:
            return 1.0, self._y_idx
        if name == self.velocity_names[1]:
            return -1.0, self._x_idx
        raise KeyError(f"{name!r} not in {self.velocity_names!r}")

    def _velocity_value(self, inner_state: FieldState, name: str) -> Tensor:
        sign, partial_axis = self._vsign_and_axis(name)
        d = inner_state.ops.derivative(
            inner_state, self.psi, axis=partial_axis, order=1,
        )
        return sign * d

    def _velocity_derivative(
        self, inner_state: FieldState, name: str, *, axis: int, order: int,
    ) -> Tensor:
        sign, partial_axis = self._vsign_and_axis(name)
        if axis == partial_axis:
            d = inner_state.ops.derivative(
                inner_state, self.psi, axis=axis, order=order + 1,
            )
        else:
            d = inner_state.ops.mixed_partial(
                inner_state, self.psi,
                (partial_axis, axis), (1, order),
            )
        return sign * d

    def _velocity_mixed(
        self,
        inner_state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        sign, partial_axis = self._vsign_and_axis(name)
        # Append the streamfunction's own partial axis into the mixed-partial
        # request, fold any duplicates.
        all_axes = list(axes) + [partial_axis]
        all_orders = list(orders) + [1]
        folded: dict[int, int] = {}
        for a, o in zip(all_axes, all_orders, strict=False):
            folded[a] = folded.get(a, 0) + int(o)
        if not folded:
            return sign * inner_state.ops.value(inner_state, self.psi)
        ax = tuple(folded.keys())
        og = tuple(folded[a] for a in ax)
        d = inner_state.ops.mixed_partial(inner_state, self.psi, ax, og)
        return sign * d


# ----------------- 3D vector potential ------------------------------


class VectorPotentialField(_CageFieldBase):
    """3D incompressible cage: ``u = curl(A)``.

    The base field must expose three vector-potential components
    (default ``"A1", "A2", "A3"``); the cage exposes ``(u, v, w)`` such
    that :math:`\\nabla \\cdot u = 0` to machine precision.

    Parameters
    ----------
    base
        Underlying field with vector-potential components and any
        pass-through components.
    A_components
        Names of the 3-vector potential components. Default
        ``("A1", "A2", "A3")``.
    velocity_names
        Names for caged velocity. Default ``("u", "v", "w")``.
    passthrough_names
        Components of the base that pass through (e.g. ``("p",)``).
    spatial_axes
        Names of the 3 spatial axes. Default ``("x", "y", "z")``.
    """

    def __init__(
        self,
        *,
        base: FieldBase,
        A_components: tuple[str, str, str] = ("A1", "A2", "A3"),
        velocity_names: tuple[str, str, str] = ("u", "v", "w"),
        passthrough_names: tuple[str, ...] = (),
        spatial_axes: tuple[str, str, str] = ("x", "y", "z"),
    ) -> None:
        if base.coordinate_spec.n_spatial != 3:
            raise ValueError(
                f"VectorPotentialField requires a 3D spatial domain; got "
                f"{base.coordinate_spec.n_spatial}"
            )
        for n in A_components:
            if not base.components.is_component(n):
                raise ValueError(
                    f"vector-potential component {n!r} not in base"
                )
        for n in passthrough_names:
            if not base.components.is_component(n):
                raise ValueError(
                    f"passthrough {n!r} not in base components"
                )
        if len(velocity_names) != 3 or len(A_components) != 3:
            raise ValueError(
                "VectorPotentialField requires exactly 3 velocity and "
                "3 vector-potential names"
            )
        super().__init__(
            base=base,
            velocity_names=velocity_names,
            passthrough_names=passthrough_names,
        )
        self.A_components = tuple(A_components)
        self.spatial_axes = tuple(spatial_axes)
        self._x_idx = base.coordinate_spec.axis_index(spatial_axes[0])
        self._y_idx = base.coordinate_spec.axis_index(spatial_axes[1])
        self._z_idx = base.coordinate_spec.axis_index(spatial_axes[2])

    def _curl_terms(self, name: str) -> tuple[
        tuple[str, int, float], tuple[str, int, float],
    ]:
        """Return the two ``(component, partial_axis, sign)`` curl
        contributions for the velocity component ``name``.

        - ``u = d_y A3 - d_z A2``  -> [(A3, y, +1), (A2, z, -1)]
        - ``v = d_z A1 - d_x A3``  -> [(A1, z, +1), (A3, x, -1)]
        - ``w = d_x A2 - d_y A1``  -> [(A2, x, +1), (A1, y, -1)]
        """
        A1, A2, A3 = self.A_components
        x_, y_, z_ = self._x_idx, self._y_idx, self._z_idx
        u, v, w = self.velocity_names
        if name == u:
            return (A3, y_, 1.0), (A2, z_, -1.0)
        if name == v:
            return (A1, z_, 1.0), (A3, x_, -1.0)
        if name == w:
            return (A2, x_, 1.0), (A1, y_, -1.0)
        raise KeyError(f"{name!r} not in velocity {self.velocity_names!r}")

    def _velocity_value(self, inner_state: FieldState, name: str) -> Tensor:
        (a, ax_a, sa), (b, ax_b, sb) = self._curl_terms(name)
        d_a = inner_state.ops.derivative(inner_state, a, axis=ax_a, order=1)
        d_b = inner_state.ops.derivative(inner_state, b, axis=ax_b, order=1)
        return sa * d_a + sb * d_b

    def _velocity_derivative(
        self, inner_state: FieldState, name: str, *, axis: int, order: int,
    ) -> Tensor:
        (a, ax_a, sa), (b, ax_b, sb) = self._curl_terms(name)
        if axis == ax_a:
            d_a = inner_state.ops.derivative(
                inner_state, a, axis=axis, order=order + 1,
            )
        else:
            d_a = inner_state.ops.mixed_partial(
                inner_state, a, (ax_a, axis), (1, order),
            )
        if axis == ax_b:
            d_b = inner_state.ops.derivative(
                inner_state, b, axis=axis, order=order + 1,
            )
        else:
            d_b = inner_state.ops.mixed_partial(
                inner_state, b, (ax_b, axis), (1, order),
            )
        return sa * d_a + sb * d_b

    def _velocity_mixed(
        self,
        inner_state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        (a, ax_a, sa), (b, ax_b, sb) = self._curl_terms(name)

        def _composed(comp: str, partial_axis: int) -> Tensor:
            full_axes = list(axes) + [partial_axis]
            full_orders = list(orders) + [1]
            folded: dict[int, int] = {}
            for ax, o in zip(full_axes, full_orders, strict=False):
                folded[ax] = folded.get(ax, 0) + int(o)
            ax_t = tuple(folded.keys())
            og_t = tuple(folded[ax] for ax in ax_t)
            return inner_state.ops.mixed_partial(
                inner_state, comp, ax_t, og_t,
            )

        return sa * _composed(a, ax_a) + sb * _composed(b, ax_b)


# ----------------- Helmholtz projection -----------------------------


class HelmholtzProjectionField(_CageFieldBase):
    """Soft Helmholtz projection: ``u = u_pred - grad(phi)``.

    Combined with the loss
    :math:`L_{gauge} = \\|\\Delta \\phi - \\nabla \\cdot u_{pred}\\|^2`
    this drives ``u`` toward divergence-free.

    Parameters
    ----------
    base
        Underlying field with components for ``u_pred_*`` and ``phi``,
        plus any pass-throughs.
    u_pred_components
        Names of the predicted velocity components. Length matches the
        number of spatial axes.
    phi
        Name of the scalar potential component.
    velocity_names
        Names for the caged velocity. Length must equal
        ``len(u_pred_components)``.
    passthrough_names
        Pass-through components.
    """

    def __init__(
        self,
        *,
        base: FieldBase,
        u_pred_components: tuple[str, ...],
        phi: str = "phi",
        velocity_names: tuple[str, ...] | None = None,
        passthrough_names: tuple[str, ...] = (),
    ) -> None:
        n_spatial = base.coordinate_spec.n_spatial
        if len(u_pred_components) != n_spatial:
            raise ValueError(
                f"HelmholtzProjectionField: u_pred has {len(u_pred_components)}"
                f" components but coordinate spec has {n_spatial} spatial axes"
            )
        for n in u_pred_components:
            if not base.components.is_component(n):
                raise ValueError(f"u_pred component {n!r} not in base")
        if not base.components.is_component(phi):
            raise ValueError(f"phi component {phi!r} not in base")
        for n in passthrough_names:
            if not base.components.is_component(n):
                raise ValueError(f"passthrough {n!r} not in base")
        if velocity_names is None:
            velocity_names = tuple(f"u{i+1}" for i in range(n_spatial))
        if len(velocity_names) != n_spatial:
            raise ValueError(
                f"HelmholtzProjectionField: velocity_names length "
                f"{len(velocity_names)} != n_spatial {n_spatial}"
            )
        super().__init__(
            base=base,
            velocity_names=velocity_names,
            passthrough_names=passthrough_names,
        )
        self.u_pred_components = tuple(u_pred_components)
        self.phi = phi
        self._spatial_axis_indices = tuple(
            base.coordinate_spec.axis_index(a)
            for a in base.coordinate_spec.spatial_axes
        )

    def _vel_index(self, name: str) -> int:
        return self.velocity_names.index(name)

    def _velocity_value(self, inner_state: FieldState, name: str) -> Tensor:
        i = self._vel_index(name)
        u_i = inner_state.ops.value(inner_state, self.u_pred_components[i])
        ax_i = self._spatial_axis_indices[i]
        d_phi = inner_state.ops.derivative(
            inner_state, self.phi, axis=ax_i, order=1,
        )
        return u_i - d_phi

    def _velocity_derivative(
        self, inner_state: FieldState, name: str, *, axis: int, order: int,
    ) -> Tensor:
        i = self._vel_index(name)
        d_u = inner_state.ops.derivative(
            inner_state, self.u_pred_components[i], axis=axis, order=order,
        )
        ax_i = self._spatial_axis_indices[i]
        if axis == ax_i:
            d_phi = inner_state.ops.derivative(
                inner_state, self.phi, axis=axis, order=order + 1,
            )
        else:
            d_phi = inner_state.ops.mixed_partial(
                inner_state, self.phi, (ax_i, axis), (1, order),
            )
        return d_u - d_phi

    def _velocity_mixed(
        self,
        inner_state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        i = self._vel_index(name)
        d_u = inner_state.ops.mixed_partial(
            inner_state, self.u_pred_components[i], axes, orders,
        )
        ax_i = self._spatial_axis_indices[i]
        full_axes = list(axes) + [ax_i]
        full_orders = list(orders) + [1]
        folded: dict[int, int] = {}
        for ax, o in zip(full_axes, full_orders, strict=False):
            folded[ax] = folded.get(ax, 0) + int(o)
        ax_t = tuple(folded.keys())
        og_t = tuple(folded[ax] for ax in ax_t)
        d_phi = inner_state.ops.mixed_partial(
            inner_state, self.phi, ax_t, og_t,
        )
        return d_u - d_phi


# ----------------- gauge constraints ---------------------------------


def coulomb_gauge_loss(
    field: VectorPotentialField,
    coords: Tensor,
    *,
    inner_state: FieldState | None = None,
) -> Tensor:
    """Return ``mean (div(A))^2`` over the batch.

    Add this to the training loss when you want a Coulomb gauge
    :math:`\\nabla \\cdot A = 0`. The constraint is always satisfied at
    the global minimum because :math:`u = \\nabla \\times A` is invariant
    under the gauge transform :math:`A \\to A + \\nabla f`.
    """
    if inner_state is None:
        inner_state = field.base.evaluate(coords)
    div_A = inner_state.ops.divergence(inner_state, field.A_components)
    return (div_A ** 2).mean()


def helmholtz_gauge_loss(
    field: HelmholtzProjectionField,
    coords: Tensor,
    *,
    inner_state: FieldState | None = None,
) -> Tensor:
    """Return ``mean (Delta phi - div(u_pred))^2`` over the batch.

    The penalty makes ``u = u_pred - grad(phi)`` divergence-free at the
    global minimum.
    """
    if inner_state is None:
        inner_state = field.base.evaluate(coords)
    lap_phi = inner_state.ops.laplacian(inner_state, field.phi)
    div_u_pred = inner_state.ops.divergence(
        inner_state, field.u_pred_components,
    )
    return ((lap_phi - div_u_pred) ** 2).mean()


# ----------------- ops dispatch hook ---------------------------------


def is_cage_field(state: FieldState) -> bool:
    return isinstance(state.field, _CageFieldBase)


__all__ = [
    "HelmholtzProjectionField",
    "StreamfunctionField",
    "VectorPotentialField",
    "coulomb_gauge_loss",
    "helmholtz_gauge_loss",
    "is_cage_field",
]

# Marker read by the omnibias-fields backend ops to select the dispatch path
# (inherited by every concrete cage field).
_CageFieldBase._omnibias_dispatch = "cage"
