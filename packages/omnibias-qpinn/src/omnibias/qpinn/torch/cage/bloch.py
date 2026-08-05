# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Bloch-periodic cage (torch backend).

Bloch's theorem: an eigenstate of a periodic Hamiltonian with crystal
momentum :math:`k` can be written

.. math::

    \psi_k(x) = e^{i\,k\cdot x}\,u_k(x),

where :math:`u_k(x)` has the same periodicity as the crystal lattice.
This cage trains the periodic part :math:`u_k` only (typically using a
spectral field with the correct periodicity); the Bloch phase
:math:`e^{i k\cdot x}` is multiplied in by construction so the output
wavefunction is exactly Bloch-periodic.

For v0.0.1 the cage supports up to second-order pure single-axis
derivatives (everything Schrodinger / Helmholtz needs). Higher orders
and mixed partials raise :class:`NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.cage.incompressible import _CageFieldBase
from omnibias.pinn.torch.fields.base import FieldBase, _import_torch_ops
from torch import Tensor


class BlochPeriodicField(_CageFieldBase):
    r"""Bloch-periodic cage exposing :math:`\psi_k = e^{ik\cdot x} u_k`.

    Parameters
    ----------
    base
        Underlying :class:`FieldBase` whose :class:`ComponentSpec`
        carries a wavefunction group (default ``"u"``) representing
        :math:`u_k(x)`. The base field should be periodic in its
        spatial axes (e.g. a :class:`SpectralVectorField`); the cage
        does not enforce this -- it just multiplies in the Bloch
        phase.
    k
        Crystal momentum vector. Length must equal the number of
        spatial axes on the coordinate spec. Components are floats
        (in inverse-length units consistent with the coordinate spec).
    base_group
        Wavefunction group name on the base. Default ``"u"``.
    output_group
        Wavefunction group name on the cage output. Default ``"psi"``.
        Component names on the cage are ``f"{output_group}_re"`` and
        ``f"{output_group}_im"``.
    """

    k: Tensor
    base_group: str
    output_group: str
    spatial_idx: tuple[int, ...]

    def __init__(
        self,
        *,
        base: FieldBase,
        k: Sequence[float] | Tensor,
        base_group: str = "u",
        output_group: str = "psi",
    ) -> None:
        if not base.components.is_group(base_group):
            raise ValueError(
                f"base does not have a wavefunction group {base_group!r}; "
                "build it with omnibias.qpinn.make_psi_components"
            )
        members = base.components.group_members(base_group)
        if len(members) != 2:
            raise ValueError(
                f"wavefunction group {base_group!r} must have exactly 2 "
                f"components (re, im); got {members!r}"
            )
        n_spatial = base.coordinate_spec.n_spatial
        k_tensor = torch.as_tensor(k, dtype=torch.float64)
        if k_tensor.shape != (n_spatial,):
            raise ValueError(
                f"k must have shape ({n_spatial},) matching coordinate_spec "
                f"spatial axes; got shape {tuple(k_tensor.shape)}"
            )
        spatial_idx = tuple(
            base.coordinate_spec.axis_index(a)
            for a in base.coordinate_spec.spatial_axes
        )
        from omnibias.pinn._core.components import ComponentSpec
        cage_components = ComponentSpec(
            (f"{output_group}_re", f"{output_group}_im"),
            groups={output_group: (f"{output_group}_re", f"{output_group}_im")},
        )
        passthrough_names = tuple(
            n for n in base.components.names if n not in members
        )
        super().__init__(
            base=base,
            velocity_names=(f"{output_group}_re", f"{output_group}_im"),
            passthrough_names=passthrough_names,
            groups={output_group: (f"{output_group}_re", f"{output_group}_im")},
        )
        # Override the spec built by _CageFieldBase (which gets it wrong
        # because the cage has different component names than the base).
        object.__setattr__(self, "components", cage_components)
        self.base_group = base_group
        self.output_group = output_group
        self.spatial_idx = spatial_idx
        self.register_buffer("k", k_tensor)

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
        # phase = sum_i k_i x_{spatial_i}
        phase = torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        for i, ax in enumerate(self.spatial_idx):
            phase = phase + self.k[i] * coords[..., ax]
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        inner_state = self.base.evaluate(coords)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_torch_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={
                "_cage_inner_state": inner_state,
                "_cos_phase": cos_phase,
                "_sin_phase": sin_phase,
            },
        )

    def _u_value(self, inner: FieldState) -> tuple[Tensor, Tensor]:
        members = self.base.components.group_members(self.base_group)
        u_re = inner.ops.value(inner, members[0])
        u_im = inner.ops.value(inner, members[1])
        return u_re, u_im

    def _u_derivative(
        self, inner: FieldState, *, axis: int, order: int,
    ) -> tuple[Tensor, Tensor]:
        members = self.base.components.group_members(self.base_group)
        d_re = inner.ops.derivative(inner, members[0], axis=axis, order=order)
        d_im = inner.ops.derivative(inner, members[1], axis=axis, order=order)
        return d_re, d_im

    def value_component(self, state: FieldState, name: str) -> Tensor:
        inner = state.extra["_cage_inner_state"]
        cos_p = state.extra["_cos_phase"]
        sin_p = state.extra["_sin_phase"]
        if name in self.passthrough_names:
            return inner.ops.value(inner, name)
        u_re, u_im = self._u_value(inner)
        if name == self.velocity_names[0]:  # psi_re
            return cos_p * u_re - sin_p * u_im
        if name == self.velocity_names[1]:  # psi_im
            return sin_p * u_re + cos_p * u_im
        raise KeyError(name)

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Tensor:
        if order == 0:
            return self.value_component(state, name)
        if order > 2:
            raise NotImplementedError(
                f"BlochPeriodicField supports derivative orders up to 2 in "
                f"v0.0.1 (got order={order}). Higher orders need the full "
                "Leibniz / multinomial expansion -- planned for v0.0.2."
            )
        inner = state.extra["_cage_inner_state"]
        cos_p = state.extra["_cos_phase"]
        sin_p = state.extra["_sin_phase"]
        if name in self.passthrough_names:
            return inner.ops.derivative(inner, name, axis=axis, order=order)
        if name not in self.velocity_names:
            raise KeyError(name)
        # Locate which spatial slot this axis is in (so we can get k_a).
        try:
            slot = self.spatial_idx.index(axis)
            k_a = float(self.k[slot])
        except ValueError:
            k_a = 0.0  # axis is the time axis -> no phase contribution

        u_re, u_im = self._u_value(inner)
        du_re, du_im = self._u_derivative(inner, axis=axis, order=1)
        if order == 1:
            if name == self.velocity_names[0]:
                # ∂_a psi_re = cos ∂u_re - sin ∂u_im - k_a (sin u_re + cos u_im)
                return cos_p * du_re - sin_p * du_im - k_a * (sin_p * u_re + cos_p * u_im)
            # name == psi_im
            return sin_p * du_re + cos_p * du_im + k_a * (cos_p * u_re - sin_p * u_im)

        # order == 2
        d2u_re, d2u_im = self._u_derivative(inner, axis=axis, order=2)
        if name == self.velocity_names[0]:
            # ∂_a^2 psi_re = cos ∂^2 u_re - sin ∂^2 u_im
            #               - 2 k_a (sin ∂u_re + cos ∂u_im)
            #               - k_a^2 (cos u_re - sin u_im)
            return (
                cos_p * d2u_re - sin_p * d2u_im
                - 2.0 * k_a * (sin_p * du_re + cos_p * du_im)
                - k_a * k_a * (cos_p * u_re - sin_p * u_im)
            )
        # psi_im
        return (
            sin_p * d2u_re + cos_p * d2u_im
            + 2.0 * k_a * (cos_p * du_re - sin_p * du_im)
            - k_a * k_a * (sin_p * u_re + cos_p * u_im)
        )

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        if name in self.passthrough_names:
            inner = state.extra["_cage_inner_state"]
            return inner.ops.mixed_partial(inner, name, axes, orders)
        # For wavefunction components we only support single-axis form
        # in v0.0.1. Fold orders and re-dispatch through derivative if
        # everything is on one axis.
        folded: dict[int, int] = {}
        for a, o in zip(axes, orders, strict=False):
            if o < 1:
                continue
            folded[a] = folded.get(a, 0) + int(o)
        if len(folded) == 1:
            ((ax, total_order),) = folded.items()
            return self.derivative(state, name, axis=ax, order=total_order)
        raise NotImplementedError(
            "BlochPeriodicField only supports single-axis derivatives in v0.0.1; "
            "true mixed partials need the multi-axis Leibniz expansion (planned)."
        )


def make_bloch_periodic_field(
    *,
    base: FieldBase,
    k: Sequence[float] | Tensor,
    base_group: str = "u",
    output_group: str = "psi",
) -> BlochPeriodicField:
    r"""Build a :class:`BlochPeriodicField` from a base field + crystal momentum.

    Parameters
    ----------
    base
        Base field providing the periodic part :math:`u_k(x)`. Must
        carry a wavefunction group with two real components.
    k
        Crystal momentum vector, length equal to the number of spatial
        axes on the base.
    base_group
        Group name on the base. Default ``"u"``.
    output_group
        Group name on the cage output. Default ``"psi"``.
    """
    return BlochPeriodicField(
        base=base, k=k, base_group=base_group, output_group=output_group,
    )


__all__ = ["BlochPeriodicField", "make_bloch_periodic_field"]
