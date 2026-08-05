# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Flux-form (finite-volume) conservation cage for the torch backend.

A conservation law in divergence form,

.. math::

    \partial_t \rho + \nabla \cdot F = 0 ,

is one equation saying a single space-time vector ``G = (rho, F)`` is
divergence-free on the space-time axes. Written that way it has an *exact*
potential representation in any dimension: for an antisymmetric
``A^{ij} = -A^{ji}``,

.. math::

    G^i = \sum_j \partial_j A^{ij}
    \quad\Longrightarrow\quad
    \nabla \cdot G = \sum_{i,j} \partial_i \partial_j A^{ij} = 0

identically, because :math:`\partial_i \partial_j` is symmetric in ``(i, j)``
while ``A^{ij}`` is antisymmetric, so the double sum cancels term by term. No
quadrature, no penalty, no tolerance: the law holds wherever the potential is
twice differentiable, which for the closed-form tower is everywhere.

:class:`FluxFormField` is that cage. It takes the ``D (D - 1) / 2`` independent
potential components ``A^{ij}``, ``i < j``, and exposes the ``D`` components of
``G``.

Why this subsumes the two cages already in
:mod:`omnibias.pinn.torch.cage.incompressible`
------------------------------------------------------------------------------
* ``D = 2``, axes ``(x, y)``: the single potential ``A^{xy} = psi`` gives
  ``G = (d_y psi, -d_x psi)`` -- exactly :class:`StreamfunctionField`.
* ``D = 3``, axes ``(x, y, z)``: the three potentials are the vector potential
  under ``A^{ij} = eps_{ijk} A_k``, and ``G = curl(A)`` -- exactly
  :class:`VectorPotentialField`.

Those two remain the ergonomic front ends for incompressible flow. This one is
the general statement, and it is what unlocks the case neither of them covers:
letting ``t`` be one of the axes, so the divergence-free object is a space-time
flux and the identity is a *conservation law* rather than incompressibility.

The finite-volume reading
-------------------------
On a control volume ``V``, the divergence theorem gives
``d/dt int_V rho = -oint_{dV} F . n``: the cell balance every finite-volume
scheme is built to respect. A flux-form field satisfies it for *every* control
volume simultaneously, because the potential is the discrete scheme's
cumulative variable made continuous -- in 1+1D, ``A^{tx}`` integrated between
two faces *is* the mass between them, so the balance telescopes exactly as it
does on a staggered grid.

Cost. Reading ``G`` costs one derivative of the potential, and ``D^alpha G``
costs a mixed partial one order higher -- the same trade the streamfunction
cage makes, and the reason the closed-form tower matters here: an autodiff
field would pay an extra backward pass per order, while ``sigma^(n)`` does not
care what ``n`` is.
"""

from __future__ import annotations

from omnibias.pinn._core.fluxform import antisymmetric_pairs, potential_table
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.cage.incompressible import _CageFieldBase
from omnibias.pinn.torch.fields.base import FieldBase
from torch import Tensor


class FluxFormField(_CageFieldBase):
    r"""Hard divergence-form conservation: ``G^i = sum_j d_j A^{ij}``, ``A`` antisymmetric.

    Parameters
    ----------
    base:
        Field carrying the potential components.
    potential_names:
        One name per ``(i, j)`` pair with ``i < j``, in
        :func:`antisymmetric_pairs` order -- ``D (D - 1) / 2`` of them for
        ``D = len(axes)``.
    flux_names:
        The ``D`` exposed component names, in axis order. With a time axis
        first this reads ``("rho", "f_x", ...)``.
    axes:
        Axis names spanned by the flux, in order. Defaults to every axis of the
        coordinate spec, which is the conservation-law reading; pass only the
        spatial axes to get incompressibility instead.
    passthrough_names:
        Base components forwarded unchanged (e.g. a pressure).

    Notes
    -----
    ``divergence`` of the exposed vector is zero by construction, so adding a
    residual penalty for it is not just redundant but actively harmful -- it
    contributes only round-off-scale gradient noise.
    """

    def __init__(
        self,
        *,
        base: FieldBase,
        potential_names: tuple[str, ...],
        flux_names: tuple[str, ...],
        axes: tuple[str, ...] | None = None,
        passthrough_names: tuple[str, ...] = (),
    ) -> None:
        axis_names = (
            tuple(base.coordinate_spec.axes) if axes is None else tuple(axes)
        )
        if len(flux_names) != len(axis_names):
            raise ValueError(
                f"need one flux name per axis: {len(axis_names)} axes "
                f"{axis_names!r} but {len(flux_names)} names {flux_names!r}"
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

        super().__init__(
            base=base,
            velocity_names=tuple(flux_names),
            passthrough_names=tuple(passthrough_names),
            groups={"flux": tuple(flux_names)},
        )
        self.potential_names = tuple(potential_names)
        self.flux_axes = axis_names
        self._axis_index = tuple(
            base.coordinate_spec.axis_index(a) for a in axis_names
        )
        self._potential = potential

    def _terms(self, name: str) -> list[tuple[str, float, int]]:
        """``G^i = sum_j d_j A^{ij}`` as ``(potential, sign, axis)`` triples."""
        i = self.velocity_names.index(name)
        out: list[tuple[str, float, int]] = []
        for j in range(len(self.flux_axes)):
            entry = self._potential.get((i, j))
            if entry is None:  # j == i: A^{ii} = 0
                continue
            potential, sign = entry
            out.append((potential, sign, self._axis_index[j]))
        return out

    def _velocity_value(self, inner_state: FieldState, name: str) -> Tensor:
        total: Tensor | None = None
        for potential, sign, axis in self._terms(name):
            d = inner_state.ops.derivative(inner_state, potential, axis=axis, order=1)
            term = sign * d
            total = term if total is None else total + term
        assert total is not None  # at least one j != i exists for D >= 2
        return total

    def _velocity_derivative(
        self, inner_state: FieldState, name: str, *, axis: int, order: int
    ) -> Tensor:
        total: Tensor | None = None
        for potential, sign, p_axis in self._terms(name):
            term = sign * self._potential_partial(
                inner_state, potential, {p_axis: 1, axis: order}
                if p_axis != axis
                else {p_axis: order + 1},
            )
            total = term if total is None else total + term
        assert total is not None
        return total

    def _velocity_mixed(
        self,
        inner_state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        total: Tensor | None = None
        for potential, sign, p_axis in self._terms(name):
            folded: dict[int, int] = {p_axis: 1}
            for a, o in zip(axes, orders, strict=True):
                folded[a] = folded.get(a, 0) + int(o)
            term = sign * self._potential_partial(inner_state, potential, folded)
            total = term if total is None else total + term
        assert total is not None
        return total

    @staticmethod
    def _potential_partial(
        inner_state: FieldState, potential: str, folded: dict[int, int]
    ) -> Tensor:
        """``D^alpha`` of one potential, given ``alpha`` as ``{axis: order}``."""
        wanted = {a: o for a, o in folded.items() if o > 0}
        if len(wanted) == 1:
            (axis, order), = wanted.items()
            return inner_state.ops.derivative(
                inner_state, potential, axis=axis, order=order
            )
        axes = tuple(wanted)
        return inner_state.ops.mixed_partial(
            inner_state, potential, axes, tuple(wanted[a] for a in axes)
        )


def make_flux_form_field(
    *,
    base: FieldBase,
    potential_names: tuple[str, ...],
    flux_names: tuple[str, ...],
    axes: tuple[str, ...] | None = None,
    passthrough_names: tuple[str, ...] = (),
) -> FluxFormField:
    """Build a :class:`FluxFormField`; see it for the arguments."""
    return FluxFormField(
        base=base,
        potential_names=potential_names,
        flux_names=flux_names,
        axes=axes,
        passthrough_names=passthrough_names,
    )


__all__ = [
    "FluxFormField",
    "antisymmetric_pairs",
    "make_flux_form_field",
]
