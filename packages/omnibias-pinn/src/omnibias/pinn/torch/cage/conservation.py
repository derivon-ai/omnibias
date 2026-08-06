# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Conservation-law cage layers for the torch backend.

Three families:

1. Skew-symmetric advection: an alternative formulation of the
   advection term ``(u . nabla) u`` that conserves kinetic energy
   exactly under spatial integration even when ``div u`` is not zero
   to machine precision (i.e. the soft-projection regime). The
   v0.1 helpers are :func:`energy_conserving_advection` (vector form)
   and :func:`enstrophy_conserving_advection` (vorticity form).

2. Hard boundary conditions: :class:`HardBoundaryField` multiplies an
   underlying field by a user-supplied *distance function*
   :math:`d(x) \\to 0` on the boundary plus a scalar *boundary value*
   :math:`g(x, t)`, so :math:`u(x, t) = g(x, t) + d(x) f_{NN}(x, t)`.
   This enforces :math:`u|_{\\partial \\Omega} = g` exactly, on a boundary
   of any shape.

   It is **Dirichlet-only and does not compose**: the multiplicative
   ansatz breaks an inner derivative condition, because the distance
   factor lands on it. For Neumann, Robin or initial conditions --
   and for several conditions at once -- use the additive switching
   form, :class:`~omnibias.pinn.torch.cage.constrained.ConstrainedExpressionField`,
   which needs an axis-aligned box in exchange.

3. Mass-flux potential: :class:`MassFluxPotentialField` is an alias of
   the vector-potential cage (Section 1) for compressible flows where
   the conserved quantity is :math:`\\rho u = \\nabla \\times \\Psi`.
   It re-exports :class:`VectorPotentialField` with names tuned to the
   compressible-flow conventions.

For all three the *trainable parameters* live on the underlying base
field. The cage is structural sugar.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.cage.incompressible import (
    VectorPotentialField,
    _CageFieldBase,
)
from omnibias.pinn.torch.fields.base import FieldBase
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    pass


# --------- Skew-symmetric advection ---------------------------------


def energy_conserving_advection(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
) -> Tensor:
    """Skew-symmetric advection of the velocity field by itself.

    Returns the vector ``A_i = 1/2 * ((u . nabla) u_i + nabla_j (u_j u_i))``.
    For divergence-free :math:`u`, this equals ``(u . nabla) u``. In
    general it is the *energy-preserving* form: under ``int_{Omega}
    A_i u_i dx``, the contribution vanishes by integration by parts when
    integrated over a closed domain (no flux through boundaries).

    Shape: ``(B, len(velocity))``.
    """
    sa = state.coordinate_spec.spatial_axes
    if len(velocity) != len(sa):
        raise ValueError(
            f"energy_conserving_advection: velocity length {len(velocity)} "
            f"!= n_spatial {len(sa)}"
        )
    ops = state.ops
    # Compute (u . grad) u_i via the standard advection kernel.
    standard = ops.advection(state, velocity=velocity)             # (B, C)
    # Compute the "conservative" half: nabla_j (u_j u_i) =
    #   sum_j d/dx_j (u_j u_i) = sum_j (d u_j/dx_j * u_i + u_j * d u_i/dx_j)
    # = (div u) * u_i + (u . grad) u_i.
    # So skew form = 1/2 * (standard + conservative)
    #              = 1/2 * (standard + (div u) * u_i + standard)
    #              = standard + 1/2 * (div u) * u_i.
    # We compute (div u) * u_i once and add the half-correction.
    div_u = ops.divergence(state, velocity)                          # (B,)
    u_i = ops.stack_components(state, velocity)                      # (B, C)
    return standard + 0.5 * div_u.unsqueeze(-1) * u_i


def enstrophy_conserving_advection(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    vorticity: str,
) -> Tensor:
    """Skew-symmetric advection of a scalar vorticity by the velocity.

    For 2D NS in vorticity-streamfunction form the vorticity equation
    reads
    .. math::
        \\partial_t \\omega + (u \\cdot \\nabla) \\omega = \\nu \\Delta \\omega.

    The *enstrophy-preserving* form replaces the advection with the
    half-sum of the convective and conservative discretisations:
    ``A = 1/2 ((u . nabla) omega + nabla . (u * omega))``. For
    div-free ``u`` they are identical; in general the skew form
    preserves enstrophy under integration.

    Returns ``A`` of shape ``(B,)``.
    """
    ops = state.ops
    standard = ops.advection(state, velocity=velocity, scalar=vorticity)  # (B,)
    div_u = ops.divergence(state, velocity)
    omega = ops.value(state, vorticity)
    return standard + 0.5 * div_u * omega


# --------- Hard-boundary distance-function ansatz -------------------


class HardBoundaryField(_CageFieldBase):
    """Cage that strictly enforces ``u|_{boundary} = g(x, t)``.

    Output: ``u(x, t) = g(x, t) + d(x) * f_NN(x, t)``, where
    :math:`d(x) \\to 0` on :math:`\\partial \\Omega` is a user-supplied
    *signed-distance-like* function and :math:`g(x, t)` is the
    user-supplied *boundary value*. Both are passed in as plain Python
    callables on a coords tensor; they must be torch-differentiable
    (typically just analytic compositions of the inputs).

    Notes
    -----
    The cage exposes the same components as the base, with the value
    of every "boundary-cared-about" component replaced by
    ``g_c(coords) + d(coords) * base_c(coords)``. Pass-through
    components are forwarded unchanged.

    For *derivatives* of bounded components, the chain rule is applied:
    ``d/dx (g + d * f) = dg/dx + (dd/dx) * f + d * (df/dx)``. The cage
    relies on torch autograd for ``g`` and ``d`` themselves (they are
    typically simple algebraic expressions and autograd is cheap), but
    keeps ``f`` (the base field) on the closed-form path through the
    base's own ops dispatch.

    For higher-order derivatives this expands by the Leibniz rule;
    v0.1 supports up to order 4 via a generic implementation.
    """

    def __init__(
        self,
        *,
        base: FieldBase,
        distance_fn: Callable[[Tensor], Tensor],
        boundary_value_fn: Callable[[Tensor], dict[str, Tensor]] | None = None,
        bounded_names: Sequence[str] | None = None,
        passthrough_names: tuple[str, ...] = (),
        groups: dict[str, tuple[str, ...]] | None = None,
        max_derivative_order: int = 4,
    ) -> None:
        if not isinstance(base, FieldBase):
            raise TypeError("base must be a FieldBase")
        if bounded_names is None:
            bounded_names = tuple(
                n for n in base.components.names if n not in passthrough_names
            )
        else:
            bounded_names = tuple(bounded_names)
        for n in bounded_names:
            if not base.components.is_component(n):
                raise ValueError(f"bounded {n!r} not in base components")
        for n in passthrough_names:
            if not base.components.is_component(n):
                raise ValueError(f"passthrough {n!r} not in base components")
        super().__init__(
            base=base,
            velocity_names=bounded_names,
            passthrough_names=passthrough_names,
            groups=groups,
        )
        self.distance_fn = distance_fn
        self.boundary_value_fn = boundary_value_fn
        self.max_derivative_order = int(max_derivative_order)

    # The cage's "velocity" slot is overloaded to mean "bounded": the
    # _CageFieldBase machinery dispatches ``state.bounded_name``
    # accesses through ``_velocity_*``.

    def _g_value(self, name: str, coords: Tensor) -> Tensor:
        if self.boundary_value_fn is None:
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        g = self.boundary_value_fn(coords)
        return g.get(name, torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device))

    def _g_partial(
        self, name: str, coords: Tensor, axis: int, order: int = 1,
    ) -> Tensor:
        """``d^order g_name / dx_axis^order`` via autograd."""
        coords_g = coords.detach().requires_grad_(True)
        if self.boundary_value_fn is None:
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        cur = self.boundary_value_fn(coords_g).get(name)
        if cur is None:
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        for _ in range(order):
            cur, = torch.autograd.grad(
                cur.sum(), coords_g, create_graph=True, retain_graph=True,
            )
            cur = cur[..., axis]
        return cur

    def _d_value(self, coords: Tensor) -> Tensor:
        return self.distance_fn(coords)

    def _d_partial(
        self, coords: Tensor, axis: int, order: int = 1,
    ) -> Tensor:
        coords_g = coords.detach().requires_grad_(True)
        cur = self.distance_fn(coords_g)
        for _ in range(order):
            cur, = torch.autograd.grad(
                cur.sum(), coords_g, create_graph=True, retain_graph=True,
            )
            cur = cur[..., axis]
        return cur

    # ----- _CageFieldBase ops hooks ---------------------------------

    def _velocity_value(self, inner_state: FieldState, name: str) -> Tensor:
        coords = inner_state.coords
        g = self._g_value(name, coords)
        d = self._d_value(coords)
        f = inner_state.ops.value(inner_state, name)
        return g + d * f

    def _velocity_derivative(
        self, inner_state: FieldState, name: str, *, axis: int, order: int,
    ) -> Tensor:
        # Apply Leibniz: d^k(d * f)/dx^k = sum_{j=0}^{k} C(k,j) d^j d / dx^j
        # times d^{k-j} f / dx^{k-j}.
        # Plus d^k g / dx^k.
        from math import comb
        coords = inner_state.coords
        out = self._g_partial(name, coords, axis=axis, order=order)
        for j in range(order + 1):
            d_j = (
                self._d_value(coords) if j == 0 else
                self._d_partial(coords, axis=axis, order=j)
            )
            f_kj = (
                inner_state.ops.value(inner_state, name) if (order - j) == 0 else
                inner_state.ops.derivative(
                    inner_state, name, axis=axis, order=order - j,
                )
            )
            out = out + comb(order, j) * d_j * f_kj
        return out

    def _velocity_mixed(
        self,
        inner_state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        # Generic mixed partial of (g + d * f). We apply Leibniz one
        # axis at a time. For simplicity we lower into a sequence of
        # single-axis derivatives using the existing derivative path,
        # which is closed-form on the base. Each axis-step is the
        # standard Leibniz rule.
        # Since the result is multilinear in the partial-derivative
        # variables, we expand the product (g + d f) one axis-step at a
        # time:
        #     d_{x1} (g + d f) = d_{x1} g + (d_{x1} d) f + d (d_{x1} f).
        # Repeat per axis. The intermediate is a sum of terms, each of
        # the form ``coeff * partial_of_g_or_d * partial_of_f``.
        # Implementation: we represent the running sum as a list of
        # ``(coeff_term, f_partials)`` where ``coeff_term`` is a Tensor
        # already multiplied by the current ``d``-derivative-product, and
        # ``f_partials`` is a tuple of (axis, order) summarising what
        # partial of ``f`` is multiplied in. After processing all axes
        # we contract every term against the corresponding partial of
        # the base field.
        from math import comb
        # Fold repeated axes into a single (axis, total_order).
        folded: dict[int, int] = {}
        for a, o in zip(axes, orders, strict=False):
            folded[a] = folded.get(a, 0) + int(o)

        coords = inner_state.coords

        # Start with one term: (1) * (no partial). Term =
        # (g_partial_dict, d_partial_dict, base_axes_orders).
        # We split into "g part" + "(d * f) part" and process them
        # separately. The g part's mixed partial is direct via autograd.
        # The (d * f) part expands by Leibniz over the multi-axis
        # partial.

        # g part:
        out_g = torch.zeros(
            coords.shape[0], dtype=coords.dtype, device=coords.device,
        )
        if self.boundary_value_fn is not None:
            coords_g = coords.detach().requires_grad_(True)
            cur = self.boundary_value_fn(coords_g).get(name)
            if cur is not None:
                for ax, ord_ in folded.items():
                    for _ in range(ord_):
                        cur, = torch.autograd.grad(
                            cur.sum(), coords_g,
                            create_graph=True, retain_graph=True,
                        )
                        cur = cur[..., ax]
                out_g = cur

        # (d * f) part: multi-axis Leibniz.
        # We represent the running expansion as a list of pairs
        # ``(d_partial_orders, f_partial_orders)``: each is a dict
        # mapping axis -> partial order.
        # After processing all axes, the expansion is
        # sum over (d_o, f_o) of (multiplicity) *
        #     (mixed partial of d with d_o) * (mixed partial of f with f_o).
        # The multiplicity comes from the multinomial coefficient
        # prod_axes C(folded[ax], d_o[ax]).
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
                    new_mult = mult * comb(total, j)
                    new_terms.append((new_d_o, new_f_o, new_mult))
            terms = new_terms

        out_df = torch.zeros_like(out_g)
        for d_o, f_o, mult in terms:
            d_part = self._d_value(coords) if not d_o else self._d_mixed(
                coords, d_o,
            )
            if not f_o:
                f_part = inner_state.ops.value(inner_state, name)
            else:
                ax_t = tuple(f_o.keys())
                og_t = tuple(f_o[a] for a in ax_t)
                f_part = inner_state.ops.mixed_partial(
                    inner_state, name, ax_t, og_t,
                )
            out_df = out_df + mult * d_part * f_part

        return out_g + out_df

    def _d_mixed(self, coords: Tensor, axes_orders: dict[int, int]) -> Tensor:
        coords_g = coords.detach().requires_grad_(True)
        cur = self.distance_fn(coords_g)
        for ax, ord_ in axes_orders.items():
            for _ in range(ord_):
                cur, = torch.autograd.grad(
                    cur.sum(), coords_g,
                    create_graph=True, retain_graph=True,
                )
                cur = cur[..., ax]
        return cur


# --------- Mass-flux potential alias -------------------------------


class MassFluxPotentialField(VectorPotentialField):
    """Compressible-flow alias: ``rho u = curl(Psi)``.

    The implementation is identical to :class:`VectorPotentialField`;
    the only differences are the default component names
    (``("Psi1", "Psi2", "Psi3")`` for the mass-flux potential and
    ``("rhou", "rhov", "rhow")`` for the conserved velocities). The
    field guarantees :math:`\\nabla \\cdot (\\rho u) = 0` to machine
    precision regardless of the (possibly non-zero) ``rho``.
    """

    def __init__(
        self,
        *,
        base: FieldBase,
        Psi_components: tuple[str, str, str] = ("Psi1", "Psi2", "Psi3"),
        flux_names: tuple[str, str, str] = ("rhou", "rhov", "rhow"),
        passthrough_names: tuple[str, ...] = ("rho",),
        spatial_axes: tuple[str, str, str] = ("x", "y", "z"),
    ) -> None:
        super().__init__(
            base=base,
            A_components=Psi_components,
            velocity_names=flux_names,
            passthrough_names=passthrough_names,
            spatial_axes=spatial_axes,
        )


__all__ = [
    "HardBoundaryField",
    "MassFluxPotentialField",
    "energy_conserving_advection",
    "enstrophy_conserving_advection",
]
