# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""L^2-norm-conservation cage for the torch backend.

This module ships two ways to enforce wavefunction normalisation
:math:`\int |\psi(x)|^2\,dx = 1`:

1. :class:`NormConservationField` -- a *hard* cage. Wraps a base field
   carrying the wavefunction group, computes the :math:`L^2` norm on a
   user-supplied fixed quadrature grid at every forward call, and
   divides component values + every spatial / temporal derivative by
   the norm. The result is that the cage's output wavefunction has unit
   :math:`L^2` norm by construction (to quadrature accuracy).

2. :func:`norm_loss` -- a *soft* loss term ``(integral |psi|^2 - 1)^2``
   useful as a regulariser when the user does not want the hard
   projection.

Mathematical sketch of the hard cage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a base field outputting the raw amplitudes
:math:`\tilde\psi_R(x), \tilde\psi_I(x)`, define

.. math::

    N^2 = \sum_q w_q \big(\tilde\psi_R(x_q)^2 + \tilde\psi_I(x_q)^2\big),

then expose the cage components

.. math::

    \psi_R(x) = \tilde\psi_R(x) / N,
    \qquad
    \psi_I(x) = \tilde\psi_I(x) / N.

Since :math:`N` is independent of :math:`x`, every spatial / temporal
derivative is simply scaled: :math:`\partial^k \psi / N`. This
preserves the omnibias closed-form derivative path; the cage adds one
extra base-field forward pass per residual evaluation.
"""

from __future__ import annotations

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.cage.incompressible import _CageFieldBase
from omnibias.pinn.torch.fields.base import FieldBase, _import_torch_ops
from torch import Tensor


class NormConservationField(_CageFieldBase):
    r"""Hard :math:`L^2`-norm conservation for a complex wavefunction.

    Parameters
    ----------
    base
        Underlying :class:`FieldBase` whose :class:`ComponentSpec` carries
        a wavefunction group with exactly two real components in
        ``(re, im)`` order. Build the base with
        :func:`omnibias.qpinn.make_psi_components`.
    quadrature_coords
        Fixed quadrature grid, shape ``(B_q, D)`` where ``D ==
        coordinate_spec.ndim``. Required.
    quadrature_weights
        Quadrature weights, shape ``(B_q,)``. The sum
        ``(weights * density)`` over the grid approximates the
        :math:`L^2` norm squared. Must be non-negative.
    psi_group
        Name of the wavefunction group on the base field. Default
        ``"psi"``.

    Notes
    -----
    The cage acts as an identity on the *base* component names: if the
    base has components ``("psi_re", "psi_im")``, the cage exposes the
    same names. Other components on the base are passed through
    unchanged. The cage's :class:`ComponentSpec` is therefore identical
    to the base's; downstream operators and residuals can treat a caged
    field as a drop-in replacement.

    The quadrature grid is registered as a buffer so it travels through
    ``.to(...)`` / ``.cuda()`` calls. It is *not* a parameter -- the
    grid is held fixed throughout training.
    """

    quadrature_coords: Tensor
    quadrature_weights: Tensor
    psi_group_name: str

    def __init__(
        self,
        *,
        base: FieldBase,
        quadrature_coords: Tensor,
        quadrature_weights: Tensor,
        psi_group: str = "psi",
    ) -> None:
        if not base.components.is_group(psi_group):
            raise ValueError(
                f"base does not have a wavefunction group {psi_group!r}; "
                "build it with omnibias.qpinn.make_psi_components"
            )
        members = base.components.group_members(psi_group)
        if len(members) != 2:
            raise ValueError(
                f"wavefunction group {psi_group!r} must have exactly 2 "
                f"components (re, im); got {len(members)}: {members!r}"
            )
        if quadrature_coords.dim() != 2:
            raise ValueError(
                f"quadrature_coords must be 2D (B_q, D), got shape "
                f"{tuple(quadrature_coords.shape)}"
            )
        if quadrature_weights.dim() != 1:
            raise ValueError(
                f"quadrature_weights must be 1D (B_q,), got shape "
                f"{tuple(quadrature_weights.shape)}"
            )
        if quadrature_coords.shape[0] != quadrature_weights.shape[0]:
            raise ValueError(
                f"quadrature batch mismatch: coords {quadrature_coords.shape[0]} "
                f"vs weights {quadrature_weights.shape[0]}"
            )
        if quadrature_coords.shape[-1] != base.coordinate_spec.ndim:
            raise ValueError(
                f"quadrature_coords last dim {quadrature_coords.shape[-1]} != "
                f"coordinate_spec.ndim {base.coordinate_spec.ndim}"
            )
        if torch.any(quadrature_weights < 0):
            raise ValueError("quadrature_weights must be non-negative")

        # The cage's component spec is identical to the base's: the cage
        # is an algebraic-identity wrapper on the names.
        passthrough = tuple(
            n for n in base.components.names if n not in members
        )
        super().__init__(
            base=base,
            velocity_names=members,
            passthrough_names=passthrough,
            groups={psi_group: members},
        )
        self.psi_group_name = psi_group
        self.register_buffer("quadrature_coords", quadrature_coords)
        self.register_buffer("quadrature_weights", quadrature_weights)

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
        re_name, im_name = self.velocity_names

        quad_state = self.base.evaluate(self.quadrature_coords)
        psi_re_q = quad_state.ops.value(quad_state, re_name)
        psi_im_q = quad_state.ops.value(quad_state, im_name)
        density_q = psi_re_q * psi_re_q + psi_im_q * psi_im_q
        norm_sq = (self.quadrature_weights * density_q).sum()
        # tiny instead of 0 for AD safety; the cage user is expected to
        # initialise the base such that ||psi_tilde|| > 0.
        eps = torch.finfo(norm_sq.dtype).tiny
        norm = torch.sqrt(norm_sq + eps)

        inner_state = self.base.evaluate(coords)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_torch_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={"_cage_inner_state": inner_state, "_norm": norm},
        )

    # ----- dispatched value / derivative / mixed_partial --------------

    def value_component(self, state: FieldState, name: str) -> Tensor:
        inner = state.extra["_cage_inner_state"]
        norm = state.extra["_norm"]
        v = inner.ops.value(inner, name)
        if name in self.velocity_names:
            return v / norm
        if name in self.passthrough_names:
            return v
        raise KeyError(
            f"{name!r} is neither a wavefunction component "
            f"{self.velocity_names!r} nor a passthrough "
            f"{self.passthrough_names!r}"
        )

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Tensor:
        if order == 0:
            return self.value_component(state, name)
        inner = state.extra["_cage_inner_state"]
        norm = state.extra["_norm"]
        d = inner.ops.derivative(inner, name, axis=axis, order=order)
        if name in self.velocity_names:
            return d / norm
        if name in self.passthrough_names:
            return d
        raise KeyError(name)

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Tensor:
        inner = state.extra["_cage_inner_state"]
        norm = state.extra["_norm"]
        d = inner.ops.mixed_partial(inner, name, axes, orders)
        if name in self.velocity_names:
            return d / norm
        if name in self.passthrough_names:
            return d
        raise KeyError(name)


def make_norm_conservation_field(
    *,
    base: FieldBase,
    quadrature_coords: Tensor,
    quadrature_weights: Tensor,
    psi_group: str = "psi",
) -> NormConservationField:
    r"""Build a :class:`NormConservationField` from a base field + quadrature.

    Parameters
    ----------
    base
        Base field with a wavefunction group; build via
        :func:`omnibias.qpinn.make_psi_components`.
    quadrature_coords
        Fixed quadrature grid, shape ``(B_q, D)``.
    quadrature_weights
        Quadrature weights, shape ``(B_q,)``.
    psi_group
        Name of the wavefunction group on the base. Default ``"psi"``.

    Returns
    -------
    NormConservationField
        The caged field. Its :class:`ComponentSpec` is identical to
        ``base.components``; only the values / derivatives are
        renormalised.

    Examples
    --------
    .. code-block:: python

        from omnibias.qpinn import make_psi_components
        from omnibias.qpinn.torch.cage import make_norm_conservation_field
        from omnibias.pinn._core.coords import CoordinateSpec
        from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
        import torch

        coord = CoordinateSpec(("x",))
        spec = make_psi_components(name="psi")
        base = OneLayerVectorField(
            coordinate_spec=coord, components=spec, hidden=16, base="gaussian",
        )
        x_grid = torch.linspace(-5, 5, 401, dtype=torch.float64).unsqueeze(-1)
        weights = torch.full((401,), 10.0 / 401, dtype=torch.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x_grid, quadrature_weights=weights,
        )
    """
    return NormConservationField(
        base=base,
        quadrature_coords=quadrature_coords,
        quadrature_weights=quadrature_weights,
        psi_group=psi_group,
    )


def norm_loss(
    state: FieldState,
    *,
    group: str = "psi",
    quadrature_weights: Tensor | None = None,
    target_norm: float = 1.0,
) -> Tensor:
    r"""Soft norm-conservation loss
    :math:`\big(\int |\psi|^2\,dx - n\big)^2`.

    Parameters
    ----------
    state
        :class:`FieldState` carrying the wavefunction group.
    group
        Wavefunction group name. Default ``"psi"``.
    quadrature_weights
        Per-collocation-point weights, shape ``(B,)``. If ``None``, use
        the uniform-mean approximation ``1 / B``.
    target_norm
        Desired :math:`L^2` norm-squared. Default 1.0.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    re_name = f"{group}_re"
    im_name = f"{group}_im"
    psi_re = state.ops.value(state, re_name)
    psi_im = state.ops.value(state, im_name)
    density = psi_re * psi_re + psi_im * psi_im
    if quadrature_weights is None:
        norm_sq = density.mean()
    else:
        if quadrature_weights.shape != density.shape:
            raise ValueError(
                f"quadrature_weights shape {tuple(quadrature_weights.shape)} "
                f"!= density shape {tuple(density.shape)}"
            )
        norm_sq = (quadrature_weights * density).sum()
    return (norm_sq - target_norm) ** 2


__all__ = [
    "NormConservationField",
    "make_norm_conservation_field",
    "norm_loss",
]
