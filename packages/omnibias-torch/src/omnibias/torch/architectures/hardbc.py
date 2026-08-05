# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hard-constraint boundary/initial-condition ansatz (torch).

A boundary-value problem is usually trained with a *soft* penalty: the PINN loss
adds a term ``|u - g|^2`` on the boundary and hopes the optimiser drives it to
zero. The boundary condition is then only approximate, it fights the PDE
residual for gradient budget, and the multi-objective loss is ill-conditioned.

The *hard-constraint* ansatz removes the boundary term entirely by baking the
condition into the architecture (Lagaris et al. 1998; Sukumar & Srivastava
2022):

.. math::

    u(x) = g(x) + b(x)\, N(x),

where

* ``g`` (the **lift**) is a fixed smooth function that already satisfies the
  boundary data (``g = g_\partial`` on ``\partial\Omega``);
* ``b`` (the **boundary mask** / approximate distance function) vanishes
  *exactly* on the constrained set (``b = 0`` there);
* ``N`` is the free neural network (any :class:`omnibias.torch.architectures.JetMLP`
  / :class:`~omnibias.torch.architectures.FourierFeatureMLP` / SIREN).

Then ``u`` satisfies the boundary condition for **every** value of the network
parameters, so only the PDE residual remains in the loss. This pairs directly
with the second-order optimiser in :mod:`omnibias.torch.optim` (one fewer loss
term to balance).

The omnibias-native part is that the ansatz is assembled at the *jet* level. The
lift ``g`` and the mask ``b`` are polynomials whose exact Taylor jets come from
:func:`omnibias.torch.jet_mv.identity_jet`, and the product ``b * N`` is the
multivariate Leibniz rule :func:`omnibias.torch.jet_mv.jet_multiply`. Hence the
constrained field keeps the closed-form, arbitrary-order derivative contract:
``D^alpha u`` is exact with no ``torch.autograd.grad`` for the differential
operator. :class:`HardConstraintField` reuses the readout machinery of
:class:`omnibias.torch.architectures.pinn._JetMLPCore`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.torch.activations.registry import ActivationSpec
from omnibias.torch.architectures.pinn import _JetMLPCore
from omnibias.torch.jet_mv import identity_jet, jet_multiply

import torch
from torch import Tensor


@dataclass(frozen=True)
class AffineFactor:
    """A degree-1 boundary factor ``scale * x[axis] + offset``.

    The factor vanishes on the hyperplane ``x[axis] = -offset / scale``; a product
    of such factors is zero wherever *any* of them is, which is how a
    :class:`BoundaryMask` encodes the constrained set.
    """

    axis: int
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.axis < 0:
            raise ValueError(f"axis must be >= 0, got {self.axis}")
        if self.scale == 0.0:
            raise ValueError("scale must be non-zero (a constant factor is not a mask)")


@dataclass(frozen=True)
class BoundaryMask:
    r"""Product of affine factors ``b(x) = \prod_k (s_k x[a_k] + o_k)``.

    ``b`` vanishes exactly on the union of the factor hyperplanes -- the set on
    which the hard constraint is imposed. Both the batched value and the exact
    multivariate jet are provided so the mask plugs into the jet-level product.
    """

    factors: tuple[AffineFactor, ...]

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError("BoundaryMask needs at least one factor")

    def value(self, x: Tensor) -> Tensor:
        """Mask value ``b(x)``, shape ``(...,)`` for input ``x`` of shape ``(..., D)``."""
        f0 = self.factors[0]
        out = f0.scale * x[..., f0.axis] + f0.offset
        for f in self.factors[1:]:
            out = out * (f.scale * x[..., f.axis] + f.offset)
        return out

    def point_jet(self, x0: Tensor, dim: int, order: int) -> Tensor:
        """Exact jet of ``b`` at ``x0``, shape ``(M,)`` (``M`` multi-indices)."""
        idj = identity_jet(x0, order)  # (M, D)
        m = idj.shape[0]
        e0 = (torch.arange(m, device=idj.device) == 0).to(idj.dtype)
        f0 = self.factors[0]
        acc = f0.scale * idj[:, f0.axis] + f0.offset * e0
        for f in self.factors[1:]:
            fac = f.scale * idj[:, f.axis] + f.offset * e0
            acc = jet_multiply(acc, fac, dim, order)
        return acc


@dataclass(frozen=True)
class AffineLift:
    """Affine lift ``g(x) = W x + c`` matching the boundary data exactly.

    ``weight`` is the ``(out_dim, D)`` matrix ``W`` and ``bias`` the ``(out_dim,)``
    vector ``c``, both stored as plain Python floats so the lift is fully
    device/dtype-agnostic and carries no trainable state (the boundary data is
    fixed). A constant lift is ``W = 0``; a 1-D Dirichlet interpolation is a
    single non-zero slope.
    """

    weight: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.weight) != len(self.bias):
            raise ValueError(
                f"weight has {len(self.weight)} rows but bias has {len(self.bias)}"
            )
        if self.weight:
            width = len(self.weight[0])
            if any(len(row) != width for row in self.weight):
                raise ValueError("weight rows must all have the same length")

    @property
    def out_dim(self) -> int:
        return len(self.bias)

    @property
    def in_dim(self) -> int:
        return len(self.weight[0]) if self.weight else 0

    def value(self, x: Tensor) -> Tensor:
        """Lift value ``g(x)``, shape ``(..., out_dim)``."""
        w = torch.tensor(self.weight, dtype=x.dtype, device=x.device)
        c = torch.tensor(self.bias, dtype=x.dtype, device=x.device)
        return x @ w.t() + c

    def point_jet(self, x0: Tensor, dim: int, order: int) -> Tensor:
        """Exact jet of ``g`` at ``x0``, shape ``(M, out_dim)``."""
        idj = identity_jet(x0, order)  # (M, D)
        w = torch.tensor(self.weight, dtype=x0.dtype, device=x0.device)  # (out, D)
        c = torch.tensor(self.bias, dtype=x0.dtype, device=x0.device)  # (out,)
        g = idj @ w.t()  # (M, out): row 0 = W x0, unit rows = columns of W
        m = g.shape[0]
        e0 = (torch.arange(m, device=g.device) == 0).to(g.dtype)
        return g + e0[:, None] * c


class HardConstraintField(_JetMLPCore):
    r"""Network wrapped so a boundary/initial condition holds exactly: ``u = g + b N``.

    ``net`` is any :class:`omnibias.torch.architectures.pinn._JetMLPCore` (a
    :class:`~omnibias.torch.architectures.JetMLP`,
    :class:`~omnibias.torch.architectures.FourierFeatureMLP`, or SIREN). The
    constrained field exposes the same exact readouts as the base network --
    :meth:`value`, :meth:`gradient`, :meth:`hessian`, :meth:`value_grad_hessian`,
    :meth:`jet`, :meth:`partials` -- but every one is taken on ``u = g + b N``
    rather than on ``net`` alone, and stays closed form because the wrapping is a
    jet-level add/multiply.

    Use the :func:`dirichlet_interval`, :func:`homogeneous_box` and
    :func:`initial_value` helpers for the common cases, or pass a custom
    ``mask`` / ``lift`` for full control.
    """

    net: _JetMLPCore
    mask: BoundaryMask
    lift: AffineLift | None

    def __init__(
        self,
        net: _JetMLPCore,
        mask: BoundaryMask,
        lift: AffineLift | None = None,
    ) -> None:
        super().__init__()
        self.net = net
        self.in_dim = int(net.in_dim)
        self.out_dim = int(net.out_dim)
        self.mask = mask
        self.lift = lift
        for f in mask.factors:
            if f.axis >= self.in_dim:
                raise ValueError(
                    f"mask factor axis {f.axis} out of range for in_dim={self.in_dim}"
                )
        if lift is not None:
            if lift.out_dim != self.out_dim:
                raise ValueError(
                    f"lift out_dim {lift.out_dim} != network out_dim {self.out_dim}"
                )
            if lift.in_dim != self.in_dim:
                raise ValueError(
                    f"lift in_dim {lift.in_dim} != network in_dim {self.in_dim}"
                )

    def _layer_specs(
        self,
    ) -> list[tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | None]]:
        """Delegate the fast-path check to the wrapped network's layers."""
        return self.net._layer_specs()

    def _point_jet(self, xi: Tensor, order: int) -> Tensor:
        net_jet = self.net._point_jet(xi, order)  # (M, out_dim)
        mask_jet = self.mask.point_jet(xi, self.in_dim, order)  # (M,)
        u_jet = jet_multiply(mask_jet, net_jet, self.in_dim, order)
        if self.lift is not None:
            u_jet = u_jet + self.lift.point_jet(xi, self.in_dim, order)
        return u_jet

    def value(self, x: Tensor) -> Tensor:
        u = self.mask.value(x).unsqueeze(-1) * self.net.value(x)
        if self.lift is not None:
            u = u + self.lift.value(x)
        return u


def dirichlet_interval(
    net: _JetMLPCore,
    lower: float,
    upper: float,
    *,
    lower_value: float = 0.0,
    upper_value: float = 0.0,
    axis: int = 0,
) -> HardConstraintField:
    r"""Hard 1-D Dirichlet BC: ``u(lower) = lower_value``, ``u(upper) = upper_value``.

    Builds the mask ``b(x) = (x_axis - lower)(upper - x_axis)`` (zero at both
    ends) and the linear lift that interpolates the two endpoint values, so the
    field equals the prescribed data at ``lower`` and ``upper`` exactly. The same
    boundary profile is applied to every output component.
    """
    if upper <= lower:
        raise ValueError(f"upper ({upper}) must exceed lower ({lower})")
    mask = BoundaryMask(
        (AffineFactor(axis, 1.0, -lower), AffineFactor(axis, -1.0, upper))
    )
    lift: AffineLift | None = None
    if lower_value != 0.0 or upper_value != 0.0:
        slope = (upper_value - lower_value) / (upper - lower)
        intercept = lower_value - slope * lower
        row = tuple(slope if j == axis else 0.0 for j in range(net.in_dim))
        lift = AffineLift((row,) * net.out_dim, (intercept,) * net.out_dim)
    return HardConstraintField(net, mask, lift)


def homogeneous_box(
    net: _JetMLPCore,
    lows: Sequence[float],
    highs: Sequence[float],
) -> HardConstraintField:
    r"""Hard homogeneous Dirichlet BC ``u = 0`` on the boundary of a box.

    The box is ``\prod_i [lows_i, highs_i]`` and the mask is the product
    ``\prod_i (x_i - lows_i)(highs_i - x_i)``, which vanishes on every face. The
    lift is zero, so ``u = b N``.
    """
    if len(lows) != net.in_dim or len(highs) != net.in_dim:
        raise ValueError(
            f"lows/highs must have length in_dim={net.in_dim}, "
            f"got {len(lows)} and {len(highs)}"
        )
    factors: list[AffineFactor] = []
    for i in range(net.in_dim):
        if highs[i] <= lows[i]:
            raise ValueError(
                f"highs[{i}] ({highs[i]}) must exceed lows[{i}] ({lows[i]})"
            )
        factors.append(AffineFactor(i, 1.0, -float(lows[i])))
        factors.append(AffineFactor(i, -1.0, float(highs[i])))
    return HardConstraintField(net, BoundaryMask(tuple(factors)), None)


def initial_value(
    net: _JetMLPCore,
    *,
    t_axis: int = -1,
    t0: float = 0.0,
    value: float = 0.0,
) -> HardConstraintField:
    r"""Hard (constant) initial condition ``u(x, t0) = value``.

    The mask is the one-sided factor ``t - t0`` (zero only on the initial slice,
    leaving later times unconstrained) and the lift is the constant ``value``. For
    a non-constant initial profile, compose a custom :class:`AffineLift` (or a
    product :class:`BoundaryMask`) and build :class:`HardConstraintField` directly.
    """
    ax = t_axis if t_axis >= 0 else net.in_dim + t_axis
    if not (0 <= ax < net.in_dim):
        raise ValueError(f"t_axis {t_axis} out of range for in_dim={net.in_dim}")
    mask = BoundaryMask((AffineFactor(ax, 1.0, -t0),))
    lift: AffineLift | None = None
    if value != 0.0:
        zeros = (0.0,) * net.in_dim
        lift = AffineLift((zeros,) * net.out_dim, (float(value),) * net.out_dim)
    return HardConstraintField(net, mask, lift)


__all__ = [
    "AffineFactor",
    "AffineLift",
    "BoundaryMask",
    "HardConstraintField",
    "dirichlet_interval",
    "homogeneous_box",
    "initial_value",
]
