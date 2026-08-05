# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hard-constraint boundary/initial-condition ansatz (JAX) -- twin of torch.

Bit-identical twin of :mod:`omnibias.torch.architectures.hardbc`. The boundary
condition is baked into the architecture,

.. math::

    u(x) = g(x) + b(x)\, N(x),

with a fixed **lift** ``g`` that matches the boundary data, a **boundary mask**
``b`` that vanishes exactly on the constrained set, and a free network ``N`` (a
:class:`omnibias.jax.architectures.JetMLP` /
:class:`~omnibias.jax.architectures.FourierFeatureMLP` / SIREN). The condition
then holds for every parameter value, so the boundary term drops out of the loss.

``g`` and ``b`` are polynomials whose exact jets come from
:func:`omnibias.jax.jet_mv.identity_jet`; the product ``b * N`` is the
multivariate Leibniz rule :func:`omnibias.jax.jet_mv.jet_multiply`. So every
``D^alpha u`` is exact and closed form -- no ``jax.grad`` for the differential
operator. :class:`HardConstraintField` is a frozen dataclass registered as a JAX
pytree (the wrapped network's arrays are the leaves; the fixed mask/lift are
static), so it flows through ``jax.grad`` / ``jax.jit`` for training.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.multi_index import multi_index_factorial, multi_indices
from omnibias.jax.architectures.pinn import FourierFeatureMLP, JetMLP
from omnibias.jax.jet_mv import (
    identity_jet,
    jet_gradient,
    jet_hessian,
    jet_multiply,
    mlp_jet_mv,
)

import jax
import jax.numpy as jnp
from jax import Array

JetNet = JetMLP | FourierFeatureMLP


@dataclass(frozen=True)
class AffineFactor:
    """A degree-1 boundary factor ``scale * x[axis] + offset`` (vanishes at its root)."""

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
    r"""Product of affine factors ``b(x) = \prod_k (s_k x[a_k] + o_k)``."""

    factors: tuple[AffineFactor, ...]

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError("BoundaryMask needs at least one factor")

    def value(self, x: Array) -> Array:
        """Mask value ``b(x)``, shape ``(...,)`` for input ``x`` of shape ``(..., D)``."""
        f0 = self.factors[0]
        out = f0.scale * x[..., f0.axis] + f0.offset
        for f in self.factors[1:]:
            out = out * (f.scale * x[..., f.axis] + f.offset)
        result: Array = out
        return result

    def point_jet(self, x0: Array, dim: int, order: int) -> Array:
        """Exact jet of ``b`` at ``x0``, shape ``(M,)`` (``M`` multi-indices)."""
        idj = identity_jet(x0, order)  # (M, D)
        m = idj.shape[0]
        e0 = (jnp.arange(m) == 0).astype(idj.dtype)
        f0 = self.factors[0]
        acc = f0.scale * idj[:, f0.axis] + f0.offset * e0
        for f in self.factors[1:]:
            fac = f.scale * idj[:, f.axis] + f.offset * e0
            acc = jet_multiply(acc, fac, dim, order)
        result: Array = acc
        return result


@dataclass(frozen=True)
class AffineLift:
    """Affine lift ``g(x) = W x + c`` matching the boundary data exactly.

    ``weight`` is the ``(out_dim, D)`` matrix and ``bias`` the ``(out_dim,)``
    vector, both plain Python floats (no trainable state; the boundary data is
    fixed). A constant lift is ``W = 0``; a 1-D Dirichlet interpolation is a single
    non-zero slope.
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

    def value(self, x: Array) -> Array:
        """Lift value ``g(x)``, shape ``(..., out_dim)``."""
        w = jnp.asarray(self.weight, dtype=x.dtype)
        c = jnp.asarray(self.bias, dtype=x.dtype)
        out: Array = x @ w.T + c
        return out

    def point_jet(self, x0: Array, dim: int, order: int) -> Array:
        """Exact jet of ``g`` at ``x0``, shape ``(M, out_dim)``."""
        idj = identity_jet(x0, order)  # (M, D)
        w = jnp.asarray(self.weight, dtype=x0.dtype)  # (out, D)
        c = jnp.asarray(self.bias, dtype=x0.dtype)  # (out,)
        g = idj @ w.T  # (M, out)
        m = g.shape[0]
        e0 = (jnp.arange(m) == 0).astype(g.dtype)
        out: Array = g + e0[:, None] * c
        return out


@dataclass(frozen=True)
class HardConstraintField:
    r"""Network wrapped so a boundary/initial condition holds exactly: ``u = g + b N``.

    ``net`` is a :class:`omnibias.jax.architectures.JetMLP` or
    :class:`~omnibias.jax.architectures.FourierFeatureMLP`. The field exposes the
    same exact readouts (:meth:`value`, :meth:`gradient`, :meth:`hessian`,
    :meth:`value_grad_hessian`, :meth:`jet`, :meth:`partials`) but evaluated on
    ``u = g + b N`` rather than ``net`` alone, staying closed form via jet-level
    add/multiply. Use :func:`dirichlet_interval`, :func:`homogeneous_box`,
    :func:`initial_value` for the common cases.
    """

    net: JetNet
    mask: BoundaryMask
    lift: AffineLift | None = None

    def __post_init__(self) -> None:
        in_dim = int(self.net.in_dim)
        out_dim = int(self.net.out_dim)
        for f in self.mask.factors:
            if f.axis >= in_dim:
                raise ValueError(
                    f"mask factor axis {f.axis} out of range for in_dim={in_dim}"
                )
        if self.lift is not None:
            if self.lift.out_dim != out_dim:
                raise ValueError(
                    f"lift out_dim {self.lift.out_dim} != network out_dim {out_dim}"
                )
            if self.lift.in_dim != in_dim:
                raise ValueError(
                    f"lift in_dim {self.lift.in_dim} != network in_dim {in_dim}"
                )

    @property
    def in_dim(self) -> int:
        return int(self.net.in_dim)

    @property
    def out_dim(self) -> int:
        return int(self.net.out_dim)

    def _point_jet(self, xi: Array, order: int) -> Array:
        net_jet = mlp_jet_mv(xi, self.net._layer_specs(), order)  # (M, out)
        mask_jet = self.mask.point_jet(xi, self.in_dim, order)  # (M,)
        u_jet = jet_multiply(mask_jet, net_jet, self.in_dim, order)
        if self.lift is not None:
            u_jet = u_jet + self.lift.point_jet(xi, self.in_dim, order)
        return u_jet

    def value(self, x: Array) -> Array:
        """Constrained field value ``u(x)``, shape ``(..., out_dim)``."""
        u = self.mask.value(x)[..., None] * self.net.value(x)
        if self.lift is not None:
            u = u + self.lift.value(x)
        result: Array = u
        return result

    def __call__(self, x: Array) -> Array:
        return self.value(x)

    def jet(self, x: Array, order: int) -> Array:
        """Batched multivariate jet of ``u``, shape ``(B, M, out_dim)``."""
        out: Array = jax.vmap(lambda xi: self._point_jet(xi, order))(x)
        return out

    def gradient(self, x: Array) -> Array:
        """Exact input gradient ``d u / d x_i``, shape ``(B, in_dim, out_dim)``."""
        d = self.in_dim
        out: Array = jax.vmap(lambda xi: jet_gradient(self._point_jet(xi, 1), d, 1))(x)
        return out

    def hessian(self, x: Array) -> Array:
        """Exact input Hessian, shape ``(B, in_dim, in_dim, out_dim)``."""
        d = self.in_dim
        out: Array = jax.vmap(lambda xi: jet_hessian(self._point_jet(xi, 2), d, 2))(x)
        return out

    def value_grad_hessian(self, x: Array) -> tuple[Array, Array, Array]:
        """One jet -> ``(value, gradient, Hessian)`` for 2nd-order PDE residuals."""
        d = self.in_dim

        def f(xi: Array) -> tuple[Array, Array, Array]:
            j = self._point_jet(xi, 2)
            return j[0], jet_gradient(j, d, 2), jet_hessian(j, d, 2)

        res = jax.vmap(f)(x)
        value_b: Array = res[0]
        grad_b: Array = res[1]
        hess_b: Array = res[2]
        return value_b, grad_b, hess_b

    def partials(self, x: Array, order: int) -> dict[tuple[int, ...], Array]:
        """All raw partials ``{alpha: D^alpha u(x)}`` to total ``order`` (``(B, out_dim)``)."""
        jet_b = self.jet(x, order)
        idx = multi_indices(self.in_dim, order)
        return {
            alpha: jet_b[:, i] * multi_index_factorial(alpha)
            for i, alpha in enumerate(idx)
        }


def _hc_tree_flatten(
    field: HardConstraintField,
) -> tuple[tuple[JetNet], tuple[BoundaryMask, AffineLift | None]]:
    return (field.net,), (field.mask, field.lift)


def _hc_tree_unflatten(
    aux: tuple[BoundaryMask, AffineLift | None], leaves: tuple[JetNet]
) -> HardConstraintField:
    mask, lift = aux
    (net,) = leaves
    return HardConstraintField(net=net, mask=mask, lift=lift)


jax.tree_util.register_pytree_node(
    HardConstraintField, _hc_tree_flatten, _hc_tree_unflatten
)


def dirichlet_interval(
    net: JetNet,
    lower: float,
    upper: float,
    *,
    lower_value: float = 0.0,
    upper_value: float = 0.0,
    axis: int = 0,
) -> HardConstraintField:
    r"""Hard 1-D Dirichlet BC: ``u(lower) = lower_value``, ``u(upper) = upper_value``.

    Mask ``b(x) = (x_axis - lower)(upper - x_axis)`` plus the linear lift
    interpolating the endpoint values, so the field matches the prescribed data at
    both ends exactly. The same boundary profile is applied to every output.
    """
    if upper <= lower:
        raise ValueError(f"upper ({upper}) must exceed lower ({lower})")
    in_dim = int(net.in_dim)
    out_dim = int(net.out_dim)
    mask = BoundaryMask(
        (AffineFactor(axis, 1.0, -lower), AffineFactor(axis, -1.0, upper))
    )
    lift: AffineLift | None = None
    if lower_value != 0.0 or upper_value != 0.0:
        slope = (upper_value - lower_value) / (upper - lower)
        intercept = lower_value - slope * lower
        row = tuple(slope if j == axis else 0.0 for j in range(in_dim))
        lift = AffineLift((row,) * out_dim, (intercept,) * out_dim)
    return HardConstraintField(net=net, mask=mask, lift=lift)


def homogeneous_box(
    net: JetNet,
    lows: Sequence[float],
    highs: Sequence[float],
) -> HardConstraintField:
    r"""Hard homogeneous Dirichlet BC ``u = 0`` on the boundary of a box.

    Mask ``\prod_i (x_i - lows_i)(highs_i - x_i)`` over the box
    ``\prod_i [lows_i, highs_i]``; the lift is zero, so ``u = b N``.
    """
    in_dim = int(net.in_dim)
    if len(lows) != in_dim or len(highs) != in_dim:
        raise ValueError(
            f"lows/highs must have length in_dim={in_dim}, "
            f"got {len(lows)} and {len(highs)}"
        )
    factors: list[AffineFactor] = []
    for i in range(in_dim):
        if highs[i] <= lows[i]:
            raise ValueError(
                f"highs[{i}] ({highs[i]}) must exceed lows[{i}] ({lows[i]})"
            )
        factors.append(AffineFactor(i, 1.0, -float(lows[i])))
        factors.append(AffineFactor(i, -1.0, float(highs[i])))
    return HardConstraintField(net=net, mask=BoundaryMask(tuple(factors)), lift=None)


def initial_value(
    net: JetNet,
    *,
    t_axis: int = -1,
    t0: float = 0.0,
    value: float = 0.0,
) -> HardConstraintField:
    r"""Hard (constant) initial condition ``u(x, t0) = value``.

    Mask ``t - t0`` (one-sided: zero only on the initial slice) plus a constant
    lift. For a non-constant initial profile, build :class:`HardConstraintField`
    directly with a custom :class:`AffineLift` / :class:`BoundaryMask`.
    """
    in_dim = int(net.in_dim)
    out_dim = int(net.out_dim)
    ax = t_axis if t_axis >= 0 else in_dim + t_axis
    if not (0 <= ax < in_dim):
        raise ValueError(f"t_axis {t_axis} out of range for in_dim={in_dim}")
    mask = BoundaryMask((AffineFactor(ax, 1.0, -t0),))
    lift: AffineLift | None = None
    if value != 0.0:
        zeros = (0.0,) * in_dim
        lift = AffineLift((zeros,) * out_dim, (float(value),) * out_dim)
    return HardConstraintField(net=net, mask=mask, lift=lift)


__all__ = [
    "AffineFactor",
    "AffineLift",
    "BoundaryMask",
    "HardConstraintField",
    "dirichlet_interval",
    "homogeneous_box",
    "initial_value",
]
