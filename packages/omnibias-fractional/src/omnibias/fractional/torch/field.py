# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form per-axis field fractional partial + fractional-diffusion residual (torch).

This module lifts the closed-form analytic operator
(:func:`omnibias.fractional.torch.ops.analytic.fractional_derivative`) onto the
:class:`~omnibias.fields._core.state.FieldState` substrate: it is the fractional
twin of a per-axis :func:`~omnibias.fields.torch.ops.basic.derivative`.

For a field ``u`` and a chosen input axis ``i`` with lower terminal ``a``, the
fractional partial ``{}_a D_{x_i}^{\alpha} u`` is evaluated by expanding the
field's Taylor jet **along axis i about the terminal** -- the field is
re-evaluated at the collocation coordinates with axis ``i`` pinned to ``a`` (the
other axes held at their collocation values), giving
``a_k = \partial_{x_i}^k u(\dots, a, \dots) / k!`` for ``k = 0..order`` -- and
summing the gamma-ratio series at ``x_i`` (offset ``t = x_i - a \ge 0``):

.. math::

    {}_a D_{x_i}^{\alpha} u(x)
        = \sum_{k=0}^{N} a_k\,\frac{\Gamma(k+1)}{\Gamma(k+1-\alpha)}\,t^{\,k-\alpha}.

This is **closed form on the analytic-function class**: an order-``N`` truncation
(``N = order``), exact for slices that are polynomials of degree ``<= N`` along
axis ``i`` and otherwise accurate within the Taylor radius about ``a``. It is
differentiable in the order ``alpha`` (a tensor / ``nn.Parameter`` /
:class:`~omnibias.fractional.torch.order.LearnableOrder`) and in the field
parameters, so it composes with autograd. A plain-number *integer* ``alpha`` is
steered to the exact closed-form derivative tower (avoids the Gamma-pole
``0 * inf``); ``alpha = 0`` recovers the field value.

It requires a field with a closed-form per-axis derivative tower (e.g. the
``one_layer`` field), reached through ``state.ops.derivative``. Unlike the
grid / spectral operators in :mod:`omnibias.fractional.torch.ops.fractional`
there is no grid and no history.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch
from omnibias.fractional.torch.ops.analytic import _gamma_ratio
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

    from omnibias.fields._core.state import FieldState


def _is_integer_order(alpha: float | Tensor) -> bool:
    """A plain-number non-negative integer order (tensor orders stay differentiable)."""
    if isinstance(alpha, Tensor):
        return False
    val = float(alpha)
    return val >= 0.0 and val == round(val)


def field_fractional_partial(
    state: FieldState,
    name: str,
    *,
    axis: int | str,
    alpha: float | Tensor,
    order: int,
    a: float = 0.0,
    kind: str = "riemann_liouville",
    x_eval: Tensor | None = None,
) -> Tensor:
    r"""Closed-form fractional partial ``{}_a D_{x_axis}^{alpha} u`` of shape ``(B,)``.

    Parameters
    ----------
    state, name
        The evaluated field and the scalar component ``u`` to differentiate.
    axis
        Input axis (name or index) to take the fractional partial along.
    alpha
        Fractional order. A ``Tensor`` (e.g. an ``nn.Parameter`` or a
        :class:`~omnibias.fractional.torch.order.LearnableOrder` output) keeps
        the gradient path to the order; a plain-number integer is steered to the
        exact derivative tower.
    order
        Jet truncation order ``N`` (number of Taylor coefficients minus one).
        Exact for slices that are degree ``<= N`` polynomials along ``axis``.
    a
        Lower terminal / expansion point along ``axis`` (branch point of
        ``t^{k-alpha}``). The field is re-evaluated with ``axis`` pinned to ``a``.
    kind
        ``"riemann_liouville"`` (sum over all ``k``) or ``"caputo"`` (drop
        ``k < ceil(alpha)``).
    x_eval
        Evaluation points along ``axis`` (shape ``(B,)``); defaults to
        ``state.coords[:, axis]``. ``t = x_eval - a >= 0`` is required.
    """
    if kind not in ("riemann_liouville", "caputo"):
        raise ValueError(
            f"kind must be 'riemann_liouville' or 'caputo', got {kind!r}"
        )
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    i = state.coordinate_spec.axis_index(axis)

    # Integer order at the collocation points reduces to the exact derivative
    # tower (Gamma poles make the series 0 * inf); tensor orders stay on the
    # differentiable series path.
    if x_eval is None and _is_integer_order(alpha):
        return state.ops.derivative(state, name, axis=i, order=int(alpha))

    coords = state.coords
    # Pin axis ``i`` to the terminal ``a`` (autograd-friendly, no in-place write):
    # expand the field's jet along axis i about a while holding the other axes.
    col_a = torch.full_like(coords[:, i : i + 1], float(a))
    coords_a = torch.cat([coords[:, :i], col_a, coords[:, i + 1 :]], dim=1)
    state_a = state.field(coords_a)

    rows = [
        state_a.ops.derivative(state_a, name, axis=i, order=k) / math.factorial(k)
        for k in range(order + 1)
    ]
    jet = torch.stack(rows, dim=0)  # (order+1, B): a_k = d_i^k u(...,a,...) / k!

    if isinstance(alpha, Tensor):
        alpha_t = alpha.to(dtype=jet.dtype, device=jet.device)
    else:
        alpha_t = torch.tensor(float(alpha), dtype=jet.dtype, device=jet.device)

    k = torch.arange(order + 1, dtype=jet.dtype, device=jet.device)
    ratio = _gamma_ratio(k, alpha_t)
    if kind == "caputo":
        ratio = ratio * (k >= torch.ceil(alpha_t)).to(jet.dtype)

    x = coords[:, i] if x_eval is None else torch.as_tensor(
        x_eval, dtype=jet.dtype, device=jet.device
    )
    t = (x - a).reshape(-1)  # (B,)

    # Batch-aligned contraction: each jet column pairs with its own t (not the
    # outer-product broadcast of the standalone jet op). Masked terms (ratio == 0)
    # contribute 0 even at the terminal (t = 0), where ``0**negative`` is inf.
    ratio_e = ratio.unsqueeze(1)
    powers = t.unsqueeze(0) ** (k.unsqueeze(1) - alpha_t)  # (order+1, B)
    terms = torch.where(ratio_e == 0.0, torch.zeros_like(powers), jet * ratio_e * powers)
    out: Tensor = terms.sum(dim=0)
    return out


class FractionalDiffusionOutput(NamedTuple):
    """Residual + diagnostics of :func:`fractional_diffusion_residual`."""

    residual: Tensor
    diag: dict[str, float]


def fractional_diffusion_residual(
    state: FieldState,
    *,
    alphas: Sequence[float | Tensor],
    order: int,
    component: str = "u",
    kind: str = "caputo",
    a: float = 0.0,
    source: Callable[[FieldState], Tensor] | None = None,
) -> FractionalDiffusionOutput:
    r"""Space-fractional diffusion residual ``u_t - sum_a {}_a D_{x_a}^{alpha_a} u - s``.

    A closed-form fractional PINN residual: the time derivative is the exact
    closed-form ``u_t`` and each spatial term is a per-axis
    :func:`field_fractional_partial` (Caputo by default). ``alphas`` gives one
    order per spatial axis (in :attr:`CoordinateSpec.spatial_axes` order).
    Returns a ``(residual, diag)`` pair; reduce ``residual`` to a scalar loss in
    the training loop (``(out.residual ** 2).mean()``).
    """
    spec = state.coordinate_spec
    if spec.time_axis is None:
        raise ValueError(
            "fractional_diffusion_residual requires a time axis in the coordinate spec"
        )
    spatial = spec.spatial_axes
    alphas = tuple(alphas)
    if len(alphas) != len(spatial):
        raise ValueError(
            f"alphas must have one order per spatial axis; got {len(alphas)} "
            f"for spatial axes {spatial!r}"
        )

    u_t = state.ops.derivative(state, component, axis=spec.time_axis, order=1)
    frac: Tensor | None = None
    for ax, al in zip(spatial, alphas, strict=True):
        term = field_fractional_partial(
            state, component, axis=ax, alpha=al, order=order, a=a, kind=kind
        )
        frac = term if frac is None else frac + term
    assert frac is not None, "at least one spatial axis is required"

    residual = u_t - frac
    if source is not None:
        residual = residual - source(state)
    diag = {"mean_sq_residual": float((residual.detach() ** 2).mean())}
    return FractionalDiffusionOutput(residual=residual, diag=diag)


__all__ = [
    "FractionalDiffusionOutput",
    "field_fractional_partial",
    "fractional_diffusion_residual",
]
