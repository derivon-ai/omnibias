# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form per-axis field fractional partial + fractional-diffusion residual (jax).

Bit-identical twin of :mod:`omnibias.fractional.torch.field`. Given a field
``u`` and an input axis ``i`` with lower terminal ``a``, the fractional partial
``{}_a D_{x_i}^{\alpha} u`` is evaluated by expanding the field's Taylor jet
along axis ``i`` about the terminal (re-evaluating the field with axis ``i``
pinned to ``a``, other axes held) and summing the gamma-ratio series at ``x_i``:

.. math::

    {}_a D_{x_i}^{\alpha} u(x)
        = \sum_{k=0}^{N} a_k\,\frac{\Gamma(k+1)}{\Gamma(k+1-\alpha)}\,t^{\,k-\alpha},
    \qquad a_k = \partial_{x_i}^k u(\dots,a,\dots)/k!,\ t = x_i - a \ge 0.

Closed form on the analytic-function class (order-``N`` truncation, exact for
degree ``<= N`` polynomial slices), differentiable in the order ``alpha`` and
the field parameters. See the torch twin for the full notes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array
from omnibias.fractional.jax.ops.analytic import _gamma_ratio

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Sequence

    from omnibias.fields._core.state import FieldState


def _is_integer_order(alpha: float | Array) -> bool:
    """A plain-number non-negative integer order (array/traced orders stay differentiable)."""
    if isinstance(alpha, Array):
        return False
    val = float(alpha)
    return val >= 0.0 and val == round(val)


def field_fractional_partial(
    state: FieldState,
    name: str,
    *,
    axis: int | str,
    alpha: float | Array,
    order: int,
    a: float = 0.0,
    kind: str = "riemann_liouville",
    x_eval: Array | None = None,
) -> Array:
    r"""Closed-form fractional partial ``{}_a D_{x_axis}^{alpha} u`` of shape ``(B,)``.

    See :func:`omnibias.fractional.torch.field.field_fractional_partial` for the
    full parameter semantics; this is the bit-identical JAX twin.
    """
    if kind not in ("riemann_liouville", "caputo"):
        raise ValueError(
            f"kind must be 'riemann_liouville' or 'caputo', got {kind!r}"
        )
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    i = state.coordinate_spec.axis_index(axis)

    if x_eval is None and _is_integer_order(alpha):
        return state.ops.derivative(state, name, axis=i, order=int(alpha))

    coords = state.coords
    # Pin axis ``i`` to the terminal ``a`` (immutable update, autograd-friendly).
    coords_a = coords.at[:, i].set(float(a))
    state_a = state.field(coords_a)

    rows = [
        state_a.ops.derivative(state_a, name, axis=i, order=k) / math.factorial(k)
        for k in range(order + 1)
    ]
    jet = jnp.stack(rows, axis=0)  # (order+1, B): a_k = d_i^k u(...,a,...) / k!

    alpha_t = jnp.asarray(alpha, dtype=jet.dtype)

    k = jnp.arange(order + 1, dtype=jet.dtype)
    ratio = _gamma_ratio(k, alpha_t)
    if kind == "caputo":
        ratio = ratio * (k >= jnp.ceil(alpha_t)).astype(jet.dtype)

    x = coords[:, i] if x_eval is None else jnp.asarray(x_eval, dtype=jet.dtype)
    t = (x - a).reshape(-1)  # (B,)

    # Masked terms (ratio == 0) contribute 0 even at the terminal (t = 0), where
    # ``0**negative`` is inf.
    ratio_e = ratio[:, None]
    powers = t[None, :] ** (k[:, None] - alpha_t)  # (order+1, B)
    terms = jnp.where(ratio_e == 0.0, 0.0, jet * ratio_e * powers)
    out: Array = terms.sum(axis=0)
    return out


class FractionalDiffusionOutput(NamedTuple):
    """Residual + diagnostics of :func:`fractional_diffusion_residual`."""

    residual: Array
    diag: dict[str, Array]


def fractional_diffusion_residual(
    state: FieldState,
    *,
    alphas: Sequence[float | Array],
    order: int,
    component: str = "u",
    kind: str = "caputo",
    a: float = 0.0,
    source: Callable[[FieldState], Array] | None = None,
) -> FractionalDiffusionOutput:
    r"""Space-fractional diffusion residual (jax twin).

    See :func:`omnibias.fractional.torch.field.fractional_diffusion_residual`.
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
    frac: Array | None = None
    for ax, al in zip(spatial, alphas, strict=True):
        term = field_fractional_partial(
            state, component, axis=ax, alpha=al, order=order, a=a, kind=kind
        )
        frac = term if frac is None else frac + term
    assert frac is not None, "at least one spatial axis is required"

    residual = u_t - frac
    if source is not None:
        residual = residual - source(state)
    diag = {"mean_sq_residual": jnp.mean(residual * residual)}
    return FractionalDiffusionOutput(residual=residual, diag=diag)


__all__ = [
    "FractionalDiffusionOutput",
    "field_fractional_partial",
    "fractional_diffusion_residual",
]
