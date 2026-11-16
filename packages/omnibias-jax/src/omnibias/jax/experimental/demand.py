# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""One interface for pointwise PDE residuals, with the derivative quantities they read declared.

The residual-Gram programs historically took two incompatible callbacks: ``residual(u, grad, lap, x)`` for the
full-gradient engine and ``residual(u, grad_sq, lap, directional, x)`` (or with a bi-Laplacian) for the
first-layer-projected engine. Which engine may serve a residual is a fact about what the residual reads, and a
callback signature hides it. :class:`Residual` states it: a function of one :class:`Quantities` value plus the
set of quantities it uses. Quantities that were not declared reach the function as ``None``, so a residual that
reads one it did not declare fails loudly instead of being served by an engine that cannot provide it.

* ``u``: the field value ``(B, C)``.
* ``grad``: the full gradient ``(B, D, C)``, available from the full-gradient engine only.
* ``grad_sq``: ``|S^T grad u|^2`` on the selected inputs ``(B, C)``.
* ``lap``: ``tr(S S^T hess u)`` ``(B, C)``.
* ``bilap``: ``lap`` applied twice ``(B, C)``, available from the projected engine only.
* ``directional``: ``V^T grad u`` along fixed drift directions ``(B, q, C)`` (or per-point ``(B, D, q)`` directions),
  projected and stream engines.
* ``hess_form``: ``tr(A hess u)`` for a symmetric ``A`` of any sign ``(B, C)``, stream engine only.
* ``mixed``, ``third``, ``fourth``: the partials ``d^alpha u`` named by the multi-indices of a ``QuantitySpec``
  ``(B, R, C)``, stream engine only.
* ``lap_grad``: ``grad(lap u)`` ``(B, D, C)``, stream engine only (cost grows with ``D^2``: the low-dimensional form).

The stream engine (:func:`streams_quantities`) is the engine for these: it evaluates residuals as plain differentiable
tensor expressions, so residual rows, parameter gradients and every second-order optimizer reach them without a
dedicated Gram program.

``legacy_full_gradient`` and ``legacy_projected`` wrap old-style callbacks; the Gram programs also keep accepting
those callbacks directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from omnibias.core.demand import (
    Quantities,
    QuantitySpec,
    Residual,
    block_order,
    legacy_full_gradient,
    legacy_projected,
    partial_requests,
)
from omnibias.jax.experimental._chain import points_2d
from omnibias.jax.experimental.streams import partial_derivatives, second_order_streams

import jax.numpy as jnp
from jax import Array

__all__ = [
    "Quantities",
    "QuantitySpec",
    "Residual",
    "block_order",
    "legacy_full_gradient",
    "legacy_projected",
    "lower_full",
    "lower_projected",
    "streams_quantities",
    "streams_residual",
]

_FULL_ENGINE = frozenset({"value", "grad", "grad_sq", "lap"})
_PROJECTED_ENGINE = frozenset({"value", "grad_sq", "lap", "bilap", "directional"})


def lower_full(residual: Residual, order: int, weights: Array) -> Callable[..., Array]:
    """The ``(u, grad, lap, x)`` callback of the full-gradient engine; ``weights`` is the metric ``s^2`` per input."""
    extra = residual.demand.quantities - _FULL_ENGINE
    if extra:
        raise ValueError(
            f"the full-gradient engine cannot provide {sorted(extra)}; use the first-layer projected engine"
        )
    want_sq = "grad_sq" in residual.demand

    def callback(u: Array, grad: Array | None, lap: Array | None, x: Array) -> Array:
        grad_sq = None
        if want_sq and grad is not None:
            grad_sq = jnp.einsum("bkc,k->bc", grad * grad, weights)
        out: Array = residual(Quantities(u=u, x=x, grad=grad, grad_sq=grad_sq, lap=lap))
        return out

    return callback


def lower_projected(residual: Residual, order: int) -> Callable[..., Array]:
    """The callback of the projected engine: ``(u, gs, lap, dv, x)``, or with ``bilap`` for order 4."""
    extra = residual.demand.quantities - _PROJECTED_ENGINE
    if extra:
        raise ValueError(
            f"the projected engine cannot provide {sorted(extra)}: the full gradient is not formed "
            "(use the full-gradient engine, or read grad_sq)"
        )
    if order == 4:

        def callback4(
            u: Array, gs: Array, lap: Array, bil: Array, dv: Array | None, x: Array
        ) -> Array:
            out: Array = residual(
                Quantities(u=u, x=x, grad_sq=gs, lap=lap, bilap=bil, directional=dv)
            )
            return out

        return callback4

    def callback(
        u: Array, gs: Array | None, lap: Array | None, dv: Array | None, x: Array
    ) -> Array:
        out: Array = residual(Quantities(u=u, x=x, grad_sq=gs, lap=lap, directional=dv))
        return out

    return callback


def streams_quantities(
    x: Array,
    layers: Sequence[Any],
    demand: Residual | Any,
    spec: QuantitySpec | None = None,
    *,
    diffusion: Array | None = None,
    point_diffusion: Array | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Array | None = None,
    weights: Array | None = None,
    form: str = "auto",
    allow_almost_everywhere: bool = False,
) -> Quantities:
    """Every quantity of ``demand`` at ``x`` from the stacked stream engine, as differentiable tensor expressions.

    ``demand`` is a :class:`Residual` or a :class:`~omnibias.core.demand.Demand`. The second-order group (``grad_sq``,
    ``lap``, ``directional``, and ``grad`` through the full Jacobian) is one stacked pass; ``hess_form`` is a second pass
    carrying ``A``; the partials (``mixed``, ``third``, ``fourth``, ``lap_grad``, ``bilap``) share one pass of directional
    Taylor streams, so a direction used by several quantities is carried once. ``diffusion``, ``point_diffusion``,
    ``coordinates``, ``weights`` and ``form`` select the metric of ``grad_sq`` and ``lap`` exactly as in
    :func:`~omnibias.jax.experimental.streams.second_order_streams`; ``directions`` is ``(D, q)`` or per-point
    ``(B, D, q)``. ``bilap`` is the plain Laplacian squared over ``coordinates`` (no metric). Unbatched ``x`` ``(D,)``
    returns unbatched quantities.
    """
    names = demand.demand if isinstance(demand, Residual) else demand
    spec = QuantitySpec() if spec is None else spec
    a, _ = points_2d(x)
    dim = a.shape[1]
    spec.check(names, dim)
    metric: dict[str, Any] = {
        "diffusion": diffusion,
        "point_diffusion": point_diffusion,
        "coordinates": coordinates,
        "weights": weights,
    }
    plain = all(v is None for v in metric.values())
    if "bilap" in names and (diffusion is not None or point_diffusion is not None):
        raise ValueError(
            "bilap through the stream engine is the plain Laplacian squared over `coordinates`"
        )
    value: Array | None = None
    grad = grad_sq = lap = directional = None
    flags = {k: k in names for k in ("grad", "grad_sq", "lap", "directional")}

    if flags["grad"]:
        full = second_order_streams(
            x,
            layers,
            directions=directions if plain else None,
            full_jacobian=True,
            allow_almost_everywhere=allow_almost_everywhere,
        )
        value, grad = full.value, full.jacobian
        if plain:
            grad_sq, lap, directional = full.grad_sq, full.laplacian, full.directional
    if (flags["grad_sq"] or flags["lap"] or flags["directional"]) and (
        grad_sq is None or not plain
    ):
        res = second_order_streams(
            x,
            layers,
            directions=directions,
            form=form,
            allow_almost_everywhere=allow_almost_everywhere,
            **metric,
        )
        value = res.value if value is None else value
        grad_sq, lap, directional = res.grad_sq, res.laplacian, res.directional
    hess = None
    if "hess_form" in names:
        res = second_order_streams(
            x,
            layers,
            coordinates=coordinates,
            hess_form=jnp.asarray(spec.hess_form),
            form=form,
            allow_almost_everywhere=allow_almost_everywhere,
        )
        value = res.value if value is None else value
        hess = res.laplacian
    coords = None if coordinates is None else [int(c) for c in np.asarray(coordinates)]
    alphas, where, bilap_w = partial_requests(names, spec, dim, coords)
    parts: dict[str, Array] = {}
    if alphas:
        v, d = partial_derivatives(
            x, layers, alphas, allow_almost_everywhere=allow_almost_everywhere
        )
        value = v if value is None else value
        for name, (lo, hi) in where.items():
            parts[name] = d[..., lo:hi, :]
    if value is None:
        value = second_order_streams(
            x, layers, allow_almost_everywhere=allow_almost_everywhere
        ).value
    batch = value.shape[:-1]
    out: dict[str, Any] = {}
    if "lap_grad" in parts:
        p = parts["lap_grad"]
        out["lap_grad"] = p.reshape(*batch, dim, dim, p.shape[-1]).sum(axis=-2)
    if "bilap" in parts:
        out["bilap"] = jnp.einsum(
            "r,...rc->...c", jnp.asarray(bilap_w, value.dtype), parts["bilap"]
        )
    for name in ("mixed", "third", "fourth"):
        if name in parts:
            out[name] = parts[name]
    return Quantities(
        u=value,
        x=x,
        grad=grad,
        grad_sq=grad_sq if flags["grad_sq"] else None,
        lap=lap if flags["lap"] else None,
        directional=directional if flags["directional"] else None,
        hess_form=hess,
        **out,
    )


def streams_residual(
    x: Array,
    layers: Sequence[Any],
    residual: Residual,
    spec: QuantitySpec | None = None,
    **options: Any,
) -> Array:
    """``residual(streams_quantities(...))``: the residual rows ``(B,)`` or ``(B, R)`` of a field, differentiable in the layers."""
    block_order(residual, None, "streams")
    return jnp.asarray(residual(streams_quantities(x, layers, residual, spec, **options)))
