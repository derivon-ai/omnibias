# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Taylor-model propagation: a polynomial enclosure of the network over an input box.

Each input coordinate starts as a degree-``order`` multivariate Taylor model
(:class:`~omnibias.core.verified.taylor_model_mv.TaylorModelMV`).  Affine layers
combine the models exactly; smooth activations are composed through the
closed-form derivative tower; nonsmooth ones use the sound relaxations in
:mod:`omnibias.verify._core.activations`.  Because the *polynomial shape* in the
input variables is carried all the way through (not flattened to a box at each
layer), the final :meth:`~TaylorModelMV.bound` is typically far tighter than
interval-bound propagation -- the dependency/wrapping error is largely cancelled.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.taylor_model_mv import TaylorModelMV
from omnibias.verify._core.activations import (
    compose_gelu,
    compose_sigma,
    max_enclosure,
    relu_taylor,
)
from omnibias.verify._core.network import (
    GELULayer,
    LinearLayer,
    MaxPoolLayer,
    Network,
    ReLULayer,
    SigmoidLayer,
    TanhLayer,
)
from omnibias.verify._core.propagate import interval_propagate


def input_models(
    input_box: Sequence[IntervalLike], order: int
) -> list[TaylorModelMV]:
    """Build one coordinate Taylor model per input axis over ``input_box``."""
    ivs = [Interval.from_value(v) for v in input_box]
    center = [iv.mid for iv in ivs]
    radius = [0.5 * iv.width for iv in ivs]
    dim = len(ivs)
    return [TaylorModelMV.coordinate(axis, center, radius, order) for axis in range(dim)]


def _linear(layer: LinearLayer, x: Sequence[TaylorModelMV]) -> list[TaylorModelMV]:
    if len(x) != layer.in_features:
        raise ValueError(f"layer expects {layer.in_features} inputs, received {len(x)}")
    out: list[TaylorModelMV] = []
    for row, b in zip(layer.weight, layer.bias, strict=True):
        acc = x[0] * Interval.point(row[0])
        for w, xi in zip(row[1:], x[1:], strict=True):
            acc = acc + xi * Interval.point(w)
        out.append(acc + Interval.point(b))
    return out


def _maxpool(
    layer: MaxPoolLayer, x: Sequence[TaylorModelMV]
) -> list[TaylorModelMV]:
    """Group ``max`` as a constant Taylor model holding the rigorous range.

    If one element dominates the group (its lower bound exceeds every other upper
    bound) the exact model is kept; otherwise the group collapses to the enclosing
    interval -- sound, and the only honest option for an active ``max``.
    """
    if not x:
        raise ValueError("max-pool received no inputs")
    center, radius, order = x[0].center, x[0].radius, x[0].order
    out: list[TaylorModelMV] = []
    for group in layer.groups:
        members = [x[i] for i in group]
        bounds = [m.bound() for m in members]
        dominant = None
        for i, bi in enumerate(bounds):
            if all(bi.lo >= bounds[j].hi for j in range(len(bounds)) if j != i):
                dominant = members[i]
                break
        if dominant is not None:
            out.append(dominant)
        else:
            rng = max_enclosure(bounds)
            out.append(TaylorModelMV.constant(rng, center, radius, order))
    return out


def taylor_propagate(
    net: Network, input_box: Sequence[IntervalLike], *, order: int = 2
) -> list[TaylorModelMV]:
    r"""Propagate ``input_box`` through ``net`` as degree-``order`` Taylor models.

    Returns one :class:`TaylorModelMV` per output; ``model.bound()`` is a sound
    enclosure of that output over the whole input box.
    """
    if order < 1:
        raise ValueError("Taylor-model propagation needs order >= 1")
    x: list[TaylorModelMV] = input_models(input_box, order)
    for layer in net:
        if isinstance(layer, LinearLayer):
            x = _linear(layer, x)
        elif isinstance(layer, ReLULayer):
            x = [relu_taylor(xi) for xi in x]
        elif isinstance(layer, TanhLayer):
            x = [compose_sigma(xi, "tanh", order) for xi in x]
        elif isinstance(layer, SigmoidLayer):
            x = [compose_sigma(xi, "sigmoid", order) for xi in x]
        elif isinstance(layer, GELULayer):
            x = [compose_gelu(xi, order) for xi in x]
        elif isinstance(layer, MaxPoolLayer):
            x = _maxpool(layer, x)
        else:  # pragma: no cover - exhaustive over the Layer union
            raise TypeError(f"unsupported layer type: {type(layer).__name__}")
    return x


def taylor_output_bounds(
    net: Network, input_box: Sequence[IntervalLike], *, order: int = 2
) -> list[Interval]:
    """Sound per-output enclosure of ``net`` over the input box via Taylor models.

    Each Taylor-model bound is intersected with an interval-bound-propagation pass,
    so the result is provably **never looser than IBP** while keeping the (usually
    much tighter) polynomial enclosure where it wins.
    """
    tm_bounds = [model.bound() for model in taylor_propagate(net, input_box, order=order)]
    ibp = interval_propagate(net, input_box).output
    return [
        Interval(max(t.lo, i.lo), min(t.hi, i.hi))
        for t, i in zip(tm_bounds, ibp, strict=True)
    ]


def linear_image(coeffs: Sequence[float], models: Sequence[TaylorModelMV]) -> TaylorModelMV:
    """The Taylor model of ``sum_i coeffs[i] * models[i]`` (a linear read-out / margin)."""
    if len(coeffs) != len(models):
        raise ValueError("coeffs and models must have equal length")
    acc = models[0] * Interval.point(coeffs[0])
    for c, m in zip(coeffs[1:], models[1:], strict=True):
        acc = acc + m * Interval.point(c)
    return acc


__all__ = [
    "input_models",
    "linear_image",
    "taylor_output_bounds",
    "taylor_propagate",
]
