# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Interval bound propagation (IBP): the sound baseline enclosure.

This is the simplest rigorous forward pass -- a box in, a box out -- and the
correctness floor every tighter method (Taylor models, branch-and-bound, added in
the propagation-engine module) must stay inside.  Every operation uses the
outward-rounded :class:`~omnibias.core.verified.interval.Interval`, so the output
box provably contains ``net(x)`` for every ``x`` in the input box.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval, IntervalLike, sum_intervals
from omnibias.core.verified.transcend import sigmoid_iv, tanh_iv
from omnibias.verify._core.activations import gelu_enclosure, max_enclosure
from omnibias.verify._core.network import (
    GELULayer,
    LinearLayer,
    MaxPoolLayer,
    Network,
    ReLULayer,
    SigmoidLayer,
    TanhLayer,
)


@dataclass(frozen=True)
class BoundResult:
    """An output enclosure plus the per-layer boxes (handy for diagnostics)."""

    output: tuple[Interval, ...]
    layer_boxes: tuple[tuple[Interval, ...], ...]

    @property
    def lower(self) -> tuple[float, ...]:
        return tuple(iv.lo for iv in self.output)

    @property
    def upper(self) -> tuple[float, ...]:
        return tuple(iv.hi for iv in self.output)


def relu_enclosure(iv: Interval) -> Interval:
    """Sound enclosure of ``max(0, x)`` over ``iv`` (exact: ReLU is monotone)."""
    return Interval(max(0.0, iv.lo), max(0.0, iv.hi))


def _linear(layer: LinearLayer, x: Sequence[Interval]) -> tuple[Interval, ...]:
    if len(x) != layer.in_features:
        raise ValueError(
            f"layer expects {layer.in_features} inputs, received {len(x)}"
        )
    out: list[Interval] = []
    for row, b in zip(layer.weight, layer.bias, strict=True):
        acc = sum_intervals([Interval.point(w) * xi for w, xi in zip(row, x, strict=True)])
        out.append(acc + Interval.point(b))
    return tuple(out)


def interval_propagate(
    net: Network, input_box: Sequence[IntervalLike]
) -> BoundResult:
    """Propagate ``input_box`` through ``net`` with interval bound propagation.

    The returned :class:`BoundResult` is sound: each output interval contains the
    corresponding coordinate of ``net(x)`` for every ``x`` in ``input_box``.
    """
    x = tuple(Interval.from_value(v) for v in input_box)
    boxes: list[tuple[Interval, ...]] = [x]
    for layer in net:
        if isinstance(layer, LinearLayer):
            x = _linear(layer, x)
        elif isinstance(layer, ReLULayer):
            x = tuple(relu_enclosure(xi) for xi in x)
        elif isinstance(layer, TanhLayer):
            x = tuple(tanh_iv(xi) for xi in x)
        elif isinstance(layer, SigmoidLayer):
            x = tuple(sigmoid_iv(xi) for xi in x)
        elif isinstance(layer, GELULayer):
            x = tuple(gelu_enclosure(xi) for xi in x)
        elif isinstance(layer, MaxPoolLayer):
            x = tuple(max_enclosure([x[i] for i in group]) for group in layer.groups)
        else:  # pragma: no cover - exhaustive over the Layer union
            raise TypeError(f"unsupported layer type: {type(layer).__name__}")
        boxes.append(x)
    return BoundResult(output=x, layer_boxes=tuple(boxes))


__all__ = [
    "BoundResult",
    "interval_propagate",
    "relu_enclosure",
]
