# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Branch-and-bound: sound, and tighter than a single Taylor-model pass."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence

from _enclosure import assert_encloses, assert_lower_bound, assert_upper_bound
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    LinearLayer,
    Network,
    TanhLayer,
    affine_layer,
    output_range,
    scalar_readout_range,
    taylor_output_bounds,
)


def _forward(net: Network, x: Sequence[float]) -> list[float]:
    vec = list(x)
    for layer in net:
        if isinstance(layer, LinearLayer):
            vec = [
                sum(w * xi for w, xi in zip(row, vec, strict=True)) + b
                for row, b in zip(layer.weight, layer.bias, strict=True)
            ]
        else:
            vec = [math.tanh(v) for v in vec]
    return vec


def _net() -> Network:
    return Network(
        [
            affine_layer([[1.2, 0.7], [-0.9, 1.1]], [0.1, -0.2]),
            TanhLayer(),
            affine_layer([[1.0, -1.5]], [0.05]),
        ]
    )


def _grid(box: Sequence[Interval], n: int) -> itertools.product:  # type: ignore[type-arg]
    axes = [[iv.lo + (iv.hi - iv.lo) * k / (n - 1) for k in range(n)] for iv in box]
    return itertools.product(*axes)


def test_output_range_is_sound() -> None:
    net = _net()
    box = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    res = output_range(net, box, 0, order=2, max_boxes=64)
    for pt in _grid(box, 11):
        (y,) = _forward(net, list(pt))
        assert_encloses(res.enclosure, y, what=f"net{tuple(pt)}")


def test_output_range_sound_vs_grid_and_random() -> None:
    """Founding delta->0 soundness rule: the branch-and-bound enclosure (both
    output_range and the scalar readout) contains the forward value at a dense
    grid AND a random sample inside the box."""
    net = _net()
    box = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    res = output_range(net, box, 0, order=2, max_boxes=128)
    # The net has a single output, so the identity readout equals output 0.
    readout = scalar_readout_range(net, box, [1.0], order=2, max_boxes=128)
    rng = random.Random(3)
    pts = [tuple(pt) for pt in _grid(box, 11)]
    pts.extend((rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(60))
    for pt in pts:
        (y,) = _forward(net, list(pt))
        assert_encloses(res.enclosure, y, what=f"output_range at {pt}")
        assert_encloses(readout.enclosure, y, what=f"scalar_readout at {pt}")


def test_bab_tightens_single_pass() -> None:
    net = _net()
    box = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    single = taylor_output_bounds(net, box, order=2)[0]
    res = output_range(net, box, 0, order=2, max_boxes=200, tol=1e-7)
    assert res.refined
    assert res.enclosure.lo >= single.lo - 1e-12
    assert res.enclosure.hi <= single.hi + 1e-12
    assert res.enclosure.width < single.width  # genuinely tighter


def test_bab_brackets_true_extremes() -> None:
    net = _net()
    box = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    res = output_range(net, box, 0, order=2, max_boxes=400, tol=1e-7)
    samples = [_forward(net, list(pt))[0] for pt in _grid(box, 21)]
    true_lo, true_hi = min(samples), max(samples)
    # The enclosure must contain the sampled extremes exactly -- these are attained
    # values, so any slack here would be slack on the soundness claim itself ...
    assert_lower_bound(res.enclosure.lo, true_lo, what="sampled minimum")
    assert_upper_bound(res.enclosure.hi, true_hi, what="sampled maximum")
    # ... and be reasonably tight around them after branch-and-bound. Tightness is a
    # quality target, so *this* is where a tolerance legitimately belongs.
    assert true_lo - res.enclosure.lo < 0.1
    assert res.enclosure.hi - true_hi < 0.1


def test_scalar_readout_margin() -> None:
    # A linear read-out (margin between two outputs) over the box.
    net = Network(
        [
            affine_layer([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]),
            TanhLayer(),
        ]
    )
    box = [Interval(-0.5, 0.5), Interval(-0.5, 0.5)]
    res = scalar_readout_range(net, box, [1.0, -1.0], order=3, max_boxes=64)
    for pt in _grid(box, 11):
        y = _forward(net, list(pt))
        margin = y[0] - y[1]
        assert_encloses(res.enclosure, margin, what=f"margin at {tuple(pt)}")
