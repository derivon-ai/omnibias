# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Taylor-model propagation: soundness vs a dense grid, and tighter than IBP."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence

from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    GELULayer,
    LinearLayer,
    MaxPoolLayer,
    Network,
    ReLULayer,
    SigmoidLayer,
    TanhLayer,
    affine_layer,
    gauss_cdf_iv,
    interval_propagate,
    taylor_output_bounds,
    taylor_propagate,
)

_SQRT2 = math.sqrt(2.0)


def _relu(v: float) -> float:
    return max(0.0, v)


def _gelu(v: float) -> float:
    return v * 0.5 * (1.0 + math.erf(v / _SQRT2))


_ACT: dict[type, Callable[[float], float]] = {
    TanhLayer: math.tanh,
    SigmoidLayer: lambda v: 1.0 / (1.0 + math.exp(-v)),
    ReLULayer: _relu,
    GELULayer: _gelu,
}


def _forward(net: Network, x: Sequence[float]) -> list[float]:
    vec = list(x)
    for layer in net:
        if isinstance(layer, LinearLayer):
            vec = [
                sum(w * xi for w, xi in zip(row, vec, strict=True)) + b
                for row, b in zip(layer.weight, layer.bias, strict=True)
            ]
        elif isinstance(layer, MaxPoolLayer):
            vec = [max(vec[i] for i in group) for group in layer.groups]
        else:
            fn = _ACT[type(layer)]
            vec = [fn(v) for v in vec]
    return vec


def _grid(box: Sequence[Interval], n: int) -> itertools.product:  # type: ignore[type-arg]
    axes = [[iv.lo + (iv.hi - iv.lo) * k / (n - 1) for k in range(n)] for iv in box]
    return itertools.product(*axes)


def _assert_sound(net: Network, box: list[Interval], order: int = 3, n: int = 7) -> None:
    bounds = taylor_output_bounds(net, box, order=order)
    for pt in _grid(box, n):
        y = _forward(net, list(pt))
        for yi, bi in zip(y, bounds, strict=True):
            assert bi.lo - 1e-12 <= yi <= bi.hi + 1e-12


def _grid_and_random(
    box: Sequence[Interval], seed: int, *, grid: int = 7, rnd: int = 40
) -> list[tuple[float, ...]]:
    pts = [tuple(pt) for pt in _grid(box, grid)]
    rng = random.Random(seed)
    pts.extend(tuple(rng.uniform(iv.lo, iv.hi) for iv in box) for _ in range(rnd))
    return pts


def test_taylor_sound_vs_grid_and_random() -> None:
    """Founding delta->0 soundness rule: the Taylor enclosure contains the true
    forward value at a dense deterministic grid AND a random sample in the box."""
    box = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    nets = {
        "tanh": Network(
            [affine_layer([[1.2, 0.7], [-0.9, 1.1]], [0.1, -0.2]), TanhLayer(), affine_layer([[1.0, -1.5]], [0.05])]
        ),
        "sigmoid": Network(
            [affine_layer([[0.8, -0.4], [0.5, 0.9]], [0.0, 0.1]), SigmoidLayer(), affine_layer([[1.3, -0.7]], [-0.05])]
        ),
        "gelu": Network(
            [affine_layer([[0.6, 0.3], [-0.7, 0.4]], [0.05, -0.1]), GELULayer(), affine_layer([[0.9, -1.1]], [0.02])]
        ),
    }
    for seed, net in enumerate(nets.values()):
        bounds = taylor_output_bounds(net, box, order=4)
        for pt in _grid_and_random(box, seed):
            y = _forward(net, list(pt))
            for yi, bi in zip(y, bounds, strict=True):
                assert bi.lo - 1e-12 <= yi <= bi.hi + 1e-12


def test_tanh_mlp_sound_vs_grid() -> None:
    net = Network(
        [
            affine_layer([[0.8, -0.5], [0.3, 0.9], [-0.6, 0.2]], [0.1, -0.2, 0.05]),
            TanhLayer(),
            affine_layer([[0.5, -0.4, 0.7], [0.2, 0.6, -0.3]], [0.0, 0.1]),
        ]
    )
    _assert_sound(net, [Interval(-0.5, 0.5), Interval(-0.5, 0.5)])


def test_sigmoid_mlp_sound_vs_grid() -> None:
    net = Network(
        [
            affine_layer([[1.0, 0.5], [-0.7, 0.8]], [0.0, 0.2]),
            SigmoidLayer(),
            affine_layer([[1.0, -1.0]], [0.3]),
        ]
    )
    _assert_sound(net, [Interval(-1.0, 1.0), Interval(-0.5, 0.5)])


def test_gelu_mlp_sound_vs_grid() -> None:
    net = Network(
        [
            affine_layer([[0.9, -0.4], [0.2, 0.7]], [0.1, -0.1]),
            GELULayer(),
            affine_layer([[0.6, 0.5]], [0.0]),
        ]
    )
    _assert_sound(net, [Interval(-0.6, 0.6), Interval(-0.4, 0.4)])


def test_relu_mlp_sound_vs_grid() -> None:
    net = Network(
        [
            affine_layer([[1.0, -1.0], [0.5, 0.5]], [0.2, -0.1]),
            ReLULayer(),
            affine_layer([[1.0, 2.0]], [0.05]),
        ]
    )
    _assert_sound(net, [Interval(-1.0, 1.0), Interval(-1.0, 1.0)])


def test_maxpool_sound_vs_grid() -> None:
    net = Network(
        [
            affine_layer([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], [0.0, 0.0, 0.1]),
            MaxPoolLayer(((0, 1, 2),)),
            affine_layer([[2.0]], [0.0]),
        ]
    )
    _assert_sound(net, [Interval(-1.0, 1.0), Interval(-1.0, 1.0)])


def test_relu_stable_neuron_is_exact() -> None:
    # Input forced strictly positive after the affine map -> ReLU is identity, so
    # the Taylor model is exact (no relaxation band).
    net = Network([affine_layer([[1.0]], [5.0]), ReLULayer()])
    box = [Interval(-1.0, 1.0)]
    (b,) = taylor_output_bounds(net, box, order=2)
    assert b.lo >= 4.0 - 1e-12 and b.hi <= 6.0 + 1e-12  # exactly [4, 6]
    assert abs(b.width - 2.0) < 1e-9


def test_taylor_tighter_than_ibp() -> None:
    net = Network(
        [
            affine_layer([[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0]),
            TanhLayer(),
            affine_layer([[1.0, 1.0]], [0.0]),
        ]
    )
    box = [Interval(-0.4, 0.4), Interval(-0.4, 0.4)]
    (tm,) = taylor_output_bounds(net, box, order=3)
    ibp = interval_propagate(net, box).output[0]
    assert tm.width < ibp.width
    # both must still be sound
    for pt in _grid(box, 9):
        (y,) = _forward(net, list(pt))
        assert tm.lo - 1e-12 <= y <= tm.hi + 1e-12


def test_higher_order_is_tighter_for_smooth() -> None:
    # Measured on the raw Taylor-model bound (a single monotone activation is
    # always beaten by IBP, so the order benefit only shows pre-intersection).
    # Inside the radius of convergence every higher order is at least as tight as
    # order 1, and order 2 is strictly tighter.
    net = Network([affine_layer([[1.0]], [0.0]), TanhLayer()])
    box = [Interval(-0.3, 0.3)]
    widths = [taylor_propagate(net, box, order=o)[0].bound().width for o in (1, 2, 3, 5)]
    assert all(w <= widths[0] + 1e-15 for w in widths[1:])
    assert widths[1] < widths[0]  # order 2 genuinely tighter than order 1


def test_gauss_cdf_sound() -> None:
    for lo, hi in [(-3.0, -1.0), (-0.5, 0.5), (1.0, 2.5)]:
        enc = gauss_cdf_iv(Interval(lo, hi))
        for x in (lo, 0.5 * (lo + hi), hi):
            true = 0.5 * (1.0 + math.erf(x / _SQRT2))
            assert enc.lo - 1e-12 <= true <= enc.hi + 1e-12
