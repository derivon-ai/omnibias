# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Property certificates: robustness, Lipschitz, monotonicity, reachable set."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence

from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    LinearLayer,
    Network,
    SigmoidLayer,
    TanhLayer,
    affine_layer,
    certify_robustness,
    interval_jacobian,
    lipschitz_bound,
    monotonicity,
    reachable_box,
)


def _forward(net: Network, x: Sequence[float]) -> list[float]:
    vec = list(x)
    for layer in net:
        if isinstance(layer, LinearLayer):
            vec = [
                sum(w * xi for w, xi in zip(row, vec, strict=True)) + b
                for row, b in zip(layer.weight, layer.bias, strict=True)
            ]
        elif isinstance(layer, TanhLayer):
            vec = [math.tanh(v) for v in vec]
        elif isinstance(layer, SigmoidLayer):
            vec = [1.0 / (1.0 + math.exp(-v)) for v in vec]
    return vec


def _grid(box: Sequence[Interval], n: int) -> itertools.product:  # type: ignore[type-arg]
    axes = [[iv.lo + (iv.hi - iv.lo) * k / (n - 1) for k in range(n)] for iv in box]
    return itertools.product(*axes)


def _classifier() -> Network:
    return Network(
        [
            affine_layer([[1.0, 0.2], [-0.3, 0.8]], [0.1, -0.1]),
            TanhLayer(),
            affine_layer([[1.5, -0.5], [-0.4, 1.2]], [0.3, -0.2]),
        ]
    )


def test_robustness_certified_is_sound() -> None:
    net = _classifier()
    x0 = [0.6, -0.4]
    # true label at x0:
    out0 = _forward(net, x0)
    true_label = max(range(len(out0)), key=lambda i: out0[i])
    cert = certify_robustness(net, x0, eps=0.1, true_label=true_label, order=2)
    assert cert.certified
    # Soundness: on a dense grid in the ball, true_label is really the argmax.
    box = [Interval(xi - 0.1, xi + 0.1) for xi in x0]
    for pt in _grid(box, 9):
        out = _forward(net, list(pt))
        assert max(range(len(out)), key=lambda i: out[i]) == true_label
        for j in range(len(out)):
            if j != true_label:
                assert out[true_label] - out[j] > 0.0


def test_robustness_large_eps_not_certified() -> None:
    net = _classifier()
    x0 = [0.05, 0.0]  # near the decision boundary
    out0 = _forward(net, x0)
    true_label = max(range(len(out0)), key=lambda i: out0[i])
    cert = certify_robustness(net, x0, eps=1.5, true_label=true_label, order=2)
    assert not cert.certified  # honest: a wide ball crosses the boundary


def test_lipschitz_bounds_empirical_inf() -> None:
    net = _classifier()
    box = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    bound = lipschitz_bound(net, box, norm="inf")
    rng = random.Random(0)
    worst = 0.0
    for _ in range(2000):
        a = [rng.uniform(-1.0, 1.0) for _ in range(2)]
        b = [rng.uniform(-1.0, 1.0) for _ in range(2)]
        if a == b:
            continue
        fa, fb = _forward(net, a), _forward(net, b)
        num = max(abs(x - y) for x, y in zip(fa, fb, strict=True))
        den = max(abs(x - y) for x, y in zip(a, b, strict=True))
        worst = max(worst, num / den)
    assert worst <= bound + 1e-9


def test_lipschitz_bounds_empirical_l2() -> None:
    net = _classifier()
    box = [Interval(-0.8, 0.8), Interval(-0.8, 0.8)]
    bound = lipschitz_bound(net, box, norm="l2")
    rng = random.Random(1)
    worst = 0.0
    for _ in range(2000):
        a = [rng.uniform(-0.8, 0.8) for _ in range(2)]
        b = [rng.uniform(-0.8, 0.8) for _ in range(2)]
        num = math.dist(_forward(net, a), _forward(net, b))
        den = math.dist(a, b)
        if den > 1e-9:
            worst = max(worst, num / den)
    assert worst <= bound + 1e-9


def test_monotonicity_increasing() -> None:
    # out = tanh(2 x0 + 0.5 x1) read out with a positive weight -> increasing in x0.
    net = Network(
        [
            affine_layer([[2.0, 0.5]], [0.0]),
            TanhLayer(),
            affine_layer([[3.0]], [0.0]),
        ]
    )
    box = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    cert = monotonicity(net, box, out_index=0, in_index=0)
    assert cert.verdict == "increasing"
    assert cert.derivative.lo >= 0.0
    # Confirm: fixing x1, the output is nondecreasing in x0 on a grid.
    for x1 in (-1.0, -0.3, 0.4, 1.0):
        prev = -math.inf
        for k in range(21):
            x0 = -1.0 + 2.0 * k / 20.0
            val = _forward(net, [x0, x1])[0]
            assert val >= prev - 1e-9
            prev = val


def test_monotonicity_decreasing() -> None:
    net = Network([affine_layer([[-1.5]], [0.0]), SigmoidLayer()])
    box = [Interval(-2.0, 2.0)]
    cert = monotonicity(net, box, out_index=0, in_index=0)
    assert cert.verdict == "decreasing"
    assert cert.derivative.hi <= 0.0


def test_monotonicity_unknown_when_sign_varies() -> None:
    # A symmetric quadratic-like response: derivative changes sign over the box.
    net = Network(
        [
            affine_layer([[1.0], [-1.0]], [0.0, 0.0]),
            TanhLayer(),
            affine_layer([[1.0, 1.0]], [0.0]),
        ]
    )
    box = [Interval(-1.0, 1.0)]
    cert = monotonicity(net, box, out_index=0, in_index=0)
    assert cert.verdict == "unknown"


def test_reachable_box_is_sound() -> None:
    net = _classifier()
    box = [Interval(-0.7, 0.7), Interval(-0.7, 0.7)]
    reach = reachable_box(net, box, order=2, max_boxes=64)
    for pt in _grid(box, 11):
        out = _forward(net, list(pt))
        for val, iv in zip(out, reach, strict=True):
            assert iv.lo - 1e-9 <= val <= iv.hi + 1e-9


def test_interval_jacobian_contains_finite_differences() -> None:
    net = _classifier()
    box = [Interval(-0.5, 0.5), Interval(-0.5, 0.5)]
    jac = interval_jacobian(net, box)
    rng = random.Random(2)
    h = 1e-6
    for _ in range(200):
        x = [rng.uniform(-0.5, 0.5) for _ in range(2)]
        for j in range(2):
            xp = list(x)
            xp[j] += h
            xm = list(x)
            xm[j] -= h
            fp, fm = _forward(net, xp), _forward(net, xm)
            for i in range(2):
                fd = (fp[i] - fm[i]) / (2.0 * h)
                assert jac[i][j].lo - 1e-4 <= fd <= jac[i][j].hi + 1e-4
