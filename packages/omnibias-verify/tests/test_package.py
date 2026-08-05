# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Scaffold smoke tests: the package imports, versions, and the IBP baseline is sound."""

from __future__ import annotations

import omnibias.verify as verify
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    Network,
    ReLULayer,
    affine_layer,
    interval_propagate,
    relu_enclosure,
)


def test_version_and_all() -> None:
    assert verify.__version__ == "0.1.0a1"
    for name in verify.__all__:
        assert hasattr(verify, name)


def test_relu_enclosure_is_sound() -> None:
    iv = Interval(-2.0, 3.0)
    enc = relu_enclosure(iv)
    for x in (-2.0, -0.5, 0.0, 1.5, 3.0):
        assert enc.lo <= max(0.0, x) <= enc.hi


def test_interval_propagate_contains_true_outputs() -> None:
    # 2 -> 2 affine, ReLU, 2 -> 1 affine.
    net = Network(
        [
            affine_layer([[1.0, -1.0], [0.5, 0.5]], [0.0, -0.25]),
            ReLULayer(),
            affine_layer([[1.0, 2.0]], [0.1]),
        ]
    )
    box = [Interval(-1.0, 1.0), Interval(-1.0, 1.0)]
    result = interval_propagate(net, box)

    def forward(x: list[float]) -> list[float]:
        h = [x[0] - x[1], 0.5 * x[0] + 0.5 * x[1] - 0.25]
        h = [max(0.0, v) for v in h]
        return [h[0] + 2.0 * h[1] + 0.1]

    import itertools

    for pt in itertools.product([-1.0, -0.3, 0.0, 0.6, 1.0], repeat=2):
        (y,) = forward(list(pt))
        assert result.output[0].lo <= y <= result.output[0].hi
