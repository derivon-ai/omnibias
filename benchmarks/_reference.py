# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent high-precision references; never use a compared kernel as truth."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import mpmath as mp
import numpy as np
from numpy.typing import NDArray


def derivative_reference(
    function: Callable[[Any], Any], points: NDArray[np.float64], order: int
) -> NDArray[np.float64]:
    """Differentiate an mpmath expression at 80 decimal digits, then round to float64."""
    if order < 0:
        raise ValueError("order must be non-negative")
    with mp.workdps(80):
        return np.asarray(
            [float(mp.diff(function, mp.mpf(float(x)), order)) for x in points],
            dtype=np.float64,
        )


def tanh_network_reference(
    layers: Sequence[tuple[NDArray[np.float64], NDArray[np.float64]]],
    points: NDArray[np.float64],
    order: int,
) -> NDArray[np.float64]:
    """Reference for a scalar-input/output tanh MLP with an affine final layer."""
    with mp.workdps(80):
        weights = [mp.matrix(w.tolist()) for w, _ in layers]
        biases = [mp.matrix(b.tolist()) for _, b in layers]

        def forward(x: Any) -> Any:
            value = mp.matrix([x])
            for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
                value = weight * value + bias
                if index != len(weights) - 1:
                    value = value.apply(mp.tanh)
            return value[0]

        return derivative_reference(forward, points, order)


def error_metrics(
    values: NDArray[np.float64], reference: NDArray[np.float64]
) -> dict[str, float]:
    """Absolute error plus relative error with an explicit near-zero floor."""
    absolute = np.abs(values - reference)
    return {
        "max_abs": float(np.max(absolute)),
        "median_scaled_error": float(np.median(absolute / np.maximum(np.abs(reference), 1e-3))),
    }
