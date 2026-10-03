# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent high-precision reference for ridge-field differential operators."""

from __future__ import annotations

import mpmath as mp
import numpy as np
from numpy.typing import NDArray


def tanh_field_laplacian_reference(
    points: NDArray[np.float64], weights: NDArray[np.float64],
    biases: NDArray[np.float64], coefficients: NDArray[np.float64], order: int,
) -> NDArray[np.float64]:
    """80-digit reference for Δ^order of a ridge field, using mpmath.diff.

    The chain rule gives sum_h c_h ||w_h||^(2k) tanh^(2k)(w_h.x+b_h).
    No omnibias coefficients or kernels are used; dot products and derivatives
    are evaluated at high precision from the actual float64 inputs.
    """
    if order < 1:
        raise ValueError("Laplacian order must be positive")
    with mp.workdps(80):
        w = [[mp.mpf(float(value)) for value in row] for row in weights]
        b = [mp.mpf(float(value)) for value in biases]
        c = [mp.mpf(float(value)) for value in coefficients]
        norms = [sum(value * value for value in row) ** order for row in w]
        results = []
        for point in points:
            x = [mp.mpf(float(value)) for value in point]
            results.append(float(sum(
                c[h] * norms[h] * mp.diff(mp.tanh,
                    sum(w[h][d] * x[d] for d in range(len(x))) + b[h], 2 * order)
                for h in range(len(w))
            )))
        return np.asarray(results, dtype=np.float64)
