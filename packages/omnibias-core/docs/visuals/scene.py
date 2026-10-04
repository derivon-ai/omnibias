# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
from omnibias.core import eval_tanh_derivative, tanh_polynomial_coeffs
from visual_story import bars, frame, line, story


def build_scene():
    x = np.linspace(-2, 2, 101)
    frames = []
    for k in range(5):
        y = [eval_tanh_derivative(float(z), k) for z in x]
        coeff = tanh_polynomial_coeffs(k)
        for _ in range(4):
            frames.append(
                frame(
                    line(f"Derivative order {k}", x, (f"σ^({k})(x)", y), xlabel="activation input"),
                    bars("Shared polynomial coefficients", coeff, xlabel="power of tanh(x)"),
                    f"Order {k}: one base activation, followed by a shared polynomial; arithmetic still grows with order.",
                )
            )
    return story(
        "core",
        "One activation. A derivative tower.",
        "Exact coefficient algebra beneath every backend.",
        "omnibias.core.eval_tanh_derivative / tanh_polynomial_coeffs",
        frames,
    )
