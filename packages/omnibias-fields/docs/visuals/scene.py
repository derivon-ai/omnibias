# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
from omnibias.core import eval_tanh_derivative
from omnibias.fields import SigmaCache
from visual_story import frame, line, story


def build_scene():
    grid = np.linspace(-1, 1, 7)
    xx, yy = np.meshgrid(grid, grid)
    x = np.linspace(-2, 2, 81)
    frames = []
    for offset in np.linspace(-0.6, 0.6, 20):
        slope = np.array(
            [
                SigmaCache(z=float(z)).get_or_compute(
                    1, lambda n, z=float(z): eval_tanh_derivative(z, n)
                )
                for z in (xx + yy + offset).ravel()
            ]
        )
        lap = [
            2
            * SigmaCache(z=float(z)).get_or_compute(
                2, lambda n, z=float(z): eval_tanh_derivative(z, n)
            )
            for z in x + offset
        ]
        frames.append(
            frame(
                {
                    "kind": "vectors",
                    "title": "Gradient of u = tanh(x + y + b)",
                    "x": xx.ravel(),
                    "y": yy.ravel(),
                    "u": slope,
                    "v": slope,
                    "scale": 4,
                },
                line(
                    "Laplacian along y = 0", x, ("∂xx u + ∂yy u", lap), ylim=[-1.7, 1.7], xlabel="x"
                ),
                "A shared SigmaCache supplies activation derivatives to the analytic field provider.",
            )
        )
    return story(
        "fields",
        "One field. Several operators.",
        "Coordinate-aware state shares derivative work across views.",
        "omnibias.fields.SigmaCache with a documented analytic tanh field",
        frames,
    )
