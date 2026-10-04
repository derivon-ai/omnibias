# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
from omnibias.difference import finite_difference_estimate
from omnibias.difference._core.stencil import stencil_offsets, stencil_signs
from visual_story import bars, frame, line, story


def build_scene():
    x = np.linspace(-1.5, 1.5, 81)
    frames = []
    steps = []
    errors = []
    for h in np.geomspace(1, 0.02, 20):
        value = finite_difference_estimate("tanh", 0.3, 1, float(h)).estimate
        nodes = np.array(stencil_offsets(1, float(h))) + 0.3
        signs = stencil_signs(1, float(h))
        steps.append(h)
        errors.append(abs(value - (1 - np.tanh(0.3) ** 2)))
        frames.append(
            frame(
                line(
                    "Samples define a stencil",
                    x,
                    ("tanh(x)", np.tanh(x)),
                    points=np.column_stack([nodes, np.tanh(nodes)]),
                    xlabel="sample coordinate",
                    ylim=[-1, 1],
                ),
                bars(
                    "Weighted sample contributions",
                    np.array(signs) * np.tanh(nodes),
                    ["left sample", "right sample"],
                ),
                f"Spacing {h:.3f} · derivative estimate {value:.5f} · absolute error {errors[-1]:.2g}.",
            )
        )
    return story(
        "difference",
        "Differentiate the samples you have.",
        "Sample locations and weights determine the discrete operator.",
        "omnibias.difference.finite_difference_estimate + stencil weights",
        frames,
    )
