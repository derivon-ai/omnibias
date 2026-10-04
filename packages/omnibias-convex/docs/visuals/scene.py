# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
from omnibias.convex import BarrierOptions
from omnibias.convex.torch import solve_qp
from visual_story import bars, frame, line, story


def build_scene():
    x = np.linspace(-1, 1, 81)
    frames = []
    for weight in np.geomspace(1.0, 1e5, 20):
        target = 1.4
        result = solve_qp(
            [[1.0]],
            [-float(target)],
            [[1.0], [-1.0]],
            [1.0, 1.0],
            x0=[0.0],
            options=BarrierOptions(t0=float(weight), max_outer=1),
        )
        optimum = float(result.x[0])
        objective = 0.5 * x * x - target * x
        frames.append(
            frame(
                line(
                    "Quadratic objective on the feasible interval",
                    x,
                    ("objective", objective),
                    points=[[optimum, 0.5 * optimum**2 - target * optimum]],
                    ylim=[-1.2, 2],
                    xlabel="decision x",
                ),
                bars("Constraint slack", result.slack.detach(), ["x ≤ 1", "−x ≤ 1"], ylim=[0, 2.1]),
                f"Barrier weight {weight:.1g} · solution {optimum:.3f} · barrier gap {result.gap:.2g}; finite iterates retain a measured gap.",
            )
        )
    return story(
        "convex",
        "Optimize inside explicit constraints.",
        "A barrier path approaches the constrained quadratic optimum.",
        "omnibias.convex.torch.solve_qp",
        frames,
    )
