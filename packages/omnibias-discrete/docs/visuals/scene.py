# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
from omnibias.discrete import AnnealSchedule, round_relaxed
from visual_story import bars, frame, story


def build_scene():
    scores = np.array([-0.8, 0.25, -0.15, 0.7])
    frames = []
    betas = AnnealSchedule(beta0=0.4, beta_growth=1.18, stages=20).betas()
    for beta in betas:
        p = 1 / (1 + np.exp(-beta * scores))
        z = round_relaxed(p)
        energy = float(scores @ z)
        bound = float(np.minimum(scores, 0).sum())
        frames.append(
            frame(
                bars(
                    "Relaxation and decoded assignment",
                    np.concatenate([p, z]),
                    ["p₀", "p₁", "p₂", "p₃", "z₀", "z₁", "z₂", "z₃"],
                    ylim=[0, 1.1],
                ),
                bars("Toy linear-energy bounds", [bound, energy], ["lower bound", "candidate"]),
                f"β = {beta:.2f}; rounding is not optimality: this candidate's gap is {energy - bound:.2f}.",
            )
        )
    return story(
        "discrete",
        "A decision needs more than rounding.",
        "Represent a relaxation, decode a candidate, then measure its gap.",
        "omnibias.discrete.AnnealSchedule / round_relaxed; exact toy linear bound",
        frames,
    )
