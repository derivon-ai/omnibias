# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
import torch
from omnibias.binary.torch.ops import binarize
from visual_story import frame, line, story


def build_scene():
    x = torch.linspace(-2, 2, 101, dtype=torch.float64, requires_grad=True)
    frames = []
    for beta in np.geomspace(0.5, 6, 20):
        hard = binarize(x, beta=float(beta))
        g = torch.autograd.grad(hard.sum(), x)[0]
        frames.append(
            frame(
                line(
                    "Forward: discrete values",
                    x.detach(),
                    ("quantized output", hard.detach()),
                    ylim=[-1.3, 1.3],
                    xlabel="input",
                ),
                line(
                    "Backward: chosen surrogate",
                    x.detach(),
                    ("surrogate gradient", g.detach()),
                    ylim=[0, 6.5],
                    xlabel="input",
                ),
                f"β = {beta:.2f}: the smooth backward rule is deliberately distinct from the derivative of a hard jump.",
            )
        )
    return story(
        "binary",
        "Hard values. A trainable backward path.",
        "Forward representation and backward optimization have separate contracts.",
        "omnibias.binary.torch.ops.binarize + autograd",
        frames,
    )
