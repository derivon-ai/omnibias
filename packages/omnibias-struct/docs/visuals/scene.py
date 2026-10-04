# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
import torch
from omnibias.struct.torch import soft_viterbi, soft_viterbi_marginals
from visual_story import frame, story


def build_scene():
    e = torch.tensor([[0.3, -0.2], [0.1, 0.5], [0.4, 0.2]], dtype=torch.float64)
    t = torch.tensor([[0.1, -0.1], [-0.2, 0.2]], dtype=torch.float64)
    frames = []
    nodes = [[i, j] for i in range(3) for j in range(2)]
    for beta in np.geomspace(0.5, 12, 20):
        m = soft_viterbi_marginals(e, t, beta=float(beta)).detach().numpy()
        value = float(soft_viterbi(e, t, beta=float(beta)))
        edges = [
            [
                2 * i + j,
                2 * (i + 1) + k,
                int(j == int(m[i].argmax()) and k == int(m[i + 1].argmax())),
            ]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ]
        frames.append(
            frame(
                {
                    "kind": "graph",
                    "title": "Competing state paths",
                    "nodes": nodes,
                    "edges": edges,
                    "labels": ["A", "B"] * 3,
                },
                {
                    "kind": "heatmap",
                    "title": "Soft state marginals",
                    "values": m.T,
                    "vmin": 0,
                    "vmax": 1,
                    "xlabel": "time step",
                    "ylabel": "state",
                },
                f"β = {beta:.2f} · soft path value {value:.3f}; gradients retain information about alternatives.",
            )
        )
    return story(
        "struct",
        "Let paths compete before selecting one.",
        "Dynamic programs expose smooth values and structured marginals.",
        "omnibias.struct.torch.soft_viterbi / soft_viterbi_marginals",
        frames,
    )
