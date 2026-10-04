# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
import torch
from omnibias.graph.torch.ops import graph_laplacian, sinkhorn_normalize
from visual_story import frame, story


def build_scene():
    frames = []
    nodes = np.column_stack([np.cos(np.arange(4) * np.pi / 2), np.sin(np.arange(4) * np.pi / 2)])
    for scale in np.linspace(0.2, 3, 20):
        A = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.3],
                [1.0, 0.0, scale, 0.0],
                [0.0, scale, 0.0, 1.0],
                [0.3, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
        )
        L = graph_laplacian(A)
        P = sinkhorn_normalize(-L, n_iters=50)
        frames.append(
            frame(
                {
                    "kind": "graph",
                    "title": "Weighted node relationships",
                    "nodes": nodes,
                    "edges": [[0, 1], [1, 2, True], [2, 3], [3, 0]],
                    "edge_weights": [float(A[i, j]) for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]],
                    "edge_labels": [f"{float(A[i, j]):.2f}" for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]],
                },
                {
                    "kind": "heatmap",
                    "title": "Doubly stochastic relaxation",
                    "values": P.detach().numpy(),
                    "vmin": 0,
                    "vmax": 1,
                    "xlabel": "destination",
                    "ylabel": "source",
                },
                f"Highlighted affinity {scale:.2f}; normalization gives soft assignments, not a decoded permutation.",
            )
        )
    return story(
        "graph",
        "From relations to soft assignments.",
        "Spectral operators and matrix relaxations expose graph structure.",
        "omnibias.graph.torch.ops.graph_laplacian / sinkhorn_normalize",
        frames,
    )
