# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The changing affinity is visible and drives the matrix in the same frame."""
import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omnibias.graph.torch.ops import graph_laplacian, sinkhorn_normalize

ROOT = Path(__file__).resolve().parents[1]


def test_graph_story_encodes_the_changing_edge():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from render_package_visuals import panel
        spec = importlib.util.spec_from_file_location("graph_scene", ROOT / "packages/omnibias-graph/docs/visuals/scene.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frames = module.build_scene()["frames"]
    finally:
        sys.path.pop(0)
    widths = []
    for frame in (frames[0], frames[-1]):
        left, right = frame["panels"]
        adjacency = torch.zeros((4, 4), dtype=torch.float64)
        for edge, weight, label in zip(left["edges"], left["edge_weights"], left["edge_labels"], strict=True):
            i, j = edge[:2]
            adjacency[i, j] = adjacency[j, i] = weight
            assert label == f"{weight:.2f}"
        expected = sinkhorn_normalize(-graph_laplacian(adjacency), n_iters=50).numpy()
        np.testing.assert_allclose(right["values"], expected)
        np.testing.assert_allclose(expected.sum(axis=0), 1, atol=1e-10)
        np.testing.assert_allclose(expected.sum(axis=1), 1, atol=1e-6)
        fig, ax = plt.subplots()
        panel(ax, left, False)
        widths.append(ax.lines[1].get_linewidth())
        plt.close(fig)
    assert widths[0] < widths[1]
    np.testing.assert_array_equal(frames[0]["panels"][0]["nodes"], frames[-1]["panels"][0]["nodes"])
