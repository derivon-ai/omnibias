# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Face-Net / arrangement graph G1/G2 (theory 02-02). Subgraph sampling."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.graph.arrangement._core import (
    build_arrangement_graph,
    certify_facenet_gap,
)
from omnibias.partition.arrangement import Arrangement, brute_force_cells, tope_graph


def _square() -> Arrangement:
    # h = Wx - t  with  x+1, -x+1, y+1, -y+1
    normals = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    offsets = np.array([-1.0, -1.0, -1.0, -1.0])
    return Arrangement(normals, offsets)


def test_g1_graph_matches_brute() -> None:
    arr = _square()
    xs = np.linspace(-2.0, 2.0, 9)
    ys = np.linspace(-2.0, 2.0, 9)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    samples = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    graph = build_arrangement_graph(arr, samples)
    brute = brute_force_cells(arr)
    assert set(graph.cells) == set(brute)
    brute_edges = set(tope_graph(brute))
    found = {(min(u, v), max(u, v)) for u, v, _k in graph.edges}
    # Reindex brute edges by cell identity.
    index = {cell: i for i, cell in enumerate(graph.cells)}
    expect = set()
    listed = list(brute)
    for i, j in tope_graph(listed):
        a, b = index[listed[i]], index[listed[j]]
        expect.add((min(a, b), max(a, b)))
    assert found == expect
    _ = brute_edges
    assert graph.representatives.shape[0] == len(graph.cells)


def test_g2_gap_sound() -> None:
    logits = (0.2, -0.5, 1.1, 0.0)
    cert = certify_facenet_gap(logits, beta=5.0)
    assert cert["sound"] is True
    assert cert["p_vs_np"] is False
    assert cert["theorem_prover_verified"] is False
    assert cert["temperature_collapse"] is True


def test_g6_torch_runs() -> None:
    pytest.importorskip("torch")
    import torch
    from omnibias.graph.arrangement.torch import FaceNet

    torch.set_default_dtype(torch.float64)
    arr = _square()
    samples = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
    graph = build_arrangement_graph(arr, samples)
    net = FaceNet(2, hidden=4, rounds=2, dtype=torch.float64)
    out = net(graph, arr)
    assert out.shape[0] == len(graph.cells)
    assert torch.isfinite(out).all()
