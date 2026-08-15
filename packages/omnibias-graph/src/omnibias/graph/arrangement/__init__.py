# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Arrangement graph / Face-Net (theory 02-02, gated).

Sampling is a subgraph. ``beta -> inf`` is temperature collapse, not
founding ``delta -> 0``. Sound gap, not P vs NP.
"""

from __future__ import annotations

from omnibias.graph.arrangement._core import (
    ArrangementGraph,
    build_arrangement_graph,
    certify_facenet_gap,
    node_features,
)

__all__ = [
    "ArrangementGraph",
    "FaceNet",
    "build_arrangement_graph",
    "certify_facenet_gap",
    "node_features",
]


def __getattr__(name: str) -> object:
    if name == "FaceNet":
        from omnibias.graph.arrangement.torch import FaceNet

        return FaceNet
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
