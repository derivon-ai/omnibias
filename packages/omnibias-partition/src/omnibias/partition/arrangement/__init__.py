# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Arrangement geometry (theory 01-03, gated).

Sampling is a subgraph / lower bound, never a complete face lattice.
``beta -> inf`` is temperature collapse, not founding ``delta -> 0``.
"""

from __future__ import annotations

from omnibias.partition.arrangement._core import (
    Arrangement,
    CellGapCertificate,
    affine_values,
    brute_force_cells,
    certify_cell_gap,
    enumerate_cells_vertices,
    general_position_normals,
    margin,
    max_cells,
    realized_cells,
    sign_vector,
    soft_membership,
    tope_graph,
    tree_arrangement,
)

__all__ = [
    "Arrangement",
    "CellGapCertificate",
    "affine_values",
    "brute_force_cells",
    "certify_cell_gap",
    "enumerate_cells_vertices",
    "general_position_normals",
    "margin",
    "max_cells",
    "realized_cells",
    "sign_vector",
    "soft_membership",
    "tope_graph",
    "tree_arrangement",
]
