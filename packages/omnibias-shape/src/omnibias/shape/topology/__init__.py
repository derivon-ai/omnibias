# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable topology surrogates (theory 03-09).

No differentiable function equals a Betti number. ``beta -> inf``
is temperature collapse (feasibility), not founding bias collapse
(``delta -> 0``). Do not conflate the two.
"""

from __future__ import annotations

from omnibias.shape.topology._core import (
    Inconclusive,
    PersistencePair,
    SoftCount,
    bimodal_saddle_index,
    bottleneck_distance,
    certified_component_count,
    cluster_laplacian,
    connected_components_1d,
    cubical_euler_grad,
    cubical_euler_value,
    cubical_faces_2d,
    digital_components_2d,
    digital_genus,
    euler_regularizer,
    euler_value_bound,
    exact_h0_persistence,
    field_euler_characteristic,
    field_euler_pair,
    field_to_occupancy,
    honesty_payload,
    occupancy_euler,
    persistence_loss,
    persistence_loss_grad,
    regularize_occupancy,
    soft_component_count,
    soft_euler_characteristic,
    soft_persistence,
)

__all__ = [
    "Inconclusive",
    "PersistencePair",
    "SoftCount",
    "bimodal_saddle_index",
    "bottleneck_distance",
    "certified_component_count",
    "cluster_laplacian",
    "connected_components_1d",
    "cubical_euler_grad",
    "cubical_euler_value",
    "cubical_faces_2d",
    "digital_components_2d",
    "digital_genus",
    "euler_regularizer",
    "euler_value_bound",
    "exact_h0_persistence",
    "field_euler_characteristic",
    "field_euler_pair",
    "field_to_occupancy",
    "honesty_payload",
    "occupancy_euler",
    "persistence_loss",
    "persistence_loss_grad",
    "regularize_occupancy",
    "soft_component_count",
    "soft_euler_characteristic",
    "soft_persistence",
]
