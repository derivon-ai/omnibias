# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import numpy as np
from omnibias.partition import PartitionConfig, init_params, partition_weights
from visual_story import frame, line, story


def build_scene():
    x = np.linspace(-3, 3, 101)
    params = init_params(PartitionConfig(n_features=1, depth=1), rng=0)
    frames = []
    experts = np.column_stack([-0.3 * x + 0.1, 0.45 * x + 0.5])
    for beta in np.geomspace(0.5, 12, 20):
        w = partition_weights(params, x[:, None], beta=float(beta))
        blend = (w * experts).sum(axis=1)
        frames.append(
            frame(
                line(
                    "Region membership",
                    x,
                    ("region A", w[:, 0]),
                    ("region B", w[:, 1]),
                    ylim=[-0.1, 1.1],
                    xlabel="input",
                ),
                line(
                    "Experts become one field",
                    x,
                    ("blended output", blend),
                    ("expert A", experts[:, 0]),
                    ("expert B", experts[:, 1]),
                    ylim=[-1.5, 2],
                    xlabel="input",
                ),
                f"β = {beta:.2f}: nonnegative memberships sum to one; ties remain explicit in the hard limit.",
            )
        )
    return story(
        "partition",
        "Learn where each expert applies.",
        "Regions, memberships and expert outputs form one trainable composition.",
        "omnibias.partition.partition_weights",
        frames,
    )
