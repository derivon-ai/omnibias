# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import jax
import jax.numpy as jnp
import numpy as np
from omnibias.jax.jet import jet_to_tower, mlp_jet
from visual_story import bars, frame, story


def build_scene():
    jax.config.update("jax_enable_x64", True)
    layers = [(jnp.array([[0.7, -0.3]]), jnp.array([0.1]), "tanh")]
    evaluate = jax.jit(jax.vmap(lambda x, v: jet_to_tower(mlp_jet(x, v, layers, order=3))[3, 0]))
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    frames = []
    x = jnp.array([[0.2, -0.1], [0.8, 0.3], [-0.5, 0.4]])
    for angle in angles:
        v = jnp.tile(jnp.array([np.cos(angle), np.sin(angle)]), (3, 1))
        y = np.asarray(evaluate(x, v))
        frames.append(
            frame(
                {
                    "kind": "vectors",
                    "title": "One direction across a batch",
                    "x": np.asarray(x[:, 0]),
                    "y": np.asarray(x[:, 1]),
                    "u": np.asarray(v[:, 0]),
                    "v": np.asarray(v[:, 1]),
                    "xlim": [-1, 1.5],
                    "ylim": [-1, 1.3],
                    "scale": 2,
                },
                bars(
                    "Third directional derivative",
                    y,
                    ["point A", "point B", "point C"],
                    ylim=[-0.9, 0.9],
                ),
                "Coordinates and directions are runtime inputs to the same traced function.",
            )
        )
    return story(
        "jax",
        "Trace once. Differentiate a batch.",
        "Functional directional jets compose with jit and vmap.",
        "omnibias.jax.jet.mlp_jet through jax.jit(jax.vmap)",
        frames,
    )
