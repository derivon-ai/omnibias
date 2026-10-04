# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""A damped curvature step on a genuine one-layer squared-error landscape."""

import jax
import jax.numpy as jnp
import numpy as np
from omnibias.curvature.sharpness import mse_loss_hessian
from visual_story import frame, line, story


def build_scene():
    jax.config.update("jax_enable_x64", True)
    X = jnp.linspace(-1, 1, 12)[:, None]
    Y = jnp.sin(2 * X[:, 0])
    W = jnp.array([[1.3]])
    beta = jnp.zeros(1)
    b, c = 0.7, -0.5
    frames = []
    path = []
    grid = np.linspace(-1.2, 1.5, 45)
    bb, cc = np.meshgrid(grid, grid)
    feature = np.asarray(jnp.tanh(X[:, 0] * W[0, 0]))
    loss = np.mean((bb[..., None] + cc[..., None] * feature - np.asarray(Y)) ** 2, axis=-1)
    for step in range(16):
        H, g = mse_loss_hessian(X, Y, W, beta, jnp.array([c]), jnp.array(b))
        h = np.asarray(H)[:2, :2]
        gradient = np.asarray(g)[:2]
        direction = -np.linalg.solve(h + 0.3 * np.eye(2), gradient)
        path.append((b, c))
        b += 0.25 * direction[0]
        c += 0.25 * direction[1]
        frames.append(
            frame(
                dict(
                    kind="landscape",
                    title="Loss over bias and readout",
                    x=grid,
                    y=grid,
                    values=loss,
                    path=path.copy(),
                    xlabel="output bias",
                    ylabel="readout coefficient",
                ),
                line(
                    "Local quadratic model along the step",
                    np.linspace(0, 1, 50),
                    (
                        "quadratic prediction",
                        float(np.mean((path[-1][0] + path[-1][1] * feature - np.asarray(Y)) ** 2))
                        + np.linspace(0, 1, 50) * (gradient @ direction)
                        + 0.5 * np.linspace(0, 1, 50) ** 2 * (direction @ h @ direction),
                    ),
                ),
                f"Damped step {step + 1}; only bias and readout move. The displayed Hessian block is exact for this slice.",
            )
        )
    return story(
        "curvature",
        "Let curvature shape the optimization step.",
        "A loss landscape, its local quadratic model and a damped update.",
        "omnibias.curvature.sharpness.mse_loss_hessian",
        frames,
    )
