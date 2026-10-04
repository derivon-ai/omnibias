# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-owned computed README illustration."""

import os

import numpy as np
from visual_story import flow, frame, line, story

os.environ.setdefault("KERAS_BACKEND", "torch")
from keras import ops
from omnibias.keras import OMBU


def build_scene():
    x = np.linspace(-2, 2, 81)
    unit = OMBU(num_channels=1, K=3, base="tanh")
    unit.build((None, 1))
    frames = []
    for k in range(4):
        y = ops.convert_to_numpy(
            unit.analytic_derivative(ops.convert_to_tensor(x[:, None]), order=k)
        ).ravel()
        for i in range(5):
            frames.append(
                frame(
                    flow(
                        "Keras layer contract",
                        [
                            "Input tensor",
                            "Operator-aware OMBU layer",
                            "Selected backend realization",
                            "Composable model output",
                        ],
                        min(i, 3),
                    ),
                    line(
                        f"Layer derivative order {k}",
                        x,
                        (f"analytic order {k}", y),
                        xlabel="layer input",
                    ),
                    "This scene evaluates the Torch-backed Keras realization; backend compatibility is separately tested.",
                )
            )
    return story(
        "keras",
        "Make operators part of the layer.",
        "A Keras model can consume analytic derivative outputs.",
        "omnibias.keras.OMBU.analytic",
        frames,
    )
