# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias Keras 3 quickstart (unified backend).

The same code runs on TensorFlow, JAX, or PyTorch. Select the backend
*before* importing keras via the KERAS_BACKEND environment variable:

    pip install omnibias-keras[jax]
    KERAS_BACKEND=jax python docs/examples/quickstart_keras.py
    KERAS_BACKEND=tensorflow python docs/examples/quickstart_keras.py
    KERAS_BACKEND=torch python docs/examples/quickstart_keras.py

Shows OMBU, OperatorBlock, and cmbDense as keras layers.
"""

from __future__ import annotations

import os

# Default to the JAX backend if the caller did not choose one.
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
from omnibias.keras import OMBU, OperatorBlock, cmbDense  # noqa: E402


def main() -> None:
    print("Keras backend:", keras.backend.backend())

    x = keras.ops.convert_to_tensor(
        [[0.1, -0.2, 0.3, 0.4]] * 8, dtype=keras.config.floatx()
    )

    # 1. OMBU: a trainable K-bias scalar operator as a keras.layers.Layer.
    ombu = OMBU(num_channels=4, K=2, base="tanh")
    print("OMBU output shape:", tuple(ombu(x).shape))

    # 2. OperatorBlock: typed scalar operator (closed-form Laplacian here).
    laplacian_block = OperatorBlock(channels=4, op="laplacian", base="gaussian")
    print("Laplacian block output shape:", tuple(laplacian_block(x).shape))

    # 3. cmbDense: keras.layers.Dense with an inline OperatorBlock.
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            cmbDense(units=16, op="identity", base="tanh"),
            cmbDense(units=1, op="identity", base="tanh"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    target = keras.ops.sum(x, axis=1, keepdims=True)
    history = model.fit(x, target, epochs=1, verbose=0)
    print(f"loss after one epoch: {history.history['loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
