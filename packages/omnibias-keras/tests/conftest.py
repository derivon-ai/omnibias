# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared fixtures / backend setup for the omnibias-keras test suite.

The Keras backend is chosen by the ``KERAS_BACKEND`` environment variable
(``tensorflow`` | ``jax`` | ``torch``); we default to ``torch`` if it is
unset. We run everything in float64 so the closed-form polynomial towers
can be checked against finite differences at a strict tolerance. On the
JAX backend that requires enabling x64 *before* keras imports.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

if os.environ["KERAS_BACKEND"] == "jax":
    # Must be set before jax is imported anywhere so float64 inputs are
    # preserved rather than silently truncated to float32.
    os.environ["JAX_ENABLE_X64"] = "true"
    import jax

    jax.config.update("jax_enable_x64", True)

import keras  # noqa: E402

keras.config.set_floatx("float64")
