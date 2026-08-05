# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared fixtures for cross-backend bit-parity tests.

The five Riccati-class activations all have closed-form derivative
towers in both backends; bit-parity is expected to ``rtol=1e-12,
atol=1e-12`` in float64.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp")


@pytest.fixture(params=_RICCATI)
def riccati(request) -> str:
    return request.param


@pytest.fixture
def shared_params():
    """Return matching parameter dicts (numpy arrays) consumed by both backends."""
    rng = np.random.default_rng(42)
    H, D, C = 8, 3, 3
    W = rng.normal(scale=0.5, size=(H, D)).astype(np.float64)
    beta = rng.normal(scale=0.1, size=(H,)).astype(np.float64)
    c = rng.normal(scale=0.5, size=(C, H)).astype(np.float64)
    b = rng.normal(scale=0.1, size=(C,)).astype(np.float64)
    coords = rng.normal(size=(7, D)).astype(np.float64)
    return dict(W=W, beta=beta, c=c, b=b, coords=coords, H=H, D=D, C=C)
