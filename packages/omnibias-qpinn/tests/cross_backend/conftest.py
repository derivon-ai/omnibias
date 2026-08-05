# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared fixtures for cross-backend bit-parity tests (qpinn).

Both backends are exercised on a :class:`OneLayerVectorField` carrying
a ``(psi_re, psi_im)`` group. We seed numpy, then copy identical
weights into the torch and jax instances; the closed-form n-th
derivative kernels in both packages are bit-stable in float64 and the
parity contract is therefore ``rtol=1e-9, atol=1e-12``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


_RICCATI = ("tanh", "sigmoid", "gaussian")


@pytest.fixture(params=_RICCATI)
def riccati(request) -> str:
    return request.param


@pytest.fixture
def shared_psi_params():
    """Return matching parameters + coords for a OneLayerVectorField with
    ``(psi_re, psi_im)`` group."""
    rng = np.random.default_rng(2026_05_27)
    H, D, C = 8, 2, 2  # 2 input axes (x, t) and 2 output channels
    W = rng.normal(scale=0.5, size=(H, D)).astype(np.float64)
    beta = rng.normal(scale=0.1, size=(H,)).astype(np.float64)
    c = rng.normal(scale=0.5, size=(C, H)).astype(np.float64)
    b = rng.normal(scale=0.1, size=(C,)).astype(np.float64)
    coords = rng.normal(size=(7, D)).astype(np.float64)
    return dict(W=W, beta=beta, c=c, b=b, coords=coords, H=H, D=D, C=C)


@pytest.fixture
def shared_psi_params_1d():
    """Same as ``shared_psi_params`` but on a 1D spatial domain (no
    time axis); used for TISE."""
    rng = np.random.default_rng(0xDEADBEEF)
    H, D, C = 8, 1, 2
    W = rng.normal(scale=0.5, size=(H, D)).astype(np.float64)
    beta = rng.normal(scale=0.1, size=(H,)).astype(np.float64)
    c = rng.normal(scale=0.5, size=(C, H)).astype(np.float64)
    b = rng.normal(scale=0.1, size=(C,)).astype(np.float64)
    coords = np.linspace(-2.0, 2.0, 17).reshape(-1, 1).astype(np.float64)
    return dict(W=W, beta=beta, c=c, b=b, coords=coords, H=H, D=D, C=C)


@pytest.fixture
def shared_psi_params_2d():
    """2D spatial (x, y) parameters used for the rotating-NLS demo."""
    rng = np.random.default_rng(0xC0FFEE)
    H, D, C = 12, 2, 2
    W = rng.normal(scale=0.5, size=(H, D)).astype(np.float64)
    beta = rng.normal(scale=0.1, size=(H,)).astype(np.float64)
    c = rng.normal(scale=0.5, size=(C, H)).astype(np.float64)
    b = rng.normal(scale=0.1, size=(C,)).astype(np.float64)
    coords = rng.normal(scale=0.7, size=(11, D)).astype(np.float64)
    return dict(W=W, beta=beta, c=c, b=b, coords=coords, H=H, D=D, C=C)


def _allclose(t, j, *, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    """Bit-parity allclose for a torch tensor vs jax array."""
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return np.allclose(np.asarray(t), np.asarray(j), rtol=rtol, atol=atol)
