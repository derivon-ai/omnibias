# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Periodic spectral Hilbert transform (torch): conventions + exactness."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.pinn.torch.hilbert import hilbert_transform

torch.set_default_dtype(torch.float64)


def _grid(n: int) -> np.ndarray:
    return -np.pi + 2.0 * np.pi * np.arange(n) / n


@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_hilbert_cos_sin_modes(k: int) -> None:
    y = _grid(128)
    hc = hilbert_transform(torch.cos(k * torch.tensor(y))).numpy()
    hs = hilbert_transform(torch.sin(k * torch.tensor(y))).numpy()
    np.testing.assert_allclose(hc, np.sin(k * y), atol=1e-10)
    np.testing.assert_allclose(hs, -np.cos(k * y), atol=1e-10)


def test_hilbert_constant_is_zero() -> None:
    y = _grid(64)
    out = hilbert_transform(torch.ones(len(y))).numpy()
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_hilbert_commutes_with_derivative() -> None:
    y = _grid(256)
    fp = -2.0 * np.sin(2 * y) + 1.5 * np.cos(3 * y)
    h_fp = hilbert_transform(torch.tensor(fp)).numpy()
    hf_prime = 2.0 * np.cos(2 * y) + 1.5 * np.sin(3 * y)
    np.testing.assert_allclose(h_fp, hf_prime, atol=1e-9)


def test_hilbert_dtype_preserved() -> None:
    y = torch.tensor(_grid(32), dtype=torch.float32)
    out = hilbert_transform(torch.cos(y))
    assert out.dtype == torch.float32


def test_hilbert_skew_adjoint_zero_mean() -> None:
    y = _grid(128)
    rng = np.random.default_rng(0)
    f = sum(rng.normal() * np.cos(k * y) + rng.normal() * np.sin(k * y) for k in range(1, 6))
    hf = hilbert_transform(torch.tensor(f)).numpy()
    assert abs(float(np.mean(f * hf))) < 1e-10


def test_hilbert_too_few_samples_raises() -> None:
    with pytest.raises(ValueError):
        hilbert_transform(torch.ones(1))
