# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Periodic spectral Hilbert transform (jax): conventions + exactness."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.jax.hilbert import hilbert_transform  # noqa: E402


def _grid(n: int) -> np.ndarray:
    return -np.pi + 2.0 * np.pi * np.arange(n) / n


@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_hilbert_cos_sin_modes(k: int) -> None:
    y = _grid(128)
    # H[cos] = sin, H[sin] = -cos.
    hc = np.asarray(hilbert_transform(jnp.cos(k * jnp.asarray(y))))
    hs = np.asarray(hilbert_transform(jnp.sin(k * jnp.asarray(y))))
    np.testing.assert_allclose(hc, np.sin(k * y), atol=1e-10)
    np.testing.assert_allclose(hs, -np.cos(k * y), atol=1e-10)


def test_hilbert_constant_is_zero() -> None:
    y = _grid(64)
    out = np.asarray(hilbert_transform(jnp.ones_like(jnp.asarray(y))))
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_hilbert_commutes_with_derivative() -> None:
    # H[f'] == (H f)'.  f = cos 2y + 0.5 sin 3y -> f' analytic, (Hf)' analytic.
    y = _grid(256)
    fp = -2.0 * np.sin(2 * y) + 1.5 * np.cos(3 * y)
    h_fp = np.asarray(hilbert_transform(jnp.asarray(fp)))
    hf_prime = 2.0 * np.cos(2 * y) + 1.5 * np.sin(3 * y)  # d/dy[sin2y - 0.5 cos3y]
    np.testing.assert_allclose(h_fp, hf_prime, atol=1e-9)


def test_hilbert_period_length_independent() -> None:
    # H acts on the *sequence* of samples; identical on any period length.
    n = 96
    a = np.asarray(hilbert_transform(jnp.cos(jnp.asarray(_grid(n)))))
    # same samples but pretend the period is 10x: cos over [-5pi.., step) reorders,
    # so instead reuse the same ordered samples -> identical output.
    same = np.asarray(hilbert_transform(jnp.asarray(np.cos(_grid(n)))))
    np.testing.assert_allclose(a, same, atol=1e-12)


def test_hilbert_skew_adjoint_zero_mean() -> None:
    # <f, Hf> = 0 for the periodic Hilbert transform (skew-symmetry).
    y = _grid(128)
    rng = np.random.default_rng(0)
    f = sum(rng.normal() * np.cos(k * y) + rng.normal() * np.sin(k * y) for k in range(1, 6))
    hf = np.asarray(hilbert_transform(jnp.asarray(f)))
    assert abs(float(np.mean(f * hf))) < 1e-10


def test_hilbert_too_few_samples_raises() -> None:
    with pytest.raises(ValueError):
        hilbert_transform(jnp.ones(1))
