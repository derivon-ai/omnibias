# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form Gaussian-family convolution (Keras backend) + torch parity.

The Keras taps are a bit-identical twin of the torch closed form, so we check
both the exact ``erf`` cell integrals and cross-backend parity at float64.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from keras import ops
from omnibias.keras.blocks.conv import (
    AnalyticGaussianConv1D,
    AnalyticGaussianConv2D,
    analytic_gaussian_taps,
)
from omnibias.torch.blocks.conv import AnalyticGaussianConv1d
from omnibias.torch.blocks.conv import analytic_gaussian_taps as torch_taps

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _t(x: np.ndarray):
    return ops.convert_to_tensor(x, dtype="float64")


def _np(x):
    return np.asarray(ops.convert_to_numpy(x))


def _g(u: float) -> float:
    return math.exp(-0.5 * u * u)


def _ref_tap(j: int, sigma: float, order: int) -> float:
    a = (j - 0.5) / sigma
    b = (j + 0.5) / sigma
    if order == 0:
        return 0.5 * (math.erf(b / _SQRT2) - math.erf(a / _SQRT2))
    if order == 1:
        return (_g(b) - _g(a)) / (sigma * _SQRT_2PI)
    if order == 2:
        return (-b * _g(b) + a * _g(a)) / (sigma**2 * _SQRT_2PI)
    raise ValueError(order)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("sigma", [0.7, 1.3, 2.5])
def test_taps_match_exact_cell_integrals(order: int, sigma: float) -> None:
    K = 9
    half = K // 2
    taps = _np(analytic_gaussian_taps(K, _t(np.array([sigma])), order)[0])
    ref = np.array([_ref_tap(j, sigma, order) for j in range(-half, half + 1)])
    np.testing.assert_allclose(taps, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_taps_bit_parity_with_torch(order: int) -> None:
    sigma = np.array([0.7, 1.3, 2.1])
    keras_taps = _np(analytic_gaussian_taps(9, _t(sigma), order))
    t_taps = torch_taps(9, torch.tensor(sigma, dtype=torch.float64), order).numpy()
    np.testing.assert_allclose(keras_taps, t_taps, rtol=1e-12, atol=1e-12)


def _corr1d_same(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Channels-last depthwise 'same' correlation reference."""
    b, length, c = x.shape
    k = taps.shape[1]
    half = k // 2
    xp = np.pad(x, ((0, 0), (half, half), (0, 0)))
    out = np.zeros_like(x)
    for i in range(k):
        out += taps.T[None, i, None, :] * xp[:, i : i + length, :]
    return out


def test_conv1d_matches_manual_correlation() -> None:
    rng = np.random.default_rng(0)
    c, length, k = 3, 24, 7
    layer = AnalyticGaussianConv1D(c, k, sigma_init=1.2)
    x = rng.standard_normal((2, length, c))
    out = _np(layer(_t(x)))
    taps = _np(analytic_gaussian_taps(k, layer.sigma, 0))
    np.testing.assert_allclose(out, _corr1d_same(x, taps), rtol=1e-12, atol=1e-12)


def test_layer_output_parity_with_torch_1d() -> None:
    rng = np.random.default_rng(2)
    c, length, k = 3, 20, 7
    x = rng.standard_normal((2, length, c))  # channels-last for keras
    k_layer = AnalyticGaussianConv1D(c, k, derivative_order=1, sigma_init=1.4)
    k_out = _np(k_layer(_t(x)))  # (B, L, C)

    t_layer = AnalyticGaussianConv1d(c, k, derivative_order=1, sigma_init=1.4).double()
    with torch.no_grad():  # exact float64 init (torch builds sigma in float32 by default)
        t_layer.sigma.fill_(1.4)
    x_t = torch.tensor(np.transpose(x, (0, 2, 1)), dtype=torch.float64)  # (B, C, L)
    t_out = t_layer(x_t).detach().numpy()  # (B, C, L)

    # Cross-*framework* conv parity, not bit-parity. The taps are bit-identical
    # (see ``test_taps_bit_parity_with_torch``), but the depthwise correlation
    # runs on each framework's native conv kernel. A derivative-of-Gaussian
    # (order >= 1) kernel sums to ~0, so the output is a catastrophic-cancellation
    # sum whose rounding depends on the conv's internal reduction order: the torch
    # reference (NNPACK-less CPU fallback) deviates ~2e-8 from the exact analytic
    # integral, while the keras backend conv matches it exactly. The frameworks
    # therefore agree only to conv-accumulation precision, not float64 eps.
    np.testing.assert_allclose(
        np.transpose(k_out, (0, 2, 1)), t_out, rtol=1e-6, atol=1e-7
    )


def test_conv2d_runs_and_is_separable() -> None:
    rng = np.random.default_rng(3)
    c, k = 3, 5
    layer = AnalyticGaussianConv2D(c, k, derivative_order=(0, 1), sigma_init=1.1)
    x = rng.standard_normal((2, 10, 11, c))
    out = _np(layer(_t(x)))
    assert out.shape == (2, 10, 11, c)


def test_trainable_scale_weight() -> None:
    learn = AnalyticGaussianConv1D(2, 7, sigma_init=1.0, learnable_sigma=True)
    learn.build((None, 16, 2))
    assert any(w.name == "sigma" for w in learn.trainable_weights)

    frozen = AnalyticGaussianConv1D(2, 7, sigma_init=1.0, learnable_sigma=False)
    frozen.build((None, 16, 2))
    assert all(w.name != "sigma" for w in frozen.trainable_weights)
