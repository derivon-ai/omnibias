# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form / differentiable Gaussian-family convolution (torch backend).

The kernel taps are *exact cell integrals* of the (derivative-of-) Gaussian via
the ``erf`` antiderivative and the closed-form Gaussian derivative tower, so the
kernel carries no sampling error and is differentiable in a continuous scale.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.torch.blocks.conv import (
    AnalyticGaussianConv1d,
    AnalyticGaussianConv2d,
    analytic_gaussian_taps,
)

DT = torch.float64
_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _g(u: float) -> float:
    return math.exp(-0.5 * u * u)


def _ref_tap(j: int, sigma: float, order: int) -> float:
    """Independent pure-Python reference for a single cell-integrated tap."""
    a = (j - 0.5) / sigma
    b = (j + 0.5) / sigma
    if order == 0:
        return 0.5 * (math.erf(b / _SQRT2) - math.erf(a / _SQRT2))
    if order == 1:  # integral of g' over the cell == g(b) - g(a), scaled
        return (_g(b) - _g(a)) / (sigma * _SQRT_2PI)
    if order == 2:  # g'(u) = -u g(u)
        return (-b * _g(b) - (-a * _g(a))) / (sigma**2 * _SQRT_2PI)
    raise ValueError(order)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("sigma", [0.7, 1.3, 2.5])
def test_taps_match_exact_cell_integrals(order: int, sigma: float) -> None:
    K = 9
    half = K // 2
    taps = analytic_gaussian_taps(K, torch.tensor([sigma], dtype=DT), order)[0]
    ref = torch.tensor(
        [_ref_tap(j, sigma, order) for j in range(-half, half + 1)], dtype=DT
    )
    torch.testing.assert_close(taps, ref, rtol=1e-12, atol=1e-12)


def test_dog_equals_closed_form_gaussian_tower() -> None:
    """Derivative-of-Gaussian taps are the closed-form sigma tower (no sampling)."""
    from omnibias.torch.fastpath.hermite import gaussian_nth_derivative

    K, sigma = 11, 1.4
    half = K // 2
    offs = torch.arange(-half, half + 1, dtype=DT)
    a = (offs - 0.5) / sigma
    b = (offs + 0.5) / sigma
    for order in (1, 2, 3, 4):
        taps = analytic_gaussian_taps(K, torch.tensor([sigma], dtype=DT), order)[0]
        tower = (
            gaussian_nth_derivative(b, order - 1) - gaussian_nth_derivative(a, order - 1)
        ) / (sigma**order * _SQRT_2PI)
        torch.testing.assert_close(taps, tower, rtol=1e-12, atol=1e-12)


def test_order0_unit_area_and_symmetry() -> None:
    taps = analytic_gaussian_taps(41, torch.tensor([1.5], dtype=DT), 0)[0]
    assert float(taps.sum()) == pytest.approx(1.0, abs=1e-6)  # area -> 1 on a wide kernel
    torch.testing.assert_close(taps, taps.flip(0), rtol=1e-12, atol=1e-12)  # even


def test_order1_is_antisymmetric_zero_mean() -> None:
    taps = analytic_gaussian_taps(21, torch.tensor([1.3], dtype=DT), 1)[0]
    torch.testing.assert_close(taps, -taps.flip(0), rtol=1e-12, atol=1e-12)  # odd
    assert float(taps.sum()) == pytest.approx(0.0, abs=1e-12)


def test_taps_broadcast_over_channels() -> None:
    sigma = torch.tensor([0.8, 1.6, 3.0], dtype=DT)
    taps = analytic_gaussian_taps(7, sigma, 0)
    assert taps.shape == (3, 7)
    for c, s in enumerate((0.8, 1.6, 3.0)):
        torch.testing.assert_close(
            taps[c], analytic_gaussian_taps(7, torch.tensor([s], dtype=DT), 0)[0]
        )


def test_invalid_arguments() -> None:
    s = torch.tensor([1.0], dtype=DT)
    with pytest.raises(ValueError):
        analytic_gaussian_taps(8, s, 0)  # even kernel
    with pytest.raises(ValueError):
        analytic_gaussian_taps(0, s, 0)  # non-positive
    with pytest.raises(ValueError):
        analytic_gaussian_taps(7, s, -1)  # negative order


def _corr1d_same(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    b, c, length = x.shape
    k = taps.shape[1]
    half = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (half, half)))
    out = np.zeros_like(x)
    for i in range(k):
        out += taps[None, :, i, None] * xp[:, :, i : i + length]
    return out


def test_conv1d_matches_manual_depthwise_correlation() -> None:
    rng = np.random.default_rng(0)
    c, length, k = 3, 24, 7
    layer = AnalyticGaussianConv1d(c, k, sigma_init=1.2).double()
    x = torch.tensor(rng.standard_normal((2, c, length)), dtype=DT)
    out = layer(x).detach().numpy()
    taps = analytic_gaussian_taps(k, layer.sigma.detach(), 0).numpy()
    ref = _corr1d_same(x.numpy(), taps)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def _corr2d_same(x: np.ndarray, ker: np.ndarray) -> np.ndarray:
    b, c, h, w = x.shape
    kh, kw = ker.shape[1], ker.shape[2]
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.zeros_like(x)
    for a in range(kh):
        for bcol in range(kw):
            out += ker[None, :, a, bcol, None, None] * xp[:, :, a : a + h, bcol : bcol + w]
    return out


def test_conv2d_is_separable_outer_product() -> None:
    rng = np.random.default_rng(1)
    c, k = 3, 5
    layer = AnalyticGaussianConv2d(c, k, derivative_order=(1, 0), sigma_init=1.1).double()
    x = torch.tensor(rng.standard_normal((2, c, 10, 11)), dtype=DT)
    out = layer(x).detach().numpy()
    sigma = layer.sigma.detach()
    ty = analytic_gaussian_taps(k, sigma, 1)  # (C, Kh)
    tx = analytic_gaussian_taps(k, sigma, 0)  # (C, Kw)
    ker = (ty.unsqueeze(-1) * tx.unsqueeze(-2)).numpy()  # (C, Kh, Kw)
    ref = _corr2d_same(x.numpy(), ker)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_gradient_flows_to_continuous_scale_1d() -> None:
    layer = AnalyticGaussianConv1d(2, 7, sigma_init=1.3).double()
    x = torch.randn(4, 2, 20, dtype=DT)
    (layer(x) ** 2).sum().backward()
    assert layer.sigma.grad is not None
    assert torch.isfinite(layer.sigma.grad).all()
    assert float(layer.sigma.grad.abs().sum()) > 0.0


def test_gradient_flows_to_continuous_scale_2d() -> None:
    layer = AnalyticGaussianConv2d(2, 5, derivative_order=(0, 0), sigma_init=1.0).double()
    x = torch.randn(2, 2, 9, 9, dtype=DT)
    (layer(x) ** 2).sum().backward()
    assert layer.sigma.grad is not None
    assert float(layer.sigma.grad.abs().sum()) > 0.0


def test_frozen_scale_has_no_grad() -> None:
    layer = AnalyticGaussianConv1d(2, 7, sigma_init=1.0, learnable_sigma=False)
    assert not layer.sigma.requires_grad
    names = dict(layer.named_buffers())
    assert "sigma" in names
