# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CCF self-similar residual (torch): exact substitution + field op."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch import equations as teq
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.equations.cordoba_cordoba_fontelos import ccf_residual_samples
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField

torch.set_default_dtype(torch.float64)


def _grid(n: int) -> np.ndarray:
    return -np.pi + 2.0 * np.pi * np.arange(n) / n


def _np_hilbert(v: np.ndarray) -> np.ndarray:
    n = v.shape[-1]
    fk = np.fft.fft(v)
    m = np.fft.fftfreq(n) * n
    mult = -1j * np.sign(m)
    if n % 2 == 0:
        mult[n // 2] = 0.0
    return np.real(np.fft.ifft(fk * mult))


def test_ccf_residual_exact_closed_form_transport() -> None:
    y = _grid(96)
    lam = 0.6057
    theta = np.cos(2 * y)
    theta_y = -2.0 * np.sin(2 * y)
    expected = (1 + lam) * y * theta_y - lam * theta - 2.0 * np.sin(2 * y) ** 2
    got = ccf_residual_samples(
        torch.tensor(y), torch.tensor(theta), torch.tensor(theta_y), lam
    ).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_ccf_residual_exact_closed_form_flux() -> None:
    y = _grid(96)
    lam = 0.5
    theta = np.cos(2 * y)
    theta_y = -2.0 * np.sin(2 * y)
    nonlocal_term = theta_y * np.sin(2 * y) + theta * (2.0 * np.cos(2 * y))
    expected = (1 + lam) * y * theta_y - lam * theta + nonlocal_term
    got = ccf_residual_samples(
        torch.tensor(y), torch.tensor(theta), torch.tensor(theta_y), lam, form="flux"
    ).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_ccf_field_op_matches_numpy_substitution() -> None:
    y = _grid(64).reshape(-1, 1)
    cspec = CoordinateSpec(axes=("y",), periodicity=(True,), time_axis=None)
    mspec = ComponentSpec(names=("theta",), groups={})
    field = OneLayerVectorField(
        coordinate_spec=cspec, components=mspec, hidden=6, base="tanh"
    )
    state = field(torch.tensor(y))
    out = teq.cordoba_cordoba_fontelos(state, lam=0.6, form="transport")
    th = tops.value(state, "theta").detach().numpy()
    thy = tops.derivative(state, "theta", axis="y", order=1).detach().numpy()
    expected = (1 + 0.6) * y[:, 0] * thy - 0.6 * th + _np_hilbert(th) * thy
    np.testing.assert_allclose(out.residual.detach().numpy(), expected, atol=1e-10)


def test_ccf_supports_backward() -> None:
    y = _grid(48).reshape(-1, 1)
    cspec = CoordinateSpec(axes=("y",), periodicity=(True,), time_axis=None)
    mspec = ComponentSpec(names=("theta",), groups={})
    field = OneLayerVectorField(
        coordinate_spec=cspec, components=mspec, hidden=5, base="tanh"
    )
    out = teq.cordoba_cordoba_fontelos(field(torch.tensor(y)), lam=0.6)
    loss = (out.residual ** 2).mean()
    loss.backward()
    grads = [p.grad for p in field.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_ccf_rejects_multi_spatial() -> None:
    cspec = CoordinateSpec(axes=("x", "y"), periodicity=(True, True), time_axis=None)
    mspec = ComponentSpec(names=("theta",), groups={})
    field = OneLayerVectorField(
        coordinate_spec=cspec, components=mspec, hidden=4, base="tanh"
    )
    coords = torch.tensor(np.random.default_rng(0).normal(size=(16, 2)))
    with pytest.raises(ValueError, match="1 spatial axis"):
        teq.cordoba_cordoba_fontelos(field(coords))
