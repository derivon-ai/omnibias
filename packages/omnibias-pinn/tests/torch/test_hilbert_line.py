# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch whole-line Hilbert vs exact H[Q]=-P (free-Ω path)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from omnibias.pinn.torch.discovery import ccf_vorticity_neural as cvn  # noqa: E402
from omnibias.pinn.torch.equations.ccf_compactified import (  # noqa: E402
    alpha_from_lambda,
    hardy_even,
    hardy_odd,
    hilbert_transform_truncated_line,
)
from omnibias.pinn.torch.hilbert_line import (  # noqa: E402
    hilbert_gl_single_raw_tail,
    hilbert_wholeline_hp,
)


LAM = 0.6057
ALPHA = float(alpha_from_lambda(LAM))


@pytest.mark.parametrize(
    ("a", "ymax", "n"),
    [
        (1.3, 40.0, 401),
        (0.25, 20.0, 801),
    ],
)
def test_torch_wholeline_hp_beats_gl96_u48_on_planted_q(
    a: float, ymax: float, n: int
) -> None:
    """Free-Ω quadrature vs exact H[Q]=-P. Stretch 1e-13 is not claimed."""
    y = torch.linspace(-ymax, ymax, n, dtype=torch.float64)

    def omega_fn(t: torch.Tensor, a: float = a) -> torch.Tensor:
        return hardy_odd(t, a, ALPHA)

    values = omega_fn(y)
    h_exact = -hardy_even(y, a, ALPHA)
    h_old = hilbert_gl_single_raw_tail(
        y, values, omega_fn, y_trunc=ymax, n_gl=96, n_tail=48
    )
    h_new = hilbert_wholeline_hp(y, values, omega_fn, decay_power=ALPHA, y_trunc=ymax)
    core = y.abs() <= 0.9 * ymax
    err_old = float(torch.max(torch.abs((h_old - h_exact)[core])))
    err_new = float(torch.max(torch.abs((h_new - h_exact)[core])))
    assert math.isfinite(err_new)
    assert err_new < err_old
    if a >= 1.0:
        assert err_new < 1e-12
    else:
        assert err_new < 5e-13


def test_torch_wholeline_hp_is_even_for_odd_omega() -> None:
    y = torch.linspace(-20.0, 20.0, 401, dtype=torch.float64)

    def omega_fn(t: torch.Tensor) -> torch.Tensor:
        return hardy_odd(t, 1.3, ALPHA)

    h = hilbert_wholeline_hp(y, omega_fn(y), omega_fn, decay_power=ALPHA)
    assert float(torch.max(torch.abs(h - h.flip(0)))) < 1e-10


def test_torch_wholeline_hp_beats_fft_and_pv_on_hardy_q_reproduce_grid() -> None:
    """Reproduce-grid table: FFT/PV floor at O(1e-1); hp matches H[Q]=-P."""
    y = torch.linspace(-40.0, 40.0, 257, dtype=torch.float64)
    a = 1.3
    omega = hardy_odd(y, a, ALPHA)
    href = -hardy_even(y, a, ALPHA)
    core = y.abs() <= 36.0

    def omega_fn(t: torch.Tensor) -> torch.Tensor:
        return hardy_odd(t, a, ALPHA)

    err_fft = float(
        torch.max(torch.abs((hilbert_transform_truncated_line(y, omega) - href)[core]))
    )
    err_pv = float(torch.max(torch.abs((cvn.hilbert_pv_line(y, omega) - href)[core])))
    err_hp = float(
        torch.max(
            torch.abs(
                (hilbert_wholeline_hp(y, omega, omega_fn, decay_power=ALPHA, y_trunc=40.0) - href)[
                    core
                ]
            )
        )
    )
    assert err_fft > 5e-2
    assert err_pv > 5e-2
    assert err_hp <= 1e-8
    assert err_hp < 1e-12


def test_torch_wholeline_hp_agrees_with_href_on_free_neural_omega() -> None:
    """Unprojected compactified Ω: default hp vs higher-order href ≪ 1e-4."""
    torch.manual_seed(0)
    net = cvn.CompactifiedOmegaOMBU(hidden=8, activation="tanh")
    y = torch.linspace(-20.0, 20.0, 201, dtype=torch.float64)
    with torch.no_grad():
        if net._ombu is not None:
            net._ombu.c.bias.fill_(0.4)
        omega, _, _, _ = cvn.omega_from_net(net, y, lam=LAM, exp_core=True)

    def omega_fn(t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            om, _, _, _ = cvn.omega_from_net(net, t, lam=LAM, exp_core=True)
        return om

    href = hilbert_wholeline_hp(
        y,
        omega,
        omega_fn,
        decay_power=ALPHA,
        y_trunc=20.0,
        n_near=192,
        n_far=96,
        n_tail=128,
    )
    h = hilbert_wholeline_hp(y, omega, omega_fn, decay_power=ALPHA, y_trunc=20.0)
    core = y.abs() <= 18.0
    err = float(torch.max(torch.abs((h - href)[core])))
    assert math.isfinite(err)
    assert err < 1e-4
