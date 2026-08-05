# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression tests for the fractional / hyperdissipative Navier-Stokes track.

Covers the shared pure math (`fractional_ns_theory`) plus the two learnable-order
recovery paths. Every assertion also pins the honesty invariants: no routine
claims a global-regularity result or an omnibias-verified analytic theorem.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from examples.certified_fluid_dynamics.fractional_ns_theory import (
    CRITICAL_ALPHA_3D,
    classify_log_supercritical,
    classify_regime,
    exact_beltrami_shell,
    exact_decaying_abc,
    exact_decaying_shear,
    exact_decaying_shear_log,
    fractional_ns_residual,
    fractional_ns_residual_torch,
    log_supercritical_rate,
    tao_dissipation_symbol,
    tao_dissipation_symbol_torch,
    tao_log_supercritical_diagnostic,
)

ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5]
NU = 0.05
M = 2
K_SHELL = 2


@pytest.mark.parametrize("alpha", ALPHAS)
def test_exact_shear_residual_is_machine_zero(alpha: float) -> None:
    u, p, u_t, _ = exact_decaying_shear(16, 0.0, m=M, nu=NU, alpha=alpha)
    res, div = fractional_ns_residual(u, p, u_t, alpha=alpha, nu=NU)
    assert float(np.max(np.abs(res))) < 1e-10
    assert float(np.max(np.abs(div))) < 1e-10


@pytest.mark.parametrize("alpha", ALPHAS)
def test_wrong_alpha_residual_is_nonzero(alpha: float) -> None:
    # The operator genuinely depends on alpha: the same field scored with a
    # different exponent must NOT satisfy the equation.
    u, p, u_t, _ = exact_decaying_shear(16, 0.0, m=M, nu=NU, alpha=alpha)
    res_wrong, _ = fractional_ns_residual(u, p, u_t, alpha=alpha + 0.25, nu=NU)
    assert float(np.max(np.abs(res_wrong))) > 1e-3


@pytest.mark.parametrize("alpha", ALPHAS)
def test_exact_abc_residual_is_machine_zero(alpha: float) -> None:
    u, p, u_t = exact_decaying_abc(16, 0.3, nu=NU)
    res, div = fractional_ns_residual(u, p, u_t, alpha=alpha, nu=NU)
    assert float(np.max(np.abs(res))) < 1e-10
    assert float(np.max(np.abs(div))) < 1e-10


def test_decay_rate_matches_prediction() -> None:
    T = 1.0
    for alpha in ALPHAS:
        u0, _, _, rate = exact_decaying_shear(16, 0.0, m=M, nu=NU, alpha=alpha)
        uT, _, _, _ = exact_decaying_shear(16, T, m=M, nu=NU, alpha=alpha)
        measured = -float(np.log(np.sum(uT * uT) / np.sum(u0 * u0))) / (2.0 * T)
        assert abs(measured - rate) < 1e-10
        assert abs(rate - NU * M ** (2.0 * alpha)) < 1e-12


def test_regime_labels_and_honesty() -> None:
    classical = classify_regime(1.0)
    assert classical["global_regularity_status"] == "open"
    assert classical["is_classical_open_problem"] is True
    crit = classify_regime(CRITICAL_ALPHA_3D)
    assert crit["global_regularity_status"] == "proven_global_regularity_external"
    assert crit["regime"] == "critical"
    assert classify_regime(1.5)["regime"] == "subcritical"
    for a in ALPHAS:
        row = classify_regime(a)
        assert row["unproven_claim"] is False
        assert row["omnibias_verified"] is False


def test_tao_log_supercritical_divergence_threshold() -> None:
    # Tao (2009): global regularity iff int dr/(r g^4) = inf, i.e. 4 beta <= 1.
    assert tao_log_supercritical_diagnostic(0.25)["tao_global_regularity_applies"] is True
    assert tao_log_supercritical_diagnostic(0.5)["tao_global_regularity_applies"] is False
    for beta in (0.125, 0.25, 0.5, 1.0):
        cls = classify_log_supercritical(beta)
        assert cls["unproven_claim"] is False
        assert cls["omnibias_verified"] is False


def test_learnable_order_recovers_operator() -> None:
    from examples.certified_fluid_dynamics.run_fractional_ns import recover_order

    row = recover_order(1.0, n=16, steps=200, lr=0.05, seed=0)
    assert row["abs_err"] < 1e-2


def test_learnable_pinn_recovers_order_and_field() -> None:
    import torch

    from examples.certified_fluid_dynamics.run_fractional_pinn import train_recover

    torch.set_num_threads(1)
    args = Namespace(
        alpha_true=1.0, alpha_init=0.6, modes=4, hidden=16, depth=2, ny=16,
        n_snapshots=6, collocation=64, nu=NU, T=1.0, steps=250, lr=3e-3,
        alpha_lr_mult=20.0, phys_weight=1.0, log_every=1000, seed=2026,
    )
    result = train_recover(args)
    assert result["alpha_abs_err"] < 0.05
    assert result["field_rel_l2"] < 0.05


# --------------------------------------------------------------------------- #
# Beltrami-shell exact solution (alpha-dependent 3D flow for the GPU PINN).    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", ALPHAS)
def test_beltrami_shell_residual_is_machine_zero(alpha: float) -> None:
    u, p, u_t, _ = exact_beltrami_shell(16, 0.3, wavenumber=K_SHELL, nu=NU, alpha=alpha)
    res, div = fractional_ns_residual(u, p, u_t, alpha=alpha, nu=NU)
    assert float(np.max(np.abs(res))) < 1e-9
    assert float(np.max(np.abs(div))) < 1e-9


@pytest.mark.parametrize("alpha", ALPHAS)
def test_beltrami_shell_rate_is_alpha_dependent(alpha: float) -> None:
    # The decay rate nu K^{2 alpha} genuinely depends on alpha (K > 1).
    _, _, _, rate = exact_beltrami_shell(16, 0.0, wavenumber=K_SHELL, nu=NU, alpha=alpha)
    assert abs(rate - NU * K_SHELL ** (2.0 * alpha)) < 1e-12


def test_beltrami_shell_torch_residual_matches_numpy_machine_zero() -> None:
    import torch

    u, p, u_t, _ = exact_beltrami_shell(16, 0.3, wavenumber=K_SHELL, nu=NU, alpha=1.0)
    res, div = fractional_ns_residual_torch(
        torch.tensor(u), torch.tensor(p), torch.tensor(u_t), alpha=1.0, nu=NU
    )
    assert float(res.abs().max()) < 1e-9
    assert float(div.abs().max()) < 1e-9


# --------------------------------------------------------------------------- #
# Log-supercritical dissipation: torch symbol + learnable-beta recovery.       #
# --------------------------------------------------------------------------- #
def test_tao_symbol_torch_matches_numpy() -> None:
    import torch

    k2 = np.linspace(0.0, 40.0, 21)
    for beta in (0.125, 0.25, 0.5):
        npv = tao_dissipation_symbol(k2.copy(), beta=beta)
        tv = tao_dissipation_symbol_torch(torch.tensor(k2), torch.tensor(float(beta))).numpy()
        assert float(np.max(np.abs(npv - tv))) < 1e-9


def test_tao_symbol_torch_is_differentiable_in_beta() -> None:
    import torch

    k2 = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    beta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    tao_dissipation_symbol_torch(k2, beta).sum().backward()
    assert beta.grad is not None
    assert float(beta.grad.abs().max()) > 0.0


def test_log_supercritical_shear_rate_matches() -> None:
    for m in (1, 2, 3):
        for beta in (0.15, 0.25, 0.6):
            _, _, _, rate = exact_decaying_shear_log(16, 0.0, m=m, nu=NU, beta=beta)
            assert abs(rate - log_supercritical_rate(m, nu=NU, beta=beta)) < 1e-12


def test_learnable_beta_pinn_recovers_beta_and_side() -> None:
    import torch

    from examples.certified_fluid_dynamics.run_log_supercritical_pinn import train_recover

    torch.set_num_threads(1)
    args = Namespace(
        beta_true=0.6, beta_init=0.35, modes=6, hidden=16, depth=2, ny=24,
        n_snapshots=8, collocation=64, nu=NU, T=1.0, steps=800, lr=3e-3,
        beta_lr_mult=20.0, phys_weight=1.0, log_every=10000, seed=2026,
    )
    result = train_recover(args)
    assert result["beta_abs_err"] < 0.08
    # beta = 0.6 is safely on the "open" side of Tao's 0.25 threshold.
    assert result["regularity_side_recovered"] is True
