# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Line-domain CCF discovery smoke tests (Hardy ansatz)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax.discovery import ccf_line, lambda_laws  # noqa: E402
from omnibias.pinn.jax.discovery import polish_mp  # noqa: E402


def test_ccf_line_adam_smoke_descent() -> None:
    cfg = ccf_line.CCFLineDiscoveryConfig(
        n_terms=3, n_grid=32, y_max=8.0, seed=0, optimizer="adam", lam_init=0.6
    )
    result = ccf_line.run_ccf_line_discovery(cfg, steps=20, lr=5e-3)
    assert result.loss_history.size >= 2
    assert result.y.shape == result.theta.shape == result.residual.shape
    assert np.isfinite(result.diagnostics["max_abs_residual"])
    assert result.extra["domain"] == "line_compactified"
    assert result.extra["hilbert_convention"] == "hardy_exact"


def test_ccf_line_funnel_smoke() -> None:
    cfg = ccf_line.CCFLineDiscoveryConfig(
        n_terms=3, n_grid=24, y_max=6.0, seed=1, optimizer="adam", lam_init=0.55
    )
    result = ccf_line.run_ccf_line_discovery(cfg, steps=30, lr=5e-3, funnel_updates=2)
    assert result.funnel is not None
    assert len(result.funnel.lambdas) == 2


def test_ccf_line_gauss_newton_smoke() -> None:
    cfg = ccf_line.CCFLineDiscoveryConfig(
        n_terms=3,
        n_grid=24,
        y_max=6.0,
        seed=2,
        optimizer="gauss_newton",
        gn_steps=4,
        adaptive_every=10,
        lam_init=0.6,
    )
    result = ccf_line.run_ccf_line_discovery(cfg, steps=4)
    assert result.loss_history.size >= 2
    assert np.isfinite(result.diagnostics["rms_residual"])


def test_lambda_laws_init_only() -> None:
    lam = lambda_laws.predict_lambda_init(1, family="ipm")
    assert 0.4 < lam < 0.5
    with pytest.raises(RuntimeError, match="anti-circularity"):
        lambda_laws.assert_not_reference_value("test")


def test_mpmath_polish_smoke() -> None:
    nodes = np.linspace(0.3, 4.0, 8)
    out = polish_mp.polish_hardy_ccf(
        coeffs=np.array([1.0, -0.2]),
        scales=np.array([0.8, 1.5]),
        lam=0.6,
        nodes=nodes,
        dps=25,
        max_iter=3,
    )
    assert np.isfinite(out["max_abs_residual_float64"])
    assert np.isfinite(out["max_abs_residual_mpmath"])
    assert out["honesty"]["navier_stokes_proof_claim"] is False
