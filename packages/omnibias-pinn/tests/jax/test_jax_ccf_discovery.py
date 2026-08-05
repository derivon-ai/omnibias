# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CCF discovery harness + CAP export (jax).

Validates the harness rigorously via the method of manufactured solutions (MMS),
checks determinism and the exact omnibias-derivative path, and asserts the
CAP-ready export schema + independent residual recomputation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.jax.discovery import cap, ccf  # noqa: E402


def _np_hilbert(v: np.ndarray) -> np.ndarray:
    n = v.shape[-1]
    fk = np.fft.fft(v)
    m = np.fft.fftfreq(n) * n
    mult = -1j * np.sign(m)
    if n % 2 == 0:
        mult[n // 2] = 0.0
    return np.real(np.fft.ifft(fk * mult))


def test_discovery_deterministic() -> None:
    cfg = ccf.CCFDiscoveryConfig(hidden=16, n_grid=128, seed=0)
    a = ccf.run_ccf_discovery(cfg, steps=100, lr=5e-3)
    b = ccf.run_ccf_discovery(cfg, steps=100, lr=5e-3)
    assert np.array_equal(a.loss_history, b.loss_history)
    np.testing.assert_array_equal(a.theta, b.theta)


def test_discovery_unforced_smoke() -> None:
    cfg = ccf.CCFDiscoveryConfig(hidden=16, n_grid=128, seed=0, lam_init=0.6)
    res = ccf.run_ccf_discovery(cfg, steps=150, lr=5e-3)
    assert np.all(np.isfinite(res.loss_history))
    assert res.loss_history[-1] < res.loss_history[0]
    for key in ("max_abs_residual", "rms_residual", "spectral_tail_fraction"):
        assert key in res.diagnostics and np.isfinite(res.diagnostics[key])
    assert res.theta.shape == res.y.shape == res.residual.shape == (128,)


def test_profile_derivative_matches_autodiff() -> None:
    # The omnibias closed-form fastpath derivative must equal autodiff.
    cfg = ccf.CCFDiscoveryConfig(hidden=12, n_grid=64, parity="even", seed=3)
    params = ccf.init_params(cfg)
    y = ccf.make_grid(cfg)
    _, theta_y = ccf.profile(params, y, cfg.parity)

    def value(y_scalar):
        th, _ = ccf.profile(params, y_scalar.reshape(1), cfg.parity)
        return th[0]

    theta_y_ad = jax.vmap(jax.grad(value))(y)
    np.testing.assert_allclose(np.asarray(theta_y), np.asarray(theta_y_ad), atol=1e-10)


def test_mms_operator_forcing_consistency() -> None:
    # By construction R[theta*; lam*] - g == 0 (exact substitution check).
    cfg = ccf.CCFDiscoveryConfig(hidden=8, n_grid=192, parity="even", lam_init=0.5)
    theta_star = ccf.default_manufactured_profile()
    forcing, th, th_y = ccf.manufactured_forcing(cfg, theta_star, 0.5)
    from omnibias.pinn.jax.equations.cordoba_cordoba_fontelos import ccf_residual_samples

    y = ccf.make_grid(cfg)
    residual = ccf_residual_samples(y, th, th_y, 0.5) - forcing
    assert float(jnp.max(jnp.abs(residual))) < 1e-10


def test_mms_recovery() -> None:
    theta_star = ccf.default_manufactured_profile()
    lam_star = 0.5
    cfg = ccf.CCFDiscoveryConfig(
        hidden=32, n_grid=192, parity="even", lam_init=lam_star,
        train_lam=False, norm_weight=0.0, far_field_weight=0.0, seed=1,
    )
    forcing, th_star, _ = ccf.manufactured_forcing(cfg, theta_star, lam_star)
    res = ccf.run_ccf_discovery(cfg, steps=1000, lr=1e-2, forcing=forcing)
    assert res.loss_history[0] / res.loss_history[-1] > 100.0
    assert res.loss_history[-1] < 1e-2
    rmse = float(np.sqrt(np.mean((res.theta - np.asarray(th_star)) ** 2)))
    assert rmse < 5e-2, rmse


def test_lambda_training_runs() -> None:
    cfg = ccf.CCFDiscoveryConfig(hidden=16, n_grid=128, train_lam=True, lam_init=0.55, seed=2)
    res = ccf.run_ccf_discovery(cfg, steps=120, lr=3e-3)
    assert np.isfinite(res.lam)
    # lambda actually moved away from the init under joint training.
    assert abs(res.lam - 0.55) > 1e-6


def test_cap_bundle_schema_and_recompute(tmp_path) -> None:
    cfg = ccf.CCFDiscoveryConfig(hidden=16, n_grid=128, seed=0, lam_init=0.6)
    res = ccf.run_ccf_discovery(cfg, steps=80, lr=5e-3)
    bundle = cap.build_cap_bundle(res, reproduces_published_lambda=None, notes="test")
    assert cap.cap_schema_errors(bundle) == []

    # JSON round-trips losslessly.
    reloaded = json.loads(json.dumps(bundle))
    assert reloaded["schema_version"] == cap.SCHEMA_VERSION
    assert reloaded["honesty"]["navier_stokes_proof_claim"] is False
    assert reloaded["honesty"]["exact_solution_claim"] is False

    # Independently recompute the residual from validation_inputs only.
    vin = bundle["validation_inputs"]
    y = np.asarray(vin["y"])
    theta = np.asarray(vin["theta"])
    theta_y = np.asarray(vin["theta_y"])
    lam = vin["lambda"]
    recomputed = (1 + lam) * y * theta_y - lam * theta + _np_hilbert(theta) * theta_y
    np.testing.assert_allclose(
        recomputed, np.asarray(bundle["residual_samples"]), atol=1e-9
    )

    # Files are written.
    path = cap.write_cap_bundle(bundle, tmp_path)
    assert path.exists()
    assert (tmp_path / "ccf_cap_summary.md").exists()


def test_cap_schema_detects_missing_keys() -> None:
    errors = cap.cap_schema_errors({"schema_version": "x"})
    assert any("problem" in e for e in errors)
