# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression: CCF honesty — freeze-lam GN, Theta residual origin, no CAP forge."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_theta_transport_residual_at_origin_is_minus_lam_theta0() -> None:
    """Even profiles cannot clear the Theta residual at y=0 unless Theta(0)=0."""
    from omnibias.pinn.jax.equations.ccf_compactified import alpha_from_lambda
    from omnibias.symbolic.ccf import ccf_self_similar_residual, hardy_profile_numpy

    lam = 0.6057
    y = np.array([0.0])
    th, thp, h, hp = hardy_profile_numpy(
        y, np.array([1.0, -0.2]), np.array([0.8, 1.5]), float(alpha_from_lambda(lam))
    )
    r = ccf_self_similar_residual(
        y,
        th,
        thp,
        lam,
        hilbert_convention="hardy_exact",
        hilbert_values=h,
        hilbert_y_values=hp,
    )
    assert float(r[0]) == pytest.approx(-lam * float(th[0]), rel=0.0, abs=1e-12)


def test_gn_path_freezes_lambda_when_train_lam_false() -> None:
    from omnibias.pinn.jax.discovery import ccf_line

    cfg = ccf_line.CCFLineDiscoveryConfig(
        n_terms=3,
        n_grid=32,
        y_max=8.0,
        seed=0,
        optimizer="gauss_newton",
        gn_steps=15,
        lam_init=0.6057,
        train_lam=False,
        adaptive_every=100,
    )
    result = ccf_line.run_ccf_line_discovery(cfg, steps=15, funnel_updates=0)
    assert result.lam == pytest.approx(0.6057, abs=1e-12)
    assert float(result.params["lam"]) == pytest.approx(0.6057, abs=1e-12)


def test_whole_line_certified_not_forged_without_closure() -> None:
    from omnibias.pinn.certified.ccf_hardy import (
        certified_ccf_hardy_wholeline_blowup_attempt,
        certified_ccf_hardy_wholeline_blowup_attempt_schema_errors,
    )

    cert = certified_ccf_hardy_wholeline_blowup_attempt(
        coeffs=[1.0, -0.2, 0.05],
        scales=[0.7, 1.4, 2.2],
        lam=0.6057,
        residual_gate=1e-11,
    )
    assert certified_ccf_hardy_wholeline_blowup_attempt_schema_errors(cert) == []
    # Random candidate must not silently claim whole-line certification.
    if not cert["collocation_closure_certified"] or not cert["honesty"]["whole_line_certified"]:
        assert cert["honesty"]["whole_line_certified"] is False
        assert cert["closure_certified"] is False
    if cert["honesty"]["whole_line_certified"]:
        assert cert["closure_certified"] is True
        assert cert["closure_report"]["residual_certified_sup"] <= 1e-11


def test_earn_path_default_train_hilbert_is_hardy() -> None:
    from omnibias.pinn.torch.discovery import ccf_vorticity_neural as cvn

    assert cvn.CCFVorticityNeuralConfig().train_hilbert == "hardy_projection"
    assert cvn.CCFVorticityNeuralConfig().adam_warmup_steps == 0
    assert cvn.CCFVorticityNeuralConfig().arm == "earn"
    assert cvn.CCFVorticityNeuralConfig().optimizer == "cubic_gauss_newton"


def test_earn_arm_forbids_adam_warmup() -> None:
    from omnibias.pinn.torch.discovery import ccf_vorticity_neural as cvn

    cfg = cvn.CCFVorticityNeuralConfig(
        arm="earn",
        adam_warmup_steps=5,
        n_grid=17,
        hidden=4,
        cubic_gn_steps=1,
        qr_gn_steps=0,
        n_scales=2,
        n_gamma_multiples=1,
    )
    with pytest.raises(ValueError, match="forbids Adam"):
        cvn.run_ccf_vorticity_neural_discovery(cfg)


def test_reproduce_config_allows_adam_and_martens_grosse() -> None:
    from omnibias.pinn.torch.discovery import ccf_vorticity_neural as cvn

    cfg = cvn.reproduce_deepmind_config(
        n_grid=17,
        hidden=4,
        mg_steps=2,
        qr_gn_steps=1,
        adam_warmup_steps=2,
        dense_n_val=41,
        y_max=4.0,
        n_scales=2,
        n_gamma_multiples=1,
        mg_solver="qr",
        d2_weight=0.0,
        resample_every=0,
    )
    assert cfg.arm == "reproduce"
    assert cfg.optimizer == "martens_grosse"
    assert cfg.train_hilbert == "hardy_corrected_pv"
    assert cfg.adam_warmup_steps == 2
    result = cvn.run_ccf_vorticity_neural_discovery(cfg)
    assert "MartensGrosse" in str(result.extra["optimizer"])
    assert np.isfinite(result.diagnostics["reproduction_dense_max_abs_for_gate"])
    assert result.extra["arm"] == "reproduce"


def test_ccf_absolute_gates_require_both_lambda_and_residual() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from _gates import ccf_absolute_gates

    # Lambda alone is not reproduction.
    g = ccf_absolute_gates(lam=0.6057, max_abs_residual=1.0, family="1st_unstable")
    assert g["earned"] is False
    assert g["honesty"]["reproduces_published_lambda"] is False
    # Both clear.
    g2 = ccf_absolute_gates(
        lam=0.6057, max_abs_residual=1e-12, family="1st_unstable"
    )
    assert g2["earned"] is True
    assert g2["honesty"]["reproduces_published_lambda"] is True


def test_vorticity_discovery_module_imports_and_dense_helper() -> None:
    from omnibias.pinn.jax.discovery import ccf_vorticity

    scales = np.array([0.8, 1.5, 2.2])
    alphas = np.array([1.62, 1.62, 3.62])
    coeffs = np.array([0.1, -0.05, 0.02])
    dense = ccf_vorticity.dense_vorticity_residual(
        coeffs, scales, alphas, 0.6057, n_val=201, y_max=10.0
    )
    assert dense["dense_max_abs_vorticity"] >= 0.0
    assert np.isfinite(dense["dense_max_abs_vorticity"])


def test_vorticity_train_residual_uses_exact_hardy_hilbert() -> None:
    """Earn-path residual must be Hardy-Ω exact Hilbert (HΩ=-P), not FFT spectral H."""
    from omnibias.pinn.jax.discovery import ccf_vorticity

    y = jnp.asarray([-1.0, -0.3, 0.0, 0.3, 1.0], dtype=jnp.float64)
    coeffs = jnp.asarray([0.2, -0.05], dtype=jnp.float64)
    scales = jnp.asarray([0.9, 1.7], dtype=jnp.float64)
    alphas = jnp.asarray([1.62, 3.24], dtype=jnp.float64)
    lam = 0.6057
    _r, fields = ccf_vorticity.vorticity_residual_samples(y, coeffs, scales, alphas, lam)
    # Closed-form Hardy atoms: Ω = Σ c Q, HΩ = -Σ c P.
    yy = y[:, None]
    aa = scales[None, :]
    gg = alphas[None, :]
    cc = coeffs[None, :]
    r = jnp.hypot(aa, yy)
    phi = jnp.arctan2(yy, aa)
    om_exact = jnp.sum(cc * (r ** (-gg)) * jnp.sin(gg * phi), axis=1)
    huy_exact = jnp.sum(cc * (-(r ** (-gg)) * jnp.cos(gg * phi)), axis=1)
    assert jnp.allclose(fields["omega"], om_exact, atol=1e-12)
    assert jnp.allclose(fields["U_y"], huy_exact, atol=1e-12)
    assert bool(jnp.all(jnp.isfinite(_r)))


def test_vorticity_discovery_reports_qr_mg_and_hardy_train_hilbert() -> None:
    from omnibias.pinn.jax.discovery import ccf_vorticity

    cfg = ccf_vorticity.CCFVorticityDiscoveryConfig(
        n_scales=2, n_gamma_multiples=1, n_grid=17, gn_steps=4, y_max=8.0, seed=0
    )
    out = ccf_vorticity.run_ccf_vorticity_discovery(cfg)
    assert out.extra["train_hilbert"] == "hardy_exact_omega"
    assert out.extra["gn_solver"] == "qr"
    assert out.extra["optimizer"] == "martens_grosse_gn"
    assert out.extra["hilbert_convention"] == "hardy_exact_omega"


def test_near_null_profile_is_not_a_rung1_win() -> None:
    """Cancelling micro-scale Hardy ghosts can fake tiny residual without gauge."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from _gates import ccf_absolute_gates

    # Mimic acceptance anti-ghost: if gauge lost, residual-for-gate is forced ≥ 1.
    omega_gauge = 1e-10
    omega_max = 1e-7
    dense_residual = 1e-12
    gauge_ok = abs(omega_gauge - 0.05) <= 0.01
    nontrivial = omega_max >= 0.02
    residual_for_gate = dense_residual if (gauge_ok and nontrivial) else max(dense_residual, 1.0)
    g = ccf_absolute_gates(
        lam=0.6057, max_abs_residual=residual_for_gate, family="1st_unstable"
    )
    assert g["earned"] is False
    assert residual_for_gate >= 1.0
