# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression tests for DeepMind-style CCF vorticity neural discoverer."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

from omnibias.pinn.certified.ccf_hardy import (  # noqa: E402
    certified_ccf_hardy_wholeline_blowup_attempt,
)
from omnibias.pinn.jax.discovery.ccf_vorticity import (  # noqa: E402
    leading_mode_far_field_cancel,
)
from omnibias.pinn.torch.discovery import ccf_vorticity_neural as cvn  # noqa: E402
from omnibias.pinn.torch.discovery import multistage as ms  # noqa: E402


def test_leading_mode_far_field_cancels_linear_operator() -> None:
    out = leading_mode_far_field_cancel(lam=0.6057)
    assert out["expected_cancel_factor"] == pytest.approx(0.0, abs=1e-12)
    assert out["far_lin_max"] < 1e-2
    assert out["far_lin_times_y_2alpha"] < 1.0


def test_neural_omega_far_field_decays_like_y_to_minus_alpha() -> None:
    """Ω = y·(1+y²)^{-(α+1)/2}·hat must decay as |y|^{-α}, not grow as |y|^{1-α}."""
    lam = 0.6057
    alpha = 1.0 / (1.0 + lam)
    net = cvn.CompactifiedOmegaOMBU(hidden=8, activation="tanh")
    with torch.no_grad():
        if net._ombu is not None:
            net._ombu.c.bias.fill_(0.5)  # softplus core ≈ O(1) at q→0
        y = torch.linspace(20.0, 80.0, 61, dtype=torch.float64)
        omega, _, _, _ = cvn.omega_from_net(net, y, lam=lam, exp_core=True)
        # |Ω| · |y|^α → finite; |Ω| · |y|^{α-1} → 0 (rules out growth |y|^{1-α})
        scaled = torch.abs(omega) * torch.pow(torch.abs(y), alpha)
        wrong = torch.abs(omega) * torch.pow(torch.abs(y), alpha - 1.0)
        assert float(torch.max(scaled)) < 10.0
        assert float(torch.min(scaled)) > 1e-6
        assert float(torch.median(wrong)) < float(torch.median(scaled)) * 0.2


def test_hilbert_pv_line_finite_and_odd_to_even() -> None:
    """PV-line Hilbert is finite and maps odd Ω to an even field."""
    y = torch.linspace(-8.0, 8.0, 201, dtype=torch.float64)
    omega = y * torch.exp(-y * y)  # odd
    h = cvn.hilbert_pv_line(y, omega)
    assert torch.isfinite(h).all()
    # H(odd) should be even: H(-y) ≈ H(y)
    assert float(torch.max(torch.abs(h - h.flip(0)))) < 1e-8


def test_hardy_corrected_hilbert_matches_exact_atom() -> None:
    """On a pure Hardy atom the corrected Hilbert recovers H[Q]=-P to ~1e-10."""
    from omnibias.pinn.torch.equations.ccf_compactified import hardy_even, hardy_odd

    y = torch.linspace(-40.0, 40.0, 801, dtype=torch.float64)
    a, g = 1.3, 1.0 / (1.0 + 0.6057)
    omega = hardy_odd(y, a, g)
    h_exact = -hardy_even(y, a, g)
    scales = torch.tensor([a], dtype=torch.float64)
    gammas = torch.tensor([g], dtype=torch.float64)
    h, _u, _c, defect = cvn.hardy_corrected_hu_from_omega(
        y, omega, scales=scales, gammas=gammas
    )
    assert float(defect) < 1e-10
    assert float(torch.max(torch.abs(h - h_exact))) < 1e-10


def test_deep_jetmlp_omega_path_smoke() -> None:
    """depth>=2 uses JetMLP closed-form dq and still yields finite Wang fields."""
    net = cvn.CompactifiedOmegaOMBU(hidden=8, depth=2, activation="tanh")
    y = torch.linspace(-3.0, 3.0, 21, dtype=torch.float64)
    with torch.no_grad():
        omega, omega_y, _, _ = cvn.omega_from_net(net, y, lam=0.6057, exp_core=True)
        assert torch.isfinite(omega).all()
        assert torch.isfinite(omega_y).all()
        assert float(torch.max(torch.abs(omega + omega.flip(0)))) < 1e-10  # odd


def test_grad_norm_downweights_peak() -> None:
    out = cvn.grad_norm_downweights_peak()
    assert out["peak_to_mean_norm"] < out["peak_to_mean_abs"]
    assert out["norm_peak"] < out["abs_peak"]


def test_neural_cubic_gn_residual_vector_contract() -> None:
    cfg = cvn.CCFVorticityNeuralConfig(
        n_grid=33,
        hidden=8,
        n_scales=2,
        n_gamma_multiples=2,
        cubic_gn_steps=2,
        qr_gn_steps=1,
        y_max=6.0,
        seed=0,
        d2_weight=0.0,
        resample_every=1,
        train_hilbert="hardy_projection",
    )
    result = cvn.run_ccf_vorticity_neural_discovery(cfg)
    assert result.y.shape == result.omega.shape == result.residual.shape
    assert result.coeffs.shape == result.scales.shape == result.gammas.shape
    assert np.isfinite(result.diagnostics["max_abs_vorticity_residual"])
    assert result.extra["rung_metric_uses_fft"] is False
    assert result.extra["train_hilbert"] == "hardy_projection"
    assert result.extra["rung_hilbert"] == "hardy_projection_exact"
    assert result.extra["use_grad_norm"] is True
    assert result.diagnostics["omega_max_abs"] > 1e-4


def test_default_train_hilbert_matches_rung_metric() -> None:
    """Earn-path default: train Hilbert is Hardy, same family as Rung/CAP."""
    cfg = cvn.CCFVorticityNeuralConfig()
    assert cfg.train_hilbert == "hardy_projection"
    assert cfg.adam_warmup_steps == 0


def test_train_hardy_residual_agrees_with_projection_fields() -> None:
    """Hardy train path uses exact H[Q]=-P fields in the Wang residual."""
    y = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    scales, gammas = cvn.hardy_dictionary(lam=0.6057, n_scales=2, n_gamma_multiples=2)
    sc = torch.as_tensor(scales, dtype=torch.float64)
    gs = torch.as_tensor(gammas, dtype=torch.float64)
    from omnibias.pinn.torch.equations.ccf_compactified import hardy_odd

    omega = 0.2 * hardy_odd(y, float(scales[0]), float(gammas[0]))
    omega_y = torch.gradient(omega, spacing=(y,))[0]
    r_h, defect, _, _ = cvn.vorticity_fields(
        y,
        omega,
        omega_y,
        lam=0.6057,
        scales=sc,
        gammas=gs,
        train_hilbert="hardy_projection",
        hilbert_n_uniform=None,
    )
    r_s, _, _, _ = cvn.vorticity_fields(
        y,
        omega,
        omega_y,
        lam=0.6057,
        scales=sc,
        gammas=gs,
        train_hilbert="truncated_line_spectral",
        hilbert_n_uniform=None,
    )
    assert float(defect) < 1e-8
    # Exact Hardy residual must differ from truncated spectral unless coincidence.
    assert float(torch.max(torch.abs(r_h - r_s))) > 1e-6 or float(
        torch.max(torch.abs(r_h))
    ) < 1e-2
    assert torch.isfinite(r_h).all()


def test_projection_defect_reported() -> None:
    y = np.linspace(-4.0, 4.0, 81)
    scales, gammas = cvn.hardy_dictionary(lam=0.6057, n_scales=3, n_gamma_multiples=2)
    from omnibias.pinn.torch.equations.ccf_compactified import hardy_odd

    yt = torch.as_tensor(y, dtype=torch.float64)
    om = 0.3 * hardy_odd(yt, float(scales[0]), float(gammas[0]))
    coeffs, defect, fields = cvn.project_omega_hardy(
        yt, om, scales=scales, gammas=gammas
    )
    assert defect < 1e-8
    assert fields["H"].shape == y.shape
    assert coeffs.shape[0] == scales.shape[0]


def test_linearized_msnn_smoke() -> None:
    y = np.linspace(-4.0, 4.0, 41)
    stage1 = 0.05 * np.sin(y) * np.exp(-0.1 * y * y)
    omy0 = np.gradient(stage1, y)

    def residual_fn(phi: np.ndarray) -> np.ndarray:
        return phi + 0.1 * np.gradient(phi, y)

    out = ms.correct_profile(
        y,
        stage1,
        residual_fn,
        cfg=ms.MultiStageConfig(steps=15, hidden=8, n_fourier=6, linearized=True),
        omega_y0=omy0,
    )
    assert out["linearized"] is True
    assert out["optimizer"] == "stage2_heuristic_adam"
    assert np.isfinite(out["max_abs_residual_after"])
    assert out["composed"].shape == stage1.shape


def test_iterate_multistage_labels_optimizer() -> None:
    y = np.linspace(-3.0, 3.0, 31)
    stage1 = 0.02 * y * np.exp(-0.2 * y * y)

    def residual_fn(phi: np.ndarray) -> np.ndarray:
        return phi

    out = ms.iterate_multistage(
        y,
        stage1,
        residual_fn,
        rounds=2,
        cfg=ms.MultiStageConfig(steps=5, hidden=6, n_fourier=4, linearized=True),
        optimizer="adam",
    )
    assert out["rounds_run"] >= 1
    assert "stage2_heuristic" in str(out["optimizer"]) or out["optimizer"] == "adam"


def test_martens_grosse_neural_smoke_decreases_or_finite() -> None:
    cfg = cvn.reproduce_deepmind_config(
        n_grid=21,
        hidden=6,
        mg_steps=3,
        qr_gn_steps=1,
        adam_warmup_steps=2,
        dense_n_val=51,
        y_max=5.0,
        n_scales=2,
        n_gamma_multiples=1,
        mg_solver="qr",
        d2_weight=0.0,
        resample_every=0,
        seed=1,
        # Tiny smoke: spectral Hilbert keeps MG history finite on coarse grids.
        train_hilbert="truncated_line_spectral",
        proj_defect_weight=0.0,
        omega_peak_floor=0.02,
        nontrivial_weight=5.0,
    )
    result = cvn.run_ccf_vorticity_neural_discovery(cfg)
    hist = result.extra["train_history_max_abs"]
    assert len(hist) >= 1
    assert all(np.isfinite(h) for h in hist)
    assert np.isfinite(result.diagnostics["reproduction_dense_max_abs"])


def test_dense_neural_metric_anti_ghost() -> None:
    net = cvn.CompactifiedOmegaOMBU(hidden=4, activation="tanh")
    scales, gammas = cvn.hardy_dictionary(lam=0.6057, n_scales=2, n_gamma_multiples=1)
    sc = torch.as_tensor(scales, dtype=torch.float64)
    gs = torch.as_tensor(gammas, dtype=torch.float64)
    dense = cvn.dense_neural_vorticity_residual(
        net,
        lam=0.6057,
        train_hilbert="truncated_line_spectral",
        scales=sc,
        gammas=gs,
        y_max=6.0,
        n_val=81,
        exp_core=True,
    )
    assert "reproduction_dense_max_abs_for_gate" in dense
    assert np.isfinite(dense["reproduction_dense_max_abs_for_gate"])


def test_anti_ghost_uses_pre_rescale_gauge() -> None:
    """Hard rescale must not hide a gauge miss from the gate score."""
    net = cvn.CompactifiedOmegaOMBU(hidden=4, activation="tanh")
    with torch.no_grad():
        # Tiny peak / wrong gauge amplitude → anti-ghost should floor to >= 1.
        if net._ombu is not None:
            net._ombu.c.bias.fill_(-8.0)
    scales, gammas = cvn.hardy_dictionary(lam=0.6057, n_scales=2, n_gamma_multiples=1)
    sc = torch.as_tensor(scales, dtype=torch.float64)
    gs = torch.as_tensor(gammas, dtype=torch.float64)
    dense = cvn.dense_neural_vorticity_residual(
        net,
        lam=0.6057,
        train_hilbert="truncated_line_spectral",
        scales=sc,
        gammas=gs,
        y_max=6.0,
        n_val=81,
        exp_core=True,
        gauge_value=0.05,
    )
    # After rescale omega_gauge_sample≈0.05, but raw profile fails nontriviality.
    assert dense["reproduction_dense_max_abs_for_gate"] >= 1.0
    assert abs(dense["omega_gauge_sample"] - 0.05) < 1e-6 or dense["omega_max_abs_raw"] < 0.02


def test_warm_mismatch_full_cold_reset() -> None:
    """Architecture mismatch must rebuild the net, not keep a hybrid strict=False load."""
    cfg = cvn.CCFVorticityNeuralConfig(
        hidden=8,
        depth=1,
        n_grid=17,
        n_scales=2,
        n_gamma_multiples=1,
        cubic_gn_steps=0,
        qr_gn_steps=1,
        mg_steps=0,
        adam_warmup_steps=0,
        y_max=4.0,
        dense_n_val=41,
        d2_weight=0.0,
        resample_every=0,
        train_hilbert="hardy_projection",
        optimizer="cubic_gauss_newton",
        seed=0,
    )
    wide = cvn.CompactifiedOmegaOMBU(hidden=16, depth=1, activation="tanh")
    warm = {k: v.detach().cpu().clone() for k, v in wide.state_dict().items()}
    result = cvn.run_ccf_vorticity_neural_discovery(cfg, warm_state_dict=warm)
    net = result.extra["net"]
    assert isinstance(net, cvn.CompactifiedOmegaOMBU)
    # Cold path: width matches cfg.hidden=8, not the warm width=16.
    assert net._ombu is not None
    assert int(net._ombu.c.weight.shape[0]) == 1
    assert int(net._ombu.W.weight.shape[0]) == 8
    assert np.isfinite(result.diagnostics["reproduction_dense_max_abs_for_gate"])


def test_campaign_tick_phase0_while_stretch_uncleared() -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "benchmarks"))
    from deepmind_campaign_tick import run_tick

    tick = run_tick(smoke=True, family="1st_unstable")
    assert tick["stage"] == "phase0_reproduce_neural"
    assert tick["gates"]["stretch_1e-13_cleared"] is False
    assert tick["honesty"]["navier_stokes_proof_claim"] is False
    assert np.isfinite(tick["diagnosis"]["reproduction_dense_residual"])


def test_vorticity_cap_does_not_forge_whole_line() -> None:
    cert = certified_ccf_hardy_wholeline_blowup_attempt(
        coeffs=[0.4, -0.1, 0.02],
        scales=[0.8, 1.5, 2.4],
        gammas=[0.62, 1.24, 1.86],
        lam=0.6057,
        form="vorticity",
        residual_gate=1e-11,
        velocity_sign=-1.0,
    )
    assert cert["form"] == "vorticity"
    assert cert["honesty"]["navier_stokes_proof_claim"] is False
    assert cert["honesty"]["whole_line_certified"] is False
    assert "quantified_gap" in cert["closure_report"]


def test_acceptance_config_rejects_multi_alpha_collapse_flag() -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[4] / "benchmarks" / "ccf_hardy_rung_acceptance.py"
    text = path.read_text(encoding="utf-8")
    assert "multi_alpha_collapse" in text
    assert "Fold higher-alpha energy" not in text
    assert 'form="vorticity"' in text
    assert 'train_hilbert="hardy_projection"' in text
    assert "MartensGrosseGN" in text or "martens" in text.lower()
    assert "OMNIBIAS_SUBMIT" in text
    # Spectral Hilbert remains available as diagnostic API, not earn default.
    assert "spectral_hu_from_omega" in (
        Path(__file__).resolve().parents[4]
        / "packages/omnibias-pinn/src/omnibias/pinn/torch/discovery/ccf_vorticity_neural.py"
    ).read_text(encoding="utf-8")
