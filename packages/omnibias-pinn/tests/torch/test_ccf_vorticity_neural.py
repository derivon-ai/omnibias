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


def test_hilbert_pv_mapped_tail_recovers_hardy_q() -> None:
    """Mapped |t|>Y tail beats truncated PV on a planted Hardy Q (H[Q]=-P)."""
    from omnibias.pinn.torch.equations.ccf_compactified import hardy_even, hardy_odd

    y = torch.linspace(-40.0, 40.0, 401, dtype=torch.float64)
    a, g = 1.3, 1.0 / (1.0 + 0.6057)
    omega = hardy_odd(y, a, g)
    h_exact = -hardy_even(y, a, g)
    h_pv = cvn.hilbert_pv_line(y, omega)
    h_tail = cvn.hilbert_pv_mapped_tail(
        y,
        omega,
        decay_power=g,
        omega_pos_fn=lambda t: hardy_odd(t, a, g),
        n_quad=64,
        n_gl=96,
    )
    core = y.abs() <= 0.9 * 40.0
    err_pv = float(torch.max(torch.abs((h_pv - h_exact)[core])))
    err_tail = float(torch.max(torch.abs((h_tail - h_exact)[core])))
    assert torch.isfinite(h_tail).all()
    assert float(torch.max(torch.abs(h_tail - h_tail.flip(0)))) < 1e-8
    # Junction |y|~Y is excluded. GL interior + mapped tail is spectral.
    assert err_pv > 0.05  # truncation floor on the core
    assert err_tail < 2e-3
    assert err_tail < 0.05 * err_pv


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


def test_wang_linearized_gn_requires_torch_residual() -> None:
    y = np.linspace(-2.0, 2.0, 17)
    stage1 = 0.1 * y

    def residual_fn(phi: np.ndarray) -> np.ndarray:
        return 2.0 * phi + 0.3

    with pytest.raises(ValueError, match="residual_fn_torch"):
        ms.correct_profile(
            y,
            stage1,
            residual_fn,
            cfg=ms.MultiStageConfig(steps=2, hidden=4, n_fourier=3, linearized=True),
            optimizer="wang_linearized_gn",
        )


def test_wang_linearized_residual_matches_exact_affine_d() -> None:
    import torch

    stage1 = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)
    corr = torch.sin(stage1)

    def residual_fn_torch(phi: torch.Tensor) -> torch.Tensor:
        return 2.0 * phi + 0.3

    r0 = residual_fn_torch(stage1)
    r_lin = ms.wang_linearized_residual_torch(
        residual_fn_torch,
        stage1,
        corr,
        r0=r0,
        eps=0.01,
        fd_eps=1e-6,
        linearized=True,
    )
    expected = r0 + 0.01 * 2.0 * corr
    assert torch.allclose(r_lin, expected, atol=1e-10)
    proxy = corr + r0 / 0.01
    assert float(torch.max(torch.abs(r_lin - proxy))) > 1.0


def test_wang_linearized_gn_reduces_affine() -> None:
    import torch

    y = np.linspace(-1.0, 1.0, 31)
    stage1 = np.zeros_like(y)
    target = np.sin(2.0 * np.pi * y)

    def residual_fn(phi: np.ndarray) -> np.ndarray:
        return 2.0 * phi - target

    def residual_fn_torch(phi: torch.Tensor) -> torch.Tensor:
        return 2.0 * phi - torch.as_tensor(target, dtype=torch.float64)

    out = ms.correct_profile(
        y,
        stage1,
        residual_fn,
        cfg=ms.MultiStageConfig(steps=5, hidden=8, n_fourier=6, eps=1.0, linearized=True),
        optimizer="wang_linearized_gn",
        residual_fn_torch=residual_fn_torch,
    )
    assert out["optimizer"] == "wang_linearized_gn"
    assert out["max_abs_residual_after"] < out["max_abs_residual_before"]


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


def test_deepmind_paper_and_signed_configs_keep_honesty_flags() -> None:
    paper = cvn.deepmind_paper_architecture_config(n_grid=17, hidden=8)
    signed = cvn.deepmind_signed_hat_config(n_grid=17, hidden=8)
    assert paper.exp_core is True
    assert signed.exp_core is False
    assert paper.optimizer == "martens_grosse"
    assert signed.optimizer == "martens_grosse"
    assert paper.use_grad_norm is True
    assert paper.train_hilbert == "wholeline_hp"
    assert paper.adam_warmup_steps == 0


def test_pv_mapped_tail_neural_smoke_finite() -> None:
    """Free-Ω mapped-tail Hilbert path stays finite (not a stretch claim)."""
    cfg = cvn.reproduce_deepmind_config(
        n_grid=21,
        hidden=6,
        mg_steps=2,
        qr_gn_steps=0,
        adam_warmup_steps=1,
        dense_n_val=41,
        y_max=5.0,
        n_scales=2,
        n_gamma_multiples=1,
        d2_weight=0.0,
        resample_every=0,
        seed=2,
        train_hilbert="pv_mapped_tail",
        proj_defect_weight=0.0,
        hilbert_n_aux=65,
        hilbert_n_quad=16,
        omega_peak_floor=0.02,
        nontrivial_weight=5.0,
    )
    result = cvn.run_ccf_vorticity_neural_discovery(cfg)
    assert result.extra["train_hilbert"] == "pv_mapped_tail"
    assert np.isfinite(result.diagnostics["reproduction_dense_max_abs"])
    assert np.isfinite(result.diagnostics["max_abs_vorticity_residual"])


def test_wholeline_hp_neural_smoke_finite() -> None:
    """Free-Ω hp Hilbert path stays finite (not a stretch claim)."""
    cfg = cvn.reproduce_deepmind_config(
        n_grid=21,
        hidden=6,
        mg_steps=2,
        qr_gn_steps=0,
        adam_warmup_steps=1,
        dense_n_val=41,
        y_max=5.0,
        n_scales=2,
        n_gamma_multiples=1,
        d2_weight=0.0,
        resample_every=0,
        seed=3,
        train_hilbert="wholeline_hp",
        proj_defect_weight=0.0,
        hilbert_n_aux=32,
        hilbert_n_quad=16,
        hilbert_y_near=1.0,
        omega_peak_floor=0.02,
        nontrivial_weight=5.0,
    )
    result = cvn.run_ccf_vorticity_neural_discovery(cfg)
    assert result.extra["train_hilbert"] == "wholeline_hp"
    assert np.isfinite(result.diagnostics["reproduction_dense_max_abs"])
    assert np.isfinite(result.diagnostics["max_abs_vorticity_residual"])


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
    assert np.isfinite(tick["diagnosis"]["conjugate_orders_to_stretch"])
    assert tick["conjugate"]["stretch_1e-13_cleared"] is False
    assert tick["diagnosis"]["conjugate_orders_to_stretch"] == pytest.approx(
        tick["conjugate"]["orders_to_stretch"]
    )


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


def test_max_order_zero_projection_matches_legacy_planted_atom() -> None:
    """``max_order=0`` recovers a planted Q atom to ~1e-12 (legacy path)."""
    from omnibias.pinn.torch.equations.ccf_compactified import hardy_odd

    y = torch.linspace(-8.0, 8.0, 161, dtype=torch.float64)
    a, g = 1.3, 1.0 / (1.0 + 0.6057)
    omega = 0.4 * hardy_odd(y, a, g)
    scales = np.array([a], dtype=float)
    gammas = np.array([g], dtype=float)
    c0, d0, f0 = cvn.project_omega_hardy(y, omega, scales=scales, gammas=gammas)
    c1, d1, f1 = cvn.project_omega_hardy(
        y,
        omega,
        scales=scales,
        gammas=gammas,
        orders=np.array([0], dtype=int),
        parities=np.array([1], dtype=int),
    )
    assert d0 < 1e-12
    assert d1 < 1e-12
    np.testing.assert_allclose(c0, c1, atol=1e-12)
    np.testing.assert_allclose(f0["H"], f1["H"], atol=1e-12)
    np.testing.assert_allclose(f0["U"], f1["U"], atol=1e-12)


def test_max_order_recovers_planted_q2_atom() -> None:
    """Spatially odd ``Q^{(2)}`` is in the N=2 span and projects to ~1e-10."""
    from omnibias.core.conjugate import hardy_q_deriv_n

    y = np.linspace(-8.0, 8.0, 201)
    a, g = 1.3, 1.0 / (1.0 + 0.6057)
    omega = np.asarray([hardy_q_deriv_n(float(yy), a, g, 2) for yy in y], dtype=float)
    scales, gammas, orders, parities = cvn.hardy_dictionary_ordered(
        lam=0.6057, n_scales=1, n_gamma_multiples=1, max_order=2
    )
    coeffs, defect, fields = cvn.project_omega_hardy(
        y, omega, scales=scales, gammas=gammas, orders=orders, parities=parities
    )
    assert defect < 1e-10
    assert fields["H"].shape == y.shape
    assert coeffs.shape[0] == scales.shape[0]
    # The Q^{(2)} column (order 2, odd) should carry the mass.
    hit = np.where((orders == 2) & (parities == 1))[0]
    assert hit.size == 1
    assert abs(float(coeffs[hit[0]])) > 0.5
