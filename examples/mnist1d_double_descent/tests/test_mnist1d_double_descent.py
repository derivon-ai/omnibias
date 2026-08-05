# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Offline, CPU, fast smoke tests for the MNIST-1D double-descent harness.

Everything uses the tiny synthetic / vendored data (no download, no GPU): data shapes +
label noise, model construction, the optimizer-arm catalogue + construction, the exact
curvature snapshot (dense vs matrix-free agreement), one training step per arm in its
register, the sweep + aggregation writeout, certified read-outs, plotting, and the cluster
job generator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.mnist1d_double_descent.arms import (
    ARMS,
    CORE_ARMS,
    SHARPNESS_ARMS,
    arms_for_register,
    get_arm,
)
from examples.mnist1d_double_descent.curvature import (
    dense_vs_matrix_free_gap,
    exact_sharpness_gap,
    spectrum_snapshot,
)
from examples.mnist1d_double_descent.data import (
    Mnist1DConfig,
    add_label_noise,
    generate_mnist1d,
    load_mnist1d,
    one_hot,
    synthetic_mnist1d,
)
from examples.mnist1d_double_descent.models import (
    MLP1D,
    REGISTERS,
    build_model,
    count_parameters,
    register_activation,
)
from examples.mnist1d_double_descent.train import RunConfig, canonical_loss, train_run


def _bundle(noise: float = 0.1):
    return synthetic_mnist1d(n_train=128, n_test=64, label_noise=noise, seed=0)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def test_synthetic_shapes_and_onehot() -> None:
    b = _bundle()
    assert b.x_train.shape == (128, 40)
    assert b.x_test.shape == (64, 40)
    assert int(b.y_train.min()) >= 0 and int(b.y_train.max()) < 10
    oh = one_hot(b.y_train, b.num_classes)
    assert oh.shape == (128, 10)
    assert torch.allclose(oh.sum(dim=1), torch.ones(128))
    assert (oh.argmax(dim=1) == b.y_train).all()


def test_label_noise_deterministic_and_train_only() -> None:
    y = torch.arange(200) % 10
    y1, m1 = add_label_noise(y, 0.25, num_classes=10, seed=3)
    y2, m2 = add_label_noise(y, 0.25, num_classes=10, seed=3)
    assert torch.equal(m1, m2) and torch.equal(y1, y2)
    assert abs(float(m1.float().mean()) - 0.25) < 0.05
    assert (y1[m1] != y[m1]).all()  # flipped labels really differ
    y0, m0 = add_label_noise(y, 0.0, num_classes=10, seed=3)
    assert int(m0.sum()) == 0 and torch.equal(y0, y)


def test_vendored_generator_and_loader() -> None:
    cfg = Mnist1DConfig(n_train=50, n_test=20, seed=1)
    x_tr, y_tr, x_te, y_te = generate_mnist1d(cfg)
    assert x_tr.shape == (50, 40) and x_te.shape == (20, 40)
    assert np.isfinite(x_tr).all()
    bundle = load_mnist1d(
        Mnist1DConfig(n_train=40, n_test=20), label_noise=0.1, noise_seed=0, allow_pip=False
    )
    assert bundle.source == "vendored" and bundle.in_dim == 40


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("register", REGISTERS)
def test_model_build_and_param_count(register: str) -> None:
    model = build_model(register, in_dim=40, hidden=7, num_classes=10, seed=0)
    assert count_parameters(model) == (40 * 7 + 7) + (7 * 10 + 10)
    assert model(torch.zeros(5, 40)).shape == (5, 10)
    assert register_activation(register) in ("relu", "tanh")


def test_model_is_verify_ingestible() -> None:
    verify_torch = pytest.importorskip("omnibias.verify.torch")
    model = build_model("mse_tanh", in_dim=40, hidden=4, num_classes=10, seed=0)
    assert isinstance(model, MLP1D)
    net = verify_torch.network_from_sequential(model.net)
    assert net is not None


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def test_arms_registry_and_groups() -> None:
    assert len(ARMS) >= 12
    assert set(CORE_ARMS) <= set(ARMS)
    assert set(SHARPNESS_ARMS) <= set(ARMS)
    ce = arms_for_register("ce_relu", ARMS)
    mse = arms_for_register("mse_tanh", ARMS)
    for gn in ("cubic_gauss_newton", "diag_gn", "natural_gradient"):
        assert gn not in ce and gn in mse
    with pytest.raises(ValueError):
        get_arm("nope")


@pytest.mark.parametrize("arm_name", ARMS)
def test_arm_builds_optimizer(arm_name: str) -> None:
    arm = get_arm(arm_name)
    register = "mse_tanh" if "mse_tanh" in arm.registers else arm.registers[0]
    model = build_model(register, in_dim=40, hidden=4, num_classes=10, seed=0)
    opt = arm.build(model, lr=arm.lr)
    assert isinstance(opt, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# curvature
# ---------------------------------------------------------------------------


def test_spectrum_snapshot_dense_and_matrix_free_agree() -> None:
    b = _bundle()
    model = build_model("mse_tanh", in_dim=40, hidden=5, num_classes=10, seed=0, dtype=torch.float64)
    x = b.x_train.to(torch.float64)
    target = one_hot(b.y_train, b.num_classes).to(torch.float64)
    params = list(model.parameters())

    snap = spectrum_snapshot(((model(x) - target) ** 2).mean(), params, dense_max_params=10_000)
    assert snap.method == "dense"
    assert math.isfinite(snap.lambda_max) and math.isfinite(snap.trace)
    assert snap.lambda_max >= snap.lambda_min

    gap = dense_vs_matrix_free_gap(((model(x) - target) ** 2).mean(), params, power_iters=120)
    assert gap < 0.2
    sharp = exact_sharpness_gap(((model(x) - target) ** 2).mean(), params)
    assert math.isfinite(sharp) and sharp >= 0.0


# ---------------------------------------------------------------------------
# training: every arm takes a step in its register
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm_name", ARMS)
def test_each_arm_takes_a_step(arm_name: str) -> None:
    arm = get_arm(arm_name)
    register = "mse_tanh" if "mse_tanh" in arm.registers else arm.registers[0]
    b = _bundle(noise=0.1)
    cfg = RunConfig(
        register=register, arm=arm_name, width=4, depth=1, seed=0, label_noise=0.1,
        steps=2, lr=arm.lr, batch_size=None, curvature=False, device="cpu",
    )
    result = train_run(b, arm, cfg)
    assert result.n_params > 0
    assert len(result.history) == cfg.steps + 1
    assert 0.0 <= result.final_test_err <= 1.0
    assert math.isfinite(result.final_train_loss)


def test_train_run_records_curvature_snapshot() -> None:
    """A tiny run with curvature enabled attaches a finite snapshot to the history (H1 wiring)."""
    arm = get_arm("adam")
    b = _bundle(noise=0.1)
    cfg = RunConfig(
        register="mse_tanh", arm="adam", width=3, depth=1, seed=0, label_noise=0.1,
        steps=2, lr=arm.lr, batch_size=None, curvature=True, curvature_every=100,
        dense_max_params=5000, device="cpu",
    )
    result = train_run(b, arm, cfg)
    snaps = [h["curvature"] for h in result.history if isinstance(h.get("curvature"), dict)]
    assert snaps, "expected at least one curvature snapshot"
    assert math.isfinite(snaps[0]["lambda_max"])
    assert snaps[0]["lambda_max"] >= snaps[0]["lambda_min"]


def test_phase1_arms_memory_and_time_telemetry() -> None:
    """The Phase-1 arms train, populate the new memory/time telemetry, and FrugalCurvature's
    optimiser state is strictly smaller than Adam's two O(P) buffers."""
    b = _bundle(noise=0.1)

    def _run(arm_name: str) -> object:
        arm = get_arm(arm_name)
        cfg = RunConfig(
            register="mse_tanh", arm=arm_name, width=8, depth=1, seed=0, label_noise=0.1,
            steps=3, lr=arm.lr, batch_size=None, curvature=False, device="cpu",
        )
        return train_run(b, arm, cfg)

    adam = _run("adam")
    frugal_hutch = _run("frugal_hutch")
    frugal_gn = _run("frugal_gn")
    exact_sam = _run("exact_sam")

    for r in (adam, frugal_hutch, frugal_gn, exact_sam):
        assert r.status == "ok"
        assert r.opt_state_bytes > 0  # the memory telemetry is populated
        assert r.update_time_s > 0.0  # the wall-clock telemetry is populated
    # Adam keeps two O(P) buffers; FrugalCurvature keeps one + O(#tensors) per-tensor scalars.
    assert frugal_hutch.opt_state_bytes < adam.opt_state_bytes
    assert frugal_gn.opt_state_bytes < adam.opt_state_bytes


def test_canonical_loss_matches_register() -> None:
    b = _bundle()
    model = build_model("ce_relu", in_dim=40, hidden=4, num_classes=10, seed=0)
    ob = one_hot(b.y_train, b.num_classes)
    ce = canonical_loss(model, b.x_train, b.y_train, ob, "ce_relu")
    assert ce.ndim == 0 and float(ce.detach()) > 0.0


# ---------------------------------------------------------------------------
# edge-of-stability controller
# ---------------------------------------------------------------------------


def test_eos_controller_rate_on_known_quadratic() -> None:
    """On loss 0.5*sum(a_i w_i^2) (Hessian diag(a)) the controller sets eta = c*2/lambda_max."""
    from examples.mnist1d_double_descent.eos import EdgeOfStabilityLR

    a = torch.tensor([3.0, 9.0, 1.0], dtype=torch.float64)
    w = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    ctrl = EdgeOfStabilityLR(
        c=0.9, eta_min=1e-6, eta_max=100.0, probe_iters=80, measure_every=1, ema=0.0, seed=0
    )
    tel = ctrl.rate(0.5 * (a * w * w).sum(), [w])
    assert abs(tel["eos_lambda_max"] - 9.0) < 0.1  # power iteration -> top eigenvalue = max(a)
    assert abs(tel["eos_eta"] - 0.9 * 2.0 / tel["eos_lambda_max"]) < 1e-9  # eta = c*2/lambda_max
    assert abs(tel["eos_lambda_eta"] - 1.8) < 0.05  # product pinned at 2c

    # momentum widens the linear-stability limit to 2(1+beta)/lambda_max
    w2 = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    ctrl_m = EdgeOfStabilityLR(
        c=0.9, momentum=0.9, eta_min=1e-6, eta_max=100.0, probe_iters=80, ema=0.0, seed=0
    )
    assert abs(ctrl_m.target - 2.0 * 0.9 * 1.9) < 1e-12  # target = 2c(1+beta) = 3.42
    tel_m = ctrl_m.rate(0.5 * (a * w2 * w2).sum(), [w2])
    assert abs(tel_m["eos_lambda_eta"] - 3.42) < 0.06  # product pinned at 2c(1+beta)


def test_eos_arm_records_and_holds_the_edge() -> None:
    """The eos arm logs eta/lambda_max telemetry and pins lambda_max*eta at the edge (2c)."""
    arm = get_arm("eos")
    c = float(arm.hypers["c"])
    b = _bundle(noise=0.35)
    cfg = RunConfig(
        register="ce_relu", arm="eos", width=24, depth=1, seed=0, label_noise=0.35,
        steps=30, lr=arm.lr, batch_size=None, curvature=False, log_every=2, device="cpu",
    )
    result = train_run(b, arm, cfg)
    tel = [h for h in result.history if "eos_lambda_eta" in h]
    assert tel, "expected eos telemetry in the history"
    products = [float(h["eos_lambda_eta"]) for h in tel]
    assert all(p <= 2.0 * c + 1e-3 for p in products)  # never past the edge target
    assert all(float(h["eos_eta"]) > 0.0 for h in tel)
    assert max(products) >= 2.0 * c - 0.3  # the edge is actually engaged as the loss sharpens


# ---------------------------------------------------------------------------
# sweep + aggregation
# ---------------------------------------------------------------------------


def test_run_sweep_and_write_summary(tmp_path: Path) -> None:
    from examples.mnist1d_double_descent.experiment import run_sweep, write_summary

    results = run_sweep(
        registers=("ce_relu",),
        arm_names=("adam", "cubic_newton"),
        widths=(2, 6),
        seeds=(0, 1),
        noise_levels=(0.15,),
        steps=2,
        device="cpu",
        scratch_dir=tmp_path,
        curvature=False,
        synthetic=True,
    )
    assert len(results) == 2 * 2 * 2
    assert all(r.status == "ok" for r in results), [r.error for r in results if r.status != "ok"]
    runs_csv, summary_csv, summary_json = write_summary(tmp_path, tmp_path / "agg")
    assert runs_csv.exists() and summary_csv.exists() and summary_json.exists()
    assert json.loads(summary_json.read_text())


def test_collect_runs_recurses_sweep_subdirs(tmp_path: Path) -> None:
    """The cluster writes each job into ``<scratch>/<tag>/``; aggregation must recurse into them."""
    from examples.mnist1d_double_descent.experiment import collect_runs, run_sweep

    # One flat run at the root and one nested under a per-tag subdirectory.
    run_sweep(
        registers=("ce_relu",), arm_names=("adam",), widths=(3,), seeds=(0,),
        noise_levels=(0.15,), steps=2, device="cpu", scratch_dir=tmp_path,
        curvature=False, synthetic=True,
    )
    nested = tmp_path / "ce_relu__adam__noise0.15__seed1__wb0"
    run_sweep(
        registers=("ce_relu",), arm_names=("adam",), widths=(3,), seeds=(1,),
        noise_levels=(0.15,), steps=2, device="cpu", scratch_dir=nested,
        curvature=False, synthetic=True,
    )
    # A non-run JSON in the tree must be ignored (no ``config`` key).
    (tmp_path / "note.json").write_text('{"hello": "world"}', encoding="utf-8")
    rows = collect_runs(tmp_path)
    assert len(rows) == 2, rows
    assert {int(r["seed"]) for r in rows} == {0, 1}


# ---------------------------------------------------------------------------
# certified read-outs (P4)
# ---------------------------------------------------------------------------


def test_certify_smoke() -> None:
    pytest.importorskip("omnibias.verify")
    from examples.mnist1d_double_descent.certify import train_and_certify

    readout = train_and_certify(
        width=3, bundle=_bundle(), seed=0, steps=15, eps=0.02, n_points=1,
        max_boxes=1, order=1,
    )
    assert math.isfinite(readout.lipschitz_inf) and readout.lipschitz_inf >= 0.0
    assert 0.0 <= readout.robust_frac <= 1.0


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def test_make_all_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from examples.mnist1d_double_descent.analysis.plots import make_all

    rows = []
    for register in ("ce_relu", "mse_tanh"):
        for arm in ("adam", "jet_subspace_o2", "jet_subspace_o3"):
            for width in (4, 16, 64):
                rows.append(
                    {
                        "register": register,
                        "arm": arm,
                        "width": width,
                        "label_noise": 0.15,
                        "test_err_mean": 0.5 - 0.001 * width,
                        "test_err_std": 0.02,
                        "lambda_max_mean": 1.0 + width,
                    }
                )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(rows))
    figures = make_all(summary, tmp_path / "figures")
    assert figures
    for fig in figures:
        assert fig.exists()


# ---------------------------------------------------------------------------
# cluster job generation
# ---------------------------------------------------------------------------


def test_gen_jobs_skips_invalid_combos() -> None:
    from examples.mnist1d_double_descent.sweep.gen_jobs import _parse_args, generate

    args = _parse_args(
        [
            "--registers", "ce_relu", "mse_tanh",
            "--arms", "adam", "cubic_gauss_newton",
            "--widths", "40", "50",
            "--seeds", "0",
            "--noises", "0.15",
            "--width-block", "2",
            "--scratch-base", "/tmp/x",
        ]
    )
    cmds = generate(args)
    assert cmds
    for cmd in cmds:
        if "--arms cubic_gauss_newton" in cmd:
            assert "--register mse_tanh" in cmd
