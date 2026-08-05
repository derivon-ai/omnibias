# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Offline smoke tests for the binary-vs-STE benchmark (synthetic data, CPU)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from omnibias.binary.torch import ops as q  # noqa: E402

from examples.binary_vs_ste.arms import ARMS, Arm, get_arm  # noqa: E402
from examples.binary_vs_ste.data import synthetic_datasets  # noqa: E402
from examples.binary_vs_ste.experiment import (  # noqa: E402
    format_table,
    run_sweep,
    summarize,
)
from examples.binary_vs_ste.kernels import KERNELS, binarize_kernel  # noqa: E402
from examples.binary_vs_ste.models import BinaryLinear, QuantCtx  # noqa: E402
from examples.binary_vs_ste.ste import binarize_ste  # noqa: E402
from examples.binary_vs_ste.train import train_arm  # noqa: E402


def test_arm_registry_has_expected_arms() -> None:
    for name in ("ste", "omnibias_b10", "omnibias_b1", "tanh", "cauchy", "anneal", "learnable_beta"):
        assert name in ARMS
    for name in ("curvature", "scaled", "scaled_anneal"):  # the per-tensor / jet-STE levers
        assert name in ARMS
    with pytest.raises(ValueError):
        get_arm("does-not-exist")


def test_ste_forward_is_hard_sign_and_backward_is_clipped_identity() -> None:
    z = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    out = binarize_ste(z)
    assert torch.equal(out, torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))
    out.sum().backward()
    assert torch.equal(z.grad, torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0]))


@pytest.mark.parametrize("kernel", list(KERNELS))
def test_every_kernel_forward_is_hard_sign(kernel: str) -> None:
    z = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0], requires_grad=True)
    out = binarize_kernel(z, 1.0, kernel=kernel, normalize="peak")
    assert torch.equal(out, torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))
    out.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_box_kernel_equals_ste() -> None:
    """The compact box kernel at ``beta=1`` *is* the STE -- ties the menu to the baseline."""
    z = torch.linspace(-3, 3, 25, requires_grad=True)
    gk = torch.autograd.grad(binarize_kernel(z, 1.0, kernel="box").sum(), z)[0]
    z2 = z.detach().clone().requires_grad_(True)
    gs = torch.autograd.grad(binarize_ste(z2).sum(), z2)[0]
    assert torch.allclose(gk, gs)


def test_tanh_exact_kernel_matches_library_binarize() -> None:
    """``kernel='tanh', normalize='exact'`` reproduces the shipped ``binarize`` backward."""
    for beta in (1.0, 5.0, 10.0):
        z = torch.linspace(-2, 2, 41, requires_grad=True)
        gk = torch.autograd.grad(
            binarize_kernel(z, beta, kernel="tanh", normalize="exact").sum(), z
        )[0]
        z2 = z.detach().clone().requires_grad_(True)
        gl = torch.autograd.grad(q.binarize(z2, beta=beta).sum(), z2)[0]
        assert torch.allclose(gk, gl, atol=1e-6)


def test_heavy_tail_kernels_keep_far_units_alive() -> None:
    """Far from the boundary the box (STE) gradient is dead; cauchy stays largest."""
    z = torch.full((1,), 5.0, requires_grad=True)
    grads = {}
    for kernel in ("box", "gaussian", "tanh", "cauchy"):
        zc = z.detach().clone().requires_grad_(True)
        grads[kernel] = float(
            torch.autograd.grad(binarize_kernel(zc, 1.0, kernel=kernel).sum(), zc)[0]
        )
    assert grads["box"] == 0.0
    assert grads["cauchy"] > grads["tanh"] > 0.0
    assert grads["cauchy"] > grads["gaussian"] > 0.0


def test_peak_normalised_kernels_have_unit_height_at_zero() -> None:
    z = torch.zeros(1, requires_grad=True)
    for kernel in KERNELS:
        zc = z.detach().clone().requires_grad_(True)
        g = float(torch.autograd.grad(binarize_kernel(zc, 1.0, kernel=kernel).sum(), zc)[0])
        assert g == pytest.approx(1.0)


def test_learnable_beta_kernel_propagates_a_beta_gradient() -> None:
    z = torch.linspace(-2, 2, 17)
    beta = torch.tensor(1.5, requires_grad=True)
    out = binarize_kernel(z, beta, kernel="tanh", normalize="peak")
    out.sum().backward()
    assert beta.grad is not None and torch.isfinite(beta.grad)


def test_invalid_arm_configs_are_rejected() -> None:
    with pytest.raises(ValueError):
        Arm("x", "nope", "tanh", "peak", "fixed", 1.0, 3.0, "")
    with pytest.raises(ValueError):
        Arm("x", "ste", "box", "peak", "anneal", 1.0, 3.0, "")
    with pytest.raises(ValueError):
        Arm("x", "kernel", "tanh", "peak", "anneal", 1.0, 0.0, "")
    with pytest.raises(ValueError):
        Arm("x", "kernel", "not-a-kernel", "peak", "fixed", 1.0, 3.0, "")


@pytest.mark.parametrize("arm_name", list(ARMS))
def test_each_arm_trains_end_to_end_on_synthetic_mnist(arm_name: str) -> None:
    train_ds, test_ds, spec = synthetic_datasets("mnist", n_train=256, n_test=128, seed=0)
    result = train_arm(
        get_arm(arm_name),
        train_ds,
        test_ds,
        spec,
        epochs=2,
        batch_size=64,
        lr=2e-3,
        device="cpu",
        seed=0,
    )
    assert 0.0 <= result.test_acc <= 1.0
    assert result.best_acc >= result.test_acc - 1e-9
    assert result.final_train_loss == result.final_train_loss  # not NaN
    assert result.final_beta > 0.0


def test_tanh_arm_reduces_loss_and_beats_chance() -> None:
    train_ds, test_ds, spec = synthetic_datasets("mnist", n_train=512, n_test=256, seed=0)
    result = train_arm(
        get_arm("tanh"),
        train_ds,
        test_ds,
        spec,
        epochs=4,
        batch_size=64,
        lr=2e-3,
        device="cpu",
        seed=0,
    )
    assert result.final_train_loss < result.init_train_loss
    assert result.best_acc > 1.0 / spec.num_classes


def test_learnable_beta_moves_off_its_initial_value() -> None:
    train_ds, test_ds, spec = synthetic_datasets("mnist", n_train=256, n_test=128, seed=0)
    arm = get_arm("learnable_beta")
    result = train_arm(
        arm, train_ds, test_ds, spec, epochs=3, batch_size=64, lr=5e-3, device="cpu", seed=0
    )
    assert abs(result.final_beta - arm.beta) > 1e-4


def test_conv_path_runs_on_synthetic_cifar10() -> None:
    train_ds, test_ds, spec = synthetic_datasets("cifar10", n_train=128, n_test=64, seed=0)
    result = train_arm(
        get_arm("cauchy"),
        train_ds,
        test_ds,
        spec,
        epochs=1,
        batch_size=64,
        lr=2e-3,
        device="cpu",
        seed=0,
    )
    assert 0.0 <= result.test_acc <= 1.0


def test_curvature_arm_backward_is_the_jet_ste_slope() -> None:
    """The curvature arm's backward equals ``s'(z)+(h^2/6)s'''(z)`` (a hard-sign forward)."""
    qfn = get_arm("curvature").make_quantizer(1.0)
    z = torch.linspace(-2.0, 2.0, 41, requires_grad=True)
    out = qfn(z)
    assert torch.equal(out, torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z)))
    grad = torch.autograd.grad(out.sum(), z)[0]
    expected = q.curvature_corrected_slope(z.detach(), 1.0)
    assert torch.allclose(grad, expected, atol=1e-6)


def test_scaled_arm_is_invariant_to_input_rescaling() -> None:
    """The scale-aware surrogate gives the *same* per-element gradient under z -> c z."""
    qfn = get_arm("scaled").make_quantizer(1.0)
    base = torch.randn(256)
    z = base.clone().requires_grad_(True)
    z5 = (5.0 * base).clone().requires_grad_(True)
    g1 = torch.autograd.grad(qfn(z).sum(), z)[0]
    g5 = torch.autograd.grad(qfn(z5).sum(), z5)[0]
    assert torch.allclose(g1, g5, atol=1e-5)
    # and a plain (non-scale-aware) tanh arm is NOT rescaling-invariant
    plain = get_arm("tanh").make_quantizer(1.0)
    zp = base.clone().requires_grad_(True)
    z5p = (5.0 * base).clone().requires_grad_(True)
    gp = torch.autograd.grad(plain(zp).sum(), zp)[0]
    g5p = torch.autograd.grad(plain(z5p).sum(), z5p)[0]
    assert not torch.allclose(gp, g5p, atol=1e-3)


def test_scaled_learnable_would_be_rejected_but_scaled_anneal_runs() -> None:
    with pytest.raises(ValueError):
        Arm("x", "curvature", "tanh", "peak", "learnable", 1.0, 3.0, "")
    with pytest.raises(ValueError):
        Arm("x", "ste", "box", "peak", "fixed", 1.0, 3.0, "", scale_aware=True)


def test_xnor_scale_rescales_each_output_filter() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    ctx = QuantCtx(fn=binarize_ste, xnor=False)
    lin = BinaryLinear(6, 5, ctx, bias=False)
    out_plain = lin(x)
    ctx.xnor = True
    out_xnor = lin(x)
    assert not torch.allclose(out_plain, out_xnor)
    alpha = lin.weight.abs().mean(dim=1)  # per output filter
    assert torch.allclose(out_xnor, out_plain * alpha.unsqueeze(0), atol=1e-5)


def test_xnor_and_cosine_lr_train_end_to_end() -> None:
    train_ds, test_ds, spec = synthetic_datasets("mnist", n_train=256, n_test=128, seed=0)
    result = train_arm(
        get_arm("tanh"),
        train_ds,
        test_ds,
        spec,
        epochs=2,
        batch_size=64,
        lr=2e-3,
        device="cpu",
        seed=0,
        xnor=True,
        lr_schedule="cosine",
    )
    assert 0.0 <= result.test_acc <= 1.0
    assert result.final_train_loss == result.final_train_loss  # not NaN


def test_invalid_lr_schedule_is_rejected() -> None:
    train_ds, test_ds, spec = synthetic_datasets("mnist", n_train=64, n_test=32, seed=0)
    with pytest.raises(ValueError):
        train_arm(
            get_arm("ste"),
            train_ds,
            test_ds,
            spec,
            epochs=1,
            batch_size=32,
            device="cpu",
            lr_schedule="nope",
        )


def test_run_sweep_and_table_offline() -> None:
    results = run_sweep(
        datasets=("mnist", "fashion_mnist"),
        arms=("ste", "tanh", "cauchy"),
        seeds=(0,),
        epochs=1,
        batch_size=64,
        synthetic=True,
        device="cpu",
    )
    assert len(results) == 6
    summary = summarize(results)
    assert ("mnist", "ste") in summary
    assert ("fashion_mnist", "cauchy") in summary
    assert summary[("mnist", "ste")]["mean"] == summary[("mnist", "ste")]["best_mean"]
    best = format_table(results, metric="best")
    final = format_table(results, metric="final")
    assert "best-epoch" in best and "cauchy" in best
    assert "final-epoch" in final
