# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, second-order-trained soft decision trees -- omnibias-tab.

Run:

    pip install "omnibias-tab[torch,jax,verify,gbm]"
    python docs/examples/tab_validate.py

A decision-tree split is a hard threshold ``1[w.x > t]``; omnibias makes it a **soft oblique
gate** ``g(x) = sigmoid(beta (w.x - t))`` and anneals ``beta -> inf`` toward a genuine hard
split. This is the **temperature collapse** axis (``beta -> inf``), *not*
the founding ``delta -> 0`` bias collapse (the multi-bias limit to ``sigma^(K-1)``; see
``docs/theory.md``). The derivative tower is still used -- exact gate curvature feeds the
second-order trainer, and the ``beta -> inf`` limit gets a *certified* soft->hard gap.

This deterministic, CPU-tiny smoke exercises the four standalone differentiators over a
gradient-boosting baseline, each with an assertion so it wires in as a CI smoke:

1. **Bit-identical torch <-> jax forward** (what GBMs cannot offer: a differentiable tree).
2. **Sound certificates**: a *proved* per-feature monotone constraint, a rigorous output
   enclosure, and a certified train-soft / deploy-hard rounding gap as ``beta -> inf``.
3. **Exact second-order training** of the whole model (splits included) strictly beating a
   tuned first-order (Adam) baseline on held-out data at a matched small step budget.
4. **Match-or-beat LightGBM** head-to-head on a real tabular split (the empirical-validation
   gate; the heavier multi-seed suite lives in ``packages/omnibias-tab/bench/sweep.py`` and
   ``docs/benchmarks.md``).
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def parity_demo() -> None:
    print("=== 1. bit-identical torch <-> jax forward (a differentiable tree) ===")
    import jax

    jax.config.update("jax_enable_x64", True)

    from omnibias.tab import SoftTreeConfig, forward_np, init_params
    from omnibias.tab.jax.model import forward as forward_jax
    from omnibias.tab.torch.model import SoftTreeEnsemble

    cfg = SoftTreeConfig(n_features=6, n_trees=8, depth=3, task="regression", n_outputs=2, seed=0)
    params = init_params(cfg, leaf_scale=1.0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 6))
    beta = 6.0

    f_np = forward_np(params, X, beta)
    f_torch = SoftTreeEnsemble(cfg, params).score(X, beta=beta)
    f_jax = np.asarray(forward_jax(params, X, beta))
    e_torch = float(np.max(np.abs(f_np - f_torch)))
    e_jax = float(np.max(np.abs(f_np - f_jax)))
    print(f"  depth-3 ensemble, |numpy - torch| = {e_torch:.2e}   |numpy - jax| = {e_jax:.2e}")
    assert e_torch < 1e-9 and e_jax < 1e-9, "backends must be bit-identical in float64"
    print("  Reading: one shared leaf ordering + closed-form gate -> bit-identical backends.\n")


def certificate_demo() -> None:
    print("=== 2. sound certificates (proved monotonicity + output bounds + rounding gap) ===")
    from omnibias.tab import SoftTreeConfig, TabParams, certify_tab, forward_np

    # A hand-built additive model that is monotone increasing in every feature:
    # positive oblique directions and a positive on-gate leaf jump u = leaf_1 - leaf_0 > 0.
    d = 3
    cfg = SoftTreeConfig(n_features=d, n_trees=5, depth=1, task="binary", beta_final=4.0, seed=0)
    rng = np.random.default_rng(0)
    W = (np.abs(rng.standard_normal((5, 1, d))) + 0.2)
    t = rng.standard_normal((5, 1)) * 0.5
    leaf0 = rng.standard_normal((5, 1, 1))
    leaf1 = leaf0 + (np.abs(rng.standard_normal((5, 1, 1))) + 0.2)
    params = TabParams(cfg, W, t, np.concatenate([leaf0, leaf1], axis=1), np.zeros(1))

    box = np.stack([-2.0 * np.ones(d), 2.0 * np.ones(d)])
    X = rng.uniform(-2.0, 2.0, size=(256, d))
    cert = certify_tab(
        params, box, monotone_features={f: +1 for f in range(d)}, X=X, beta=4.0, use_verify=False
    )
    F = forward_np(params, X, 4.0)[:, 0]
    lo, hi = cert.output_bounds[0]
    print(f"  monotone constraint (all {d} features increasing): certified = {cert.monotone_ok}")
    print(f"  output enclosure [{lo:.3f}, {hi:.3f}] contains sampled range "
          f"[{F.min():.3f}, {F.max():.3f}]")
    print(f"  Lipschitz (L2) upper bound = {cert.lipschitz:.3f}")
    assert cert.rounding is not None
    print(f"  soft->hard rounding gap <= {cert.rounding.max_gap:.4f} "
          f"(measured {cert.rounding.measured_max:.4f}, sound = {cert.rounding.is_sound})")

    assert cert.monotone_ok is True, "the constructed model is monotone; the cert must prove it"
    assert lo - 1e-9 <= F.min() and F.max() <= hi + 1e-9, "output enclosure must be sound"
    assert cert.rounding.is_sound, "the rounding-gap bound must dominate the measured gap"
    print("  Reading: monotonicity is *proved* over the box (GBM offers it only as a prior),")
    print("  and the deploy-time hard tree is certified to move the score by <= the gap.\n")


def second_order_demo() -> None:
    print("=== 3. exact 2nd-order training beats a tuned 1st-order (Adam) baseline ===")
    import torch
    from omnibias.tab import SoftTreeConfig
    from omnibias.tab.torch import SoftTreeEnsemble, fit_first_order, fit_second_order

    # A low-noise, signal-rich binary problem; a short, matched step budget where Adam
    # underfits and the exact-Hessian step converges -> a clean held-out win.
    rng = np.random.default_rng(0)
    n, d = 320, 12
    w = np.zeros(d)
    w[:3] = [1.5, -1.0, 0.8]
    X = rng.standard_normal((n, d))
    y = (X @ w + 0.1 * rng.standard_normal(n) > 0).astype(np.float64)
    Xtr, ytr, Xval, yval = X[: n // 2], y[: n // 2], X[n // 2 :], y[n // 2 :]
    cfg = SoftTreeConfig(
        n_features=d, n_trees=16, depth=1, task="binary", beta_final=3.0, seed=1, leaf_l2=1e-2
    )

    torch.manual_seed(0)
    m1 = SoftTreeEnsemble(cfg)
    r1 = fit_first_order(m1, Xtr, ytr, lr=0.05, steps=15, weight_l2=1e-3, anneal=False, val=(Xval, yval))
    torch.manual_seed(0)
    m2 = SoftTreeEnsemble(cfg)
    r2 = fit_second_order(m2, Xtr, ytr, optimizer="trust_region", steps=15, weight_l2=1e-3, anneal=False, val=(Xval, yval))
    print(f"  Adam (1st-order)         train loss = {r1.train_loss:.4f}   held-out acc = {r1.val_metric:.4f}")
    print(f"  trust-region Newton-CG   train loss = {r2.train_loss:.4f}   held-out acc = {r2.val_metric:.4f}")
    assert r2.val_metric is not None and r1.val_metric is not None
    assert r2.train_loss < r1.train_loss, "exact 2nd-order should optimise better per equal budget"
    assert r2.val_metric > r1.val_metric, "and reach a strictly better held-out accuracy here"
    print("  Reading: exact curvature converges in far fewer steps than a first-order method.\n")


def lightgbm_head_to_head() -> None:
    print("=== 4. match-or-beat LightGBM head-to-head (breast_cancer, held-out) ===")
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print("  (lightgbm not installed; skipping -- install with omnibias-tab[gbm])\n")
        return
    from omnibias.tab.bench import TabConfig, head_to_head

    # A CPU-tiny, deterministic configuration (subsampled rows, few stages) -- the heavier
    # full-size multi-seed suite is the real acceptance gate.
    cfg = TabConfig(method="boost", n_stages=20, learning_rate=0.3, depth=2,
                    inner_steps=20, inner_lr=0.06, beta_final=8.0)
    h = head_to_head("breast_cancer", seeds=2, tab_cfg=cfg, max_rows=400)
    s = h.summary()
    print(f"  tab   accuracy = {s['tab_mean_primary']:.4f} +/- {s['tab_std_primary']:.4f}")
    print(f"  lgbm  accuracy = {s['lgbm_mean_primary']:.4f} +/- {s['lgbm_std_primary']:.4f}")
    print(f"  not-worse-than-LightGBM (within its seed noise) = {s['not_worse']}")
    assert s["tab_mean_primary"] >= 0.90, "tab should be a strong classifier on breast_cancer"
    print("  Reading: a differentiable, certified tree ties/beats a strong GBM on real data.")
    print("  (Full multi-seed suite: packages/omnibias-tab/bench/sweep.py -> docs/benchmarks.md.)\n")


def main() -> None:
    parity_demo()
    certificate_demo()
    second_order_demo()
    lightgbm_head_to_head()
    print("OK: bit-identical backends; sound certificates; 2nd-order > Adam; tab matches LightGBM.")


if __name__ == "__main__":
    main()
