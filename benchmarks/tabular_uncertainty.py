# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Theory 05-03: noise-damped exact-Newton, boosted heteroscedastic reweighting, and
band/integral soft-binning embeddings in the high-aleatoric-noise regime (Kartashev,
Rubachev & Babenko, arXiv:2509.04430).

Six gates, in order (``theory/05-applications/03-data-uncertainty-training-signal.md``
section 8):

* **G0 plumbing identity** -- ``fit_noise_aware(lam=0.0)`` is bit-identical to
  ``fit_second_order`` (exact equality, true by construction).
* **G1b Hessian correctness** -- numerically finite-differencing the penalized
  closure's Hessian (central differences, independent of the autodiff path
  used to *build* the penalty) matches the analytic ``H_task + lam *
  Sigma_noise`` to ``<= 1e-6`` relative error, for ``lam in {0.1, 1.0, 10.0}``.
* **G1 -- the falsifier (Phase 1).** On both ``saw_wave_2d`` and
  ``mlp_heteroscedastic_20d``, with ``lam`` selected on a validation split from
  the predeclared grid ``{0.1, 0.3, 1.0, 3.0, 10.0}`` (never on test): (1)
  reference validity (``Spearman(s_hat, s_true) >= 0.6`` on test), (2) skill
  (both arms beat the mean predictor on the *true*-noise top decile of test),
  (3) absolute (the selected ``lam>0`` arm's top-decile RMSE is ``<= 0.95x``
  the ``lam=0`` arm's). Worst-seed over >= 5 seeds in ``--full``.
  **G1 has run `--full` and failed on both datasets** -- Proposal A
  (``fit_noise_aware``) is a recorded negative result (spec section 1/8); G2-G5
  below are the independent Phase 2 mechanisms this does not affect.
* **G2 -- local target consistency (Phase 2).** On ``mlp_heteroscedastic_20d``
  (exact ``df/dx`` from the frozen generator MLP) and one public set (a frozen
  surrogate ``df/dx`` from a small fitted MLP, section 10's honestly-softer
  real-data case): a ``BandFeatureEmbedder`` trained by
  ``local_target_consistency_loss`` feeds a downstream head whose top-decile
  RMSE beats the same head on raw features by a predeclared ``5%``.
* **G3 -- boosted heteroscedastic win (Phase 2).** On both synthetic sets,
  ``fit_boosted_heteroscedastic(weighting="gls")`` beats
  ``fit_boosted_heteroscedastic(weighting="shrinkage")`` (bit-identical to
  plain ``fit_boosted`` -- a second G0-style check) on top-decile RMSE by
  ``5%``.
* **G4 -- public-suite honesty report (Phase 2).** On ``>= 6`` public
  regression sets, the full top-decile-RMSE win/loss table for the predeclared
  "best Phase 1+2 arm" (``fit_boosted_heteroscedastic(weighting="gls")`` --
  Proposal A is excluded per G1) against tuned LightGBM, CatBoost, RealMLP,
  and TabM. This is a **reporting-completeness** gate (did the full table get
  produced for every named baseline on ``>= 6`` datasets), not a win/loss
  claim -- matching 05-02 G3's explicit "no aggregate-only reporting" rule.
* **G5 -- no regression on the aggregate metric (Phase 2).** Across the same
  suite, the G4 omnibias arm is not worse than plain ``fit_second_order`` /
  ``fit_boosted`` on **overall** (not decile-conditioned) RMSE by more than
  that baseline's own across-seed noise.

**A falsifiable gate failing on either dataset is a valid, honestly-recorded
negative result** -- unlike ``benchmarks/_gates.require_all_seeds`` (which
raises), the per-dataset verdicts here never raise; failure is recorded in the
JSON and only turned into a nonzero exit code at the very end, so a falsified
hypothesis still produces a complete artifact instead of a stack trace.

Tiers: ``--full`` is the acceptance experiment (5 seeds, ``n=4000``,
``steps=80``, 32 trees, CatBoost 500 iterations for G0/G1b/G1 -- heavy enough
to run as a cluster job; ``--workers N`` fans the independent ``(dataset, seed)``
units of *every* sweep (G1, G2, G3, and G4/G5) across a process pool to cut
wall-clock, without changing a single reported number -- see
:func:`_run_g1_sweep` and :func:`_parallel_map`). Smoke (default) proves every code path end to end at
reduced scale (1-2 seeds, small ``n``, few steps/stages, LightGBM + CatBoost
baselines only -- RealMLP/TabM are ``--full``-only, matching the existing
convention that only LightGBM runs in CI while richer baselines are
``--full``-only) so it finishes in a few minutes on a login node; it always
computes and *records* every gate but only G0/G1b gate the smoke exit code,
matching ``tabular_arrangement.py``'s "smoke is a wiring gate" convention -- a
scientific hypothesis is not something a reduced budget gets to decide either
way, and G4/G5 are structurally unsatisfiable below the full ``>= 6``-dataset,
5-seed suite regardless of tier.

``s_hat`` is a frozen, train-only-fit, **model-based** estimate
(``GuaranteeKind.MODEL_BASED``); it is never a sound enclosure and never
touches ``certify_tab``. The ``beta`` gate anneal (and the embedder's own) is
temperature collapse (feasibility sense); no founding ``delta -> 0`` bias
collapse is invoked anywhere in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

SEEDS_FULL: tuple[int, ...] = (0, 1, 2, 3, 4)
SEEDS_SMOKE: tuple[int, ...] = (0, 1)
LAM_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
SPEARMAN_FLOOR = 0.6
TAU = 0.05  # G1 absolute: selected-lam top-decile RMSE <= (1 - TAU) * lam=0's
G1B_LAM_VALUES: tuple[float, ...] = (0.1, 1.0, 10.0)
G1B_REL_TOL = 1e-6

BETA_FINAL = 16.0
DATASETS: tuple[str, ...] = ("saw_wave_2d", "mlp_heteroscedastic_20d")

# n_trees / depth are tier-scoped (not just n / steps): a 32-tree, depth-2 model
# on the 20-D dataset has 1473 parameters, and a single fit_second_order(steps=20)
# call on it was measured at 227s on a contended CPU core -- fine for a cluster
# `--full` job, wrong for a login-node "does the code path run" check. Smoke keeps
# depth=2 (still exercises the multiplicative tier) but shrinks n_trees/steps/n so
# the whole G0+G1b+G1 sweep finishes in low single-digit minutes.
BUDGET: dict[str, dict[str, int]] = {
    "full": {"n": 4000, "steps": 80, "catboost_iterations": 500, "n_trees": 32, "depth": 2},
    "smoke": {"n": 250, "steps": 6, "catboost_iterations": 40, "n_trees": 8, "depth": 2},
}

# --------------------------------------------------------------------------- #
# Phase 2 (G2-G5) constants.                                                 #
# --------------------------------------------------------------------------- #

TAU_G2 = 0.05  # G2 absolute: embed arm's top-decile RMSE <= (1 - TAU_G2) * raw arm's
TAU_G3 = 0.05  # G3 absolute: gls arm's top-decile RMSE <= (1 - TAU_G3) * shrinkage arm's
G4_MIN_DATASETS = 6  # spec section 8: ">= 6 public regression sets"

# mlp_heteroscedastic_20d carries an *exact* df/dx (frozen generator MLP); the public set
# needs a *frozen surrogate* df/dx instead (spec section 10's honestly-softer real-data
# case) -- kin8nm is smooth/periodic and small (8192 rows), a reasonable surrogate target.
G2_SYNTHETIC_DATASET = "mlp_heteroscedastic_20d"
G2_PUBLIC_DATASET = "kin8nm"
G2_DATASETS: tuple[str, ...] = (G2_SYNTHETIC_DATASET, G2_PUBLIC_DATASET)

# Heavier baselines (RealMLP/TabM via pytabkit) are --full-only, matching spec section 9's
# "only LightGBM runs in CI while richer baselines are --full-only" convention; CatBoost
# (plain, not the uncertainty head) is cheap enough to run in smoke alongside it.
PUBLIC_BASELINES_SMOKE: tuple[str, ...] = ("lightgbm", "catboost")
PUBLIC_BASELINES_FULL: tuple[str, ...] = ("lightgbm", "catboost", "realmlp", "tabm")

# n_stages/inner_steps here are deliberately far below what a first draft of this
# budget used (n_stages=30, inner_steps=60 -- 1800 sequential trust-region Newton
# steps per fit_boosted-family call). Measured directly: a single 60-inner-step
# boosting stage (16 trees, depth=2, n~500) took 22.8s even on a contended 2-core
# login node, which would put the original draft at close to a day of wall-clock
# for the combined G2+G3+G4/G5 sweep. G2's raw-vs-embed and G3's gls-vs-shrinkage
# comparisons are *relative* at a shared budget (like G1's lam=0 vs lam>0), and
# G4/G5 compare the omnibias arm against baselines fit with their own tuned
# defaults -- a smaller-but-equal budget for every omnibias arm changes absolute
# accuracy, not which side of a relative comparison wins. See
# docs/benchmarks/tabular_uncertainty.json's "budget_g2g3"/"budget_g4g5" for the
# numbers an actual `--full` run used.
BUDGET_G2G3: dict[str, dict[str, Any]] = {
    "full": {
        "n": 2000, "n_trees": 16, "depth": 2, "steps": 40,
        "n_stages": 10, "learning_rate": 0.3, "inner_steps": 20, "inner_lr": 0.05,
        "catboost_iterations": 200, "n_bins": 12, "embed_epochs": 200, "embed_lr": 0.05,
        "surrogate_epochs": 200, "surrogate_lr": 0.01,
    },
    "smoke": {
        "n": 200, "n_trees": 4, "depth": 1, "steps": 8,
        "n_stages": 6, "learning_rate": 0.3, "inner_steps": 15, "inner_lr": 0.08,
        "catboost_iterations": 30, "n_bins": 6, "embed_epochs": 40, "embed_lr": 0.05,
        "surrogate_epochs": 40, "surrogate_lr": 0.02,
    },
}

# realmlp_epochs/tabm_epochs were originally unbounded (pytabkit's own early-
# stopping default), measured directly at n~500 rows: 258s (TabM) / 59s (RealMLP)
# for a *single* fit -- with 9 public datasets x 5 seeds x 2 arms that alone is
# multiple hours. Capping both at 30 epochs (measured 33.9s / 37.4s respectively,
# a >4x cut) keeps the full public-suite sweep tractable; see the same budget
# comment above BUDGET_G2G3 for why a smaller-but-equal budget does not bias the
# omnibias-vs-baseline comparison's direction, only its absolute accuracy.
BUDGET_G4G5: dict[str, dict[str, Any]] = {
    "full": {
        "max_rows": 2000, "n_trees": 16, "depth": 2,
        "n_stages": 10, "learning_rate": 0.3, "inner_steps": 20, "inner_lr": 0.05,
        "catboost_iterations": 200, "realmlp_epochs": 30, "tabm_epochs": 30,
        "baselines": PUBLIC_BASELINES_FULL, "datasets": None,  # None -> full NOISE_PUBLIC_SUITE
    },
    "smoke": {
        "max_rows": 200, "n_trees": 4, "depth": 1,
        "n_stages": 6, "learning_rate": 0.3, "inner_steps": 15, "inner_lr": 0.08,
        "catboost_iterations": 30, "realmlp_epochs": 5, "tabm_epochs": 3,
        "baselines": PUBLIC_BASELINES_SMOKE, "datasets": ("energy_efficiency",),
    },
}


# --------------------------------------------------------------------------- #
# Small local helpers (own gate math -- see module docstring for why this     #
# file does not call benchmarks._gates.require_all_seeds / require_backend_  #
# parity directly: a falsifiable hypothesis must be allowed to fail without  #
# crashing the artifact writer).                                             #
# --------------------------------------------------------------------------- #


def _soft_all_seeds(
    per_seed: list[dict[str, Any]],
    *,
    key: str,
    expected: float,
    direction: str,
    name: str,
    min_seeds: int,
) -> dict[str, Any]:
    r"""Non-raising twin of ``benchmarks._gates.require_all_seeds``.

    Same worst-seed semantics (``direction="min"`` requires ``value >=
    expected``; ``"max"`` requires ``value <= expected``) but always *returns*
    a verdict dict rather than raising -- the caller decides what a failure
    means (here: a recorded negative result, not a crash).
    """
    if direction not in ("min", "max"):
        raise ValueError(f"{name}: direction must be 'min' or 'max'")
    values = [float(row[key]) for row in per_seed]
    if direction == "min":
        deviations = [max(0.0, float(expected) - v) for v in values]
    else:
        deviations = [max(0.0, v - float(expected)) for v in values]
    worst_idx = int(np.argmax(deviations)) if deviations else 0
    worst_deviation = float(deviations[worst_idx]) if deviations else float("nan")
    passed = bool(len(values) >= min_seeds and worst_deviation == 0.0)
    return {
        "name": name,
        "key": key,
        "expected": float(expected),
        "direction": direction,
        "n_seeds": len(values),
        "min_seeds": int(min_seeds),
        "values": values,
        "worst_seed": per_seed[worst_idx].get("seed", worst_idx) if per_seed else None,
        "worst_value": values[worst_idx] if values else float("nan"),
        "worst_deviation": worst_deviation,
        "passed": passed,
    }


def _soft_backend_parity(a: np.ndarray, b: np.ndarray, *, name: str) -> dict[str, Any]:
    """Non-raising twin of ``benchmarks._gates.require_backend_parity``."""
    left, right = np.asarray(a), np.asarray(b)
    equal = bool(left.shape == right.shape and np.array_equal(left, right, equal_nan=True))
    return {
        "name": name,
        "shape": list(left.shape),
        "passed": equal,
    }


def _top_decile_rmse(err2: np.ndarray, rank: np.ndarray, frac: float = 0.1) -> float:
    r"""RMSE over the top ``frac`` of rows by ``rank`` (descending -- highest first)."""
    n = err2.shape[0]
    k = max(1, int(round(n * frac)))
    order = np.argsort(-rank)
    return float(np.sqrt(np.mean(err2[order[:k]])))


def _mean_baseline_skill(pred: np.ndarray, y: np.ndarray, train_mean: float) -> float:
    r"""Nash-Sutcliffe skill against the **constant mean predictor** (section 8's
    "beat the constant mean-predictor", distinct from ``_gates.skill_score``'s
    zero-predictor baseline -- regression targets here are not zero-centred)."""
    mse_pred = float(np.mean((pred - y) ** 2))
    mse_mean = float(np.mean((train_mean - y) ** 2))
    if mse_mean < 1e-30:
        return float("nan")
    return 1.0 - mse_pred / mse_mean


# --------------------------------------------------------------------------- #
# G0: fit_noise_aware(lam=0.0) is bit-identical to fit_second_order.          #
# --------------------------------------------------------------------------- #


def _check_g0(seed: int = 0) -> dict[str, Any]:
    import torch
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab.torch.heteroscedastic import fit_noise_aware
    from omnibias.tab.torch.model import SoftTreeEnsemble
    from omnibias.tab.torch.train import fit_second_order

    rng = np.random.default_rng(seed)
    n, d = 120, 5
    X = rng.standard_normal((n, d))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.standard_normal(n)
    cfg = SoftTreeConfig(n_features=d, n_trees=4, depth=1, task="regression", beta_final=4.0, seed=1)

    torch.manual_seed(0)
    m0 = SoftTreeEnsemble(cfg)
    fit_second_order(m0, X, y, optimizer="trust_region", steps=10, anneal=False)

    torch.manual_seed(0)
    m1 = SoftTreeEnsemble(cfg)
    fit_noise_aware(
        m1, X, y, log_scale=np.zeros(n), lam=0.0, optimizer="trust_region", steps=10, anneal=False
    )

    flat0 = torch.cat([p.reshape(-1) for p in m0.parameters()]).detach().numpy()
    flat1 = torch.cat([p.reshape(-1) for p in m1.parameters()]).detach().numpy()
    return _soft_backend_parity(flat0, flat1, name="g0_lam0_vs_fit_second_order")


# --------------------------------------------------------------------------- #
# G1b: numerically finite-differenced Hessian matches H_task + lam*Sigma.     #
# --------------------------------------------------------------------------- #


def _numerical_hessian(f: Any, theta0: Any, *, h: float = 1e-4) -> np.ndarray:
    r"""Central finite-difference Hessian of a scalar torch closure ``f(theta)``.

    Independent of the autodiff path the trainer uses to *build* the penalty:
    this treats ``f`` as a black box, perturbing the flat parameter vector
    numerically (diagonal via the 3-point stencil, off-diagonal via the 4-point
    mixed stencil). ``O(P**2)`` evaluations -- fine for the small models this
    gate uses.
    """
    import torch

    p = theta0.shape[0]
    theta_np = theta0.detach().cpu().numpy().astype(np.float64)

    def f_at(vec: np.ndarray) -> float:
        with torch.no_grad():
            t = torch.tensor(vec, dtype=theta0.dtype)
            return float(f(t))

    f0 = f_at(theta_np)
    hess = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        ei = np.zeros(p)
        ei[i] = h
        hess[i, i] = (f_at(theta_np + ei) - 2.0 * f0 + f_at(theta_np - ei)) / (h * h)
    for i in range(p):
        for j in range(i + 1, p):
            ei = np.zeros(p)
            ej = np.zeros(p)
            ei[i] = h
            ej[j] = h
            val = (
                f_at(theta_np + ei + ej)
                - f_at(theta_np + ei - ej)
                - f_at(theta_np - ei + ej)
                + f_at(theta_np - ei - ej)
            ) / (4.0 * h * h)
            hess[i, j] = hess[j, i] = val
    return hess


def _check_g1b(lam_values: tuple[float, ...] = G1B_LAM_VALUES, seed: int = 7) -> dict[str, Any]:
    import torch
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab.torch.model import SoftTreeEnsemble
    from torch.func import functional_call

    n, d = 12, 3
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(n)
    cfg = SoftTreeConfig(n_features=d, n_trees=2, depth=1, task="regression", beta_final=3.0, seed=5)
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    model.set_beta(cfg.beta_final)

    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    log_scale = rng.normal(0.0, 0.4, size=n)
    s2 = torch.tensor(np.exp(2.0 * log_scale), dtype=torch.float64)

    names = [nm for nm, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    numels = [int(np.prod(sh)) for sh in shapes]
    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def flat_forward(theta: Any) -> Any:
        pieces, idx = {}, 0
        for nm, sh, k in zip(names, shapes, numels, strict=True):
            pieces[nm] = theta[idx : idx + k].reshape(sh)
            idx += k
        out = functional_call(model, pieces, (Xt,))
        return out.reshape(-1)

    def base_loss(theta: Any) -> Any:
        F = flat_forward(theta)
        return ((F - yt) ** 2).mean()

    # Analytic Sigma_noise = (2/n) sum_i s_i**2 g_i g_i^T via exact autodiff (this
    # is the *reference* the numerical FD Hessian is checked against, not the
    # thing being validated).
    J = torch.func.jacrev(flat_forward)(theta0)  # (n, P)
    sigma_noise = (2.0 / n) * (J * s2[:, None]).T @ J
    H0_analytic = torch.func.hessian(base_loss)(theta0)

    H0_numeric = _numerical_hessian(base_loss, theta0)
    base_check_rel = float(
        np.linalg.norm(H0_numeric - H0_analytic.numpy())
        / (np.linalg.norm(H0_analytic.numpy()) + 1e-30)
    )

    # The reference must be a *fixed constant* captured once at theta0, not
    # recomputed from the perturbed theta inside the closure below. Finite
    # differences only ever see function *values*: ``F - F.detach()`` is the
    # zero tensor at every theta (detach changes no value, only the graph), so
    # a closure written that way is, as a value-only function of theta,
    # identically base_loss everywhere -- no black-box FD stencil could ever
    # recover a nonzero Hessian contribution from it. What actually makes the
    # trainer's closure work is that TrustRegionNewtonCG/CubicNewton call
    # closure() exactly *once* per outer step (create_graph=True) and then
    # differentiate that *same* graph twice via double-backward -- so the
    # detached node is a fixed constant (this step's theta0) for the entire
    # CG solve, never re-evaluated at a perturbed theta. Freezing phi_ref here
    # reproduces that -- it is the value-level analogue of "differentiate the
    # one graph built at theta0", so the FD stencil below now targets a
    # genuine, non-constant function of theta and stays an autodiff-independent
    # check of the closed-form Sigma_noise formula.
    phi_ref = flat_forward(theta0).detach()

    per_lam: list[dict[str, Any]] = []
    worst_rel = base_check_rel
    for lam in lam_values:
        def penalized_loss(theta: Any, lam: float = lam) -> Any:
            F = flat_forward(theta)
            base = ((F - yt) ** 2).mean()
            resid = F - phi_ref
            penalty = (lam / n) * (s2 * resid * resid).sum()
            return base + penalty

        H1_numeric = _numerical_hessian(penalized_loss, theta0)
        expected = (H0_analytic + lam * sigma_noise).numpy()
        rel = float(np.linalg.norm(H1_numeric - expected) / (np.linalg.norm(expected) + 1e-30))
        per_lam.append({"lam": float(lam), "rel_error": rel})
        worst_rel = max(worst_rel, rel)

    passed = bool(worst_rel <= G1B_REL_TOL)
    return {
        "name": "g1b_hessian_correctness",
        "base_hessian_fd_vs_analytic_rel_error": base_check_rel,
        "per_lam": per_lam,
        "worst_rel_error": worst_rel,
        "tol": G1B_REL_TOL,
        "passed": passed,
    }


# --------------------------------------------------------------------------- #
# G1: the falsifier.                                                          #
# --------------------------------------------------------------------------- #


def _make_dataset(name: str, n: int, *, seed: int) -> Any:
    from omnibias.tab.bench import mlp_heteroscedastic_20d, saw_wave_2d

    if name == "saw_wave_2d":
        return saw_wave_2d(n, seed=seed)
    if name == "mlp_heteroscedastic_20d":
        return mlp_heteroscedastic_20d(n, seed=seed)
    raise ValueError(f"unknown dataset {name!r}")


def _fit_omnibias(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    *,
    lam: float,
    log_scale: np.ndarray | None,
    seed: int,
    steps: int,
    n_trees: int,
    depth: int,
) -> Any:
    import torch
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab.torch.heteroscedastic import fit_noise_aware
    from omnibias.tab.torch.model import SoftTreeEnsemble
    from omnibias.tab.torch.train import fit_second_order

    cfg = SoftTreeConfig(
        n_features=Xtr.shape[1],
        n_trees=n_trees,
        depth=depth,
        task="regression",
        n_outputs=1,
        beta_init=1.0,
        beta_final=BETA_FINAL,
        anneal_steps=steps,
        seed=seed,
    )
    torch.manual_seed(seed)
    model = SoftTreeEnsemble(cfg)
    if lam <= 0.0:
        fit_second_order(model, Xtr, ytr, optimizer="trust_region", steps=steps)
    else:
        assert log_scale is not None
        fit_noise_aware(
            model, Xtr, ytr, log_scale=log_scale, lam=lam, optimizer="trust_region", steps=steps
        )
    return model


def _pin_worker_threads(n_threads: int = 1) -> None:
    r"""``ProcessPoolExecutor`` initializer -- each worker process gets its own
    torch/BLAS thread pool by default (typically all visible cores), which
    oversubscribes badly once ``--workers`` > 1 fans out across processes on a
    shared allocation. Called once per worker at pool start-up."""
    import torch

    torch.set_num_threads(max(1, int(n_threads)))


def _parallel_map(fn: Any, units: list[tuple[Any, ...]], *, workers: int) -> list[Any]:
    r"""Run ``fn(*unit)`` for each ``unit`` in ``units``, in input order.

    Shared by the G2/G3/G4-G5 sweeps below; mirrors :func:`_run_g1_sweep`'s
    process-pool pattern (same "spawn" context and per-worker thread pinning, for
    the same reason -- torch's autograd engine cannot safely ``fork()`` once a
    ``create_graph=True`` closure has run) so a single ``--workers`` fans out
    every gate's sweep, not just G1's.
    """
    if workers <= 1:
        return [fn(*unit) for unit in units]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_pin_worker_threads, initargs=(1,)
    ) as ex:
        futures = [ex.submit(fn, *unit) for unit in units]
        return [f.result() for f in futures]


def _run_g1_one_seed(
    name: str, seed: int, budget: dict[str, int], *, cb_thread_count: int = 1
) -> dict[str, Any]:
    r"""One (dataset, seed) unit of the G1 sweep -- independent of every other
    (dataset, seed) pair given a frozen budget, so this is the parallelism
    granularity :func:`main` dispatches across a process pool with."""
    from omnibias.tab.bench import fit_predict_catboost_uncertainty
    from scipy.stats import spearmanr
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    n, steps, cb_iters = budget["n"], budget["steps"], budget["catboost_iterations"]
    n_trees, depth = budget["n_trees"], budget["depth"]

    ds = _make_dataset(name, n, seed=seed)
    assert ds.f_clean is not None and ds.s_true is not None
    Xtr_raw, Xrest_raw, ytr, yrest, _ftr, _frest, _str, srest = train_test_split(
        ds.X, ds.y, ds.f_clean, ds.s_true, test_size=0.4, random_state=seed
    )
    Xva_raw, Xte_raw, yva, yte, _sva, ste = train_test_split(
        Xrest_raw, yrest, srest, test_size=0.5, random_state=seed + 101
    )
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xva = scaler.transform(Xva_raw)
    Xte = scaler.transform(Xte_raw)

    query = np.vstack([Xtr, Xte])
    _, log_scale_all = fit_predict_catboost_uncertainty(
        Xtr, ytr, query, seed=seed, iterations=cb_iters, thread_count=cb_thread_count
    )
    n_tr = Xtr.shape[0]
    log_scale_tr = log_scale_all[:n_tr]
    log_scale_te = log_scale_all[n_tr:]

    rho, _ = spearmanr(np.exp(log_scale_te), ste)
    rho = float(rho) if np.isfinite(rho) else float("-inf")
    reference_valid = bool(rho >= SPEARMAN_FLOOR)
    if not reference_valid:
        print(
            f"  {name} seed={seed}: INVALID EXPERIMENT -- Spearman(s_hat, s_true)="
            f"{rho:.3f} < {SPEARMAN_FLOOR} on test"
        )

    model_lam0 = _fit_omnibias(
        Xtr, ytr, lam=0.0, log_scale=None, seed=seed, steps=steps, n_trees=n_trees, depth=depth
    )

    best_lam = LAM_GRID[0]
    best_val_rmse = float("inf")
    model_selected = model_lam0
    val_curve: list[dict[str, float]] = []
    for lam in LAM_GRID:
        m = _fit_omnibias(
            Xtr, ytr, lam=lam, log_scale=log_scale_tr, seed=seed, steps=steps,
            n_trees=n_trees, depth=depth,
        )
        pred_va = m.score(Xva, beta=BETA_FINAL)[:, 0]
        val_rmse = float(np.sqrt(np.mean((pred_va - yva) ** 2)))
        val_curve.append({"lam": float(lam), "val_rmse": val_rmse})
        if val_rmse < best_val_rmse:
            best_val_rmse, best_lam, model_selected = val_rmse, lam, m

    pred_lam0_te = model_lam0.score(Xte, beta=BETA_FINAL)[:, 0]
    pred_sel_te = model_selected.score(Xte, beta=BETA_FINAL)[:, 0]
    err2_lam0 = (pred_lam0_te - yte) ** 2
    err2_sel = (pred_sel_te - yte) ** 2

    rmse_top_lam0 = _top_decile_rmse(err2_lam0, ste)
    rmse_top_sel = _top_decile_rmse(err2_sel, ste)
    ratio = rmse_top_sel / rmse_top_lam0 if rmse_top_lam0 > 0.0 else float("inf")

    n_top = max(1, int(round(len(yte) * 0.1)))
    top_idx = np.argsort(-ste)[:n_top]
    train_mean = float(np.mean(ytr))
    skill_lam0 = _mean_baseline_skill(pred_lam0_te[top_idx], yte[top_idx], train_mean)
    skill_sel = _mean_baseline_skill(pred_sel_te[top_idx], yte[top_idx], train_mean)

    row = {
        "seed": int(seed),
        "spearman_shat_strue_test": rho,
        "reference_valid": reference_valid,
        "lam_selected": float(best_lam),
        "lam_selected_val_rmse": best_val_rmse,
        "val_curve": val_curve,
        "top_decile_rmse_lam0": rmse_top_lam0,
        "top_decile_rmse_lam_selected": rmse_top_sel,
        "ratio_selected_over_lam0": ratio,
        "skill_lam0_top_decile": skill_lam0,
        "skill_lam_selected_top_decile": skill_sel,
    }
    print(
        f"  {name} seed={seed}: rho={rho:.3f} lam*={best_lam:g} "
        f"top_rmse(0)={rmse_top_lam0:.4f} top_rmse(sel)={rmse_top_sel:.4f} "
        f"ratio={ratio:.4f} skill0={skill_lam0:+.3f} skill_sel={skill_sel:+.3f}",
        flush=True,
    )
    return row


def _aggregate_g1_dataset(name: str, per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    r"""Fold a dataset's ``per_seed`` rows (in seed order) into the worst-seed
    gate verdicts and the overall ``g1_passed`` decision for ``name``."""
    min_seeds = len(per_seed)
    gate_entries = [
        _soft_all_seeds(
            per_seed, key="spearman_shat_strue_test", expected=SPEARMAN_FLOOR,
            direction="min", name=f"g1_reference_valid_{name}", min_seeds=min_seeds,
        ),
        _soft_all_seeds(
            per_seed, key="skill_lam0_top_decile", expected=0.0,
            direction="min", name=f"g1_skill_lam0_{name}", min_seeds=min_seeds,
        ),
        _soft_all_seeds(
            per_seed, key="skill_lam_selected_top_decile", expected=0.0,
            direction="min", name=f"g1_skill_selected_{name}", min_seeds=min_seeds,
        ),
        _soft_all_seeds(
            per_seed, key="ratio_selected_over_lam0", expected=1.0 - TAU,
            direction="max", name=f"g1_absolute_{name}", min_seeds=min_seeds,
        ),
    ]
    reference_valid_all = all(bool(r["reference_valid"]) for r in per_seed)
    g1_dataset_passed = reference_valid_all and all(bool(e["passed"]) for e in gate_entries)
    return {
        "dataset": name,
        "per_seed": per_seed,
        "gate_entries": gate_entries,
        "reference_valid_all_seeds": reference_valid_all,
        "g1_passed": bool(g1_dataset_passed),
    }


# --------------------------------------------------------------------------- #
# G2: local target consistency -- BandFeatureEmbedder + local_target_        #
# consistency_loss vs the same downstream head on raw features.              #
# --------------------------------------------------------------------------- #


def _surrogate_df_dx(
    Xtr: np.ndarray, ytr: np.ndarray, Xquery: np.ndarray, *, seed: int, epochs: int, lr: float
) -> np.ndarray:
    r"""A **frozen surrogate** ``df/dx``: fit a small 2-layer ``tanh`` MLP regressor on
    ``(Xtr, ytr)`` and return its exact autodiff gradient at ``Xquery``.

    Used only where no known-closed-form target exists (real public data, spec section
    10's "a frozen surrogate estimate for real data... whose error propagates into the
    loss and is reported, not hidden") -- distinct from
    :func:`omnibias.tab.bench.mlp_heteroscedastic_20d`'s *exact* generator-MLP gradient,
    which gate G2 also uses on the synthetic side. This surrogate is architecturally
    distinct from :class:`~omnibias.tab.torch.model.SoftTreeEnsemble` (a plain dense MLP,
    not a soft tree), so it does not smuggle the downstream head's own inductive bias
    into the direction the embedder is asked to preserve.
    """
    import torch

    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(d, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 16), torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    ).to(torch.float64)
    Xt = torch.tensor(np.asarray(Xtr, dtype=np.float64), dtype=torch.float64)
    yt = torch.tensor(np.asarray(ytr, dtype=np.float64), dtype=torch.float64).reshape(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = ((net(Xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    Xq = torch.tensor(np.asarray(Xquery, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    out = net(Xq).sum()  # rows are independent, so d(sum)/d Xq recovers the per-row gradient
    (grad,) = torch.autograd.grad(out, Xq)
    return grad.detach().cpu().numpy()


def _fit_head_top_decile_rmse(
    X_train: np.ndarray,
    ytr: np.ndarray,
    X_test: np.ndarray,
    yte: np.ndarray,
    rank_key_te: np.ndarray,
    *,
    seed: int,
    n_trees: int,
    depth: int,
    steps: int,
) -> float:
    r"""Train a plain :func:`fit_second_order` ``SoftTreeEnsemble`` head on
    ``(X_train, ytr)`` and return its top-decile (by ``rank_key_te``) test RMSE -- shared
    by G2's "raw" and "embed" arms so the only difference between them is the input
    features."""
    import torch
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab.torch.model import SoftTreeEnsemble
    from omnibias.tab.torch.train import fit_second_order

    cfg = SoftTreeConfig(
        n_features=X_train.shape[1], n_trees=n_trees, depth=depth, task="regression",
        n_outputs=1, beta_init=1.0, beta_final=BETA_FINAL, anneal_steps=steps, seed=seed,
    )
    torch.manual_seed(seed)
    model = SoftTreeEnsemble(cfg)
    fit_second_order(model, X_train, ytr, optimizer="trust_region", steps=steps)
    pred = model.score(X_test, beta=BETA_FINAL)[:, 0]
    return _top_decile_rmse((pred - yte) ** 2, rank_key_te)


def _check_g2_one_seed(name: str, seed: int, budget: dict[str, Any]) -> dict[str, Any]:
    r"""One (dataset, seed) unit of the G2 sweep: raw-feature head vs
    embedded-feature head, both :func:`fit_second_order`, top-decile RMSE.
    """
    import torch
    from omnibias.tab.bench import (
        fit_predict_catboost_uncertainty,
        load_dataset,
        mlp_heteroscedastic_20d,
    )
    from omnibias.tab.torch.embed import BandFeatureEmbedder, local_target_consistency_loss
    from sklearn.model_selection import train_test_split as _split
    from sklearn.preprocessing import StandardScaler

    n_trees, depth, steps = budget["n_trees"], budget["depth"], budget["steps"]
    n_bins, embed_epochs, embed_lr = budget["n_bins"], budget["embed_epochs"], budget["embed_lr"]

    if name == G2_SYNTHETIC_DATASET:
        ds = mlp_heteroscedastic_20d(budget["n"], seed=seed)
        assert ds.df_dx is not None and ds.s_true is not None
        Xtr_raw, Xte_raw, ytr, yte, dtr_raw, _dte_raw, _str, ste = _split(
            ds.X, ds.y, ds.df_dx, ds.s_true, test_size=0.3, random_state=seed
        )
        df_source = "exact"
        rank_key_te = ste  # true noise ranks the decile (matches G1's convention)
    else:
        try:
            ds = load_dataset(name, max_rows=budget["n"], seed=seed)
        except RuntimeError as exc:
            print(f"  G2 {name} seed={seed}: SKIPPED ({exc})", flush=True)
            return {"seed": int(seed), "status": "skipped", "skip_reason": str(exc)}
        Xtr_raw, Xte_raw, ytr, yte = _split(ds.X, ds.y, test_size=0.3, random_state=seed)
        dtr_raw = None
        df_source = "surrogate"
        rank_key_te = None  # filled in below once Xte is standardized

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    if df_source == "exact":
        assert dtr_raw is not None
        dtr = dtr_raw * scaler.scale_[None, :]  # chain rule: df/dx' = df/dx * dx/dx'
    else:
        dtr = _surrogate_df_dx(
            Xtr, ytr, Xtr, seed=seed, epochs=budget["surrogate_epochs"], lr=budget["surrogate_lr"]
        )
        # No ground-truth noise on real data -- rank the decile by a frozen, train-only-fit
        # CatBoost uncertainty estimate instead (model-based, spec section 4(c)/10).
        _, log_scale_te = fit_predict_catboost_uncertainty(
            Xtr, ytr, Xte, seed=seed, iterations=budget["catboost_iterations"]
        )
        rank_key_te = np.exp(log_scale_te)

    d = Xtr.shape[1]
    embedder = BandFeatureEmbedder(d, n_bins=n_bins, role="band", init="quantile", X_ref=Xtr)
    opt = torch.optim.Adam(embedder.parameters(), lr=embed_lr)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float64)
    dtr_t = torch.tensor(np.asarray(dtr, dtype=np.float64), dtype=torch.float64)
    for _ in range(embed_epochs):
        opt.zero_grad(set_to_none=True)
        loss = local_target_consistency_loss(embedder, Xtr_t, df_dx=dtr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        Etr = embedder(Xtr_t).numpy()
        Ete = embedder(torch.tensor(Xte, dtype=torch.float64)).numpy()

    rmse_top_raw = _fit_head_top_decile_rmse(
        Xtr, ytr, Xte, yte, rank_key_te, seed=seed, n_trees=n_trees, depth=depth, steps=steps
    )
    rmse_top_embed = _fit_head_top_decile_rmse(
        Etr, ytr, Ete, yte, rank_key_te, seed=seed, n_trees=n_trees, depth=depth, steps=steps
    )
    ratio = rmse_top_embed / rmse_top_raw if rmse_top_raw > 0.0 else float("inf")
    row = {
        "seed": int(seed),
        "status": "completed",
        "df_dx_source": df_source,
        "top_decile_rmse_raw": rmse_top_raw,
        "top_decile_rmse_embed": rmse_top_embed,
        "ratio_embed_over_raw": ratio,
    }
    print(
        f"  G2 {name} seed={seed}: df_dx={df_source} raw={rmse_top_raw:.4f} "
        f"embed={rmse_top_embed:.4f} ratio={ratio:.4f}",
        flush=True,
    )
    return row


def _run_g2_sweep(
    *, seeds: tuple[int, ...], budget: dict[str, Any], workers: int = 1
) -> dict[str, dict[str, Any]]:
    units = [(name, seed, budget) for name in G2_DATASETS for seed in seeds]
    rows = _parallel_map(_check_g2_one_seed, units, workers=workers)
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in G2_DATASETS}
    for (name, _seed, _budget), row in zip(units, rows, strict=True):
        by_name[name].append(row)

    by_dataset: dict[str, dict[str, Any]] = {}
    for name in G2_DATASETS:
        per_seed = by_name[name]
        completed = [r for r in per_seed if r["status"] == "completed"]
        if not completed:
            by_dataset[name] = {
                "dataset": name,
                "per_seed": per_seed,
                "gate_entry": {"name": f"g2_absolute_{name}", "passed": False, "reason": "no seed completed"},
                "g2_passed": False,
            }
            continue
        gate = _soft_all_seeds(
            completed, key="ratio_embed_over_raw", expected=1.0 - TAU_G2,
            direction="max", name=f"g2_absolute_{name}", min_seeds=len(seeds),
        )
        by_dataset[name] = {
            "dataset": name,
            "per_seed": per_seed,
            "gate_entry": gate,
            "g2_passed": bool(gate["passed"]),
        }
    return by_dataset


# --------------------------------------------------------------------------- #
# G3: fit_boosted_heteroscedastic(weighting="gls") vs weighting="shrinkage"  #
# (bit-identical to plain fit_boosted) on top-decile RMSE.                   #
# --------------------------------------------------------------------------- #


def _check_g3_one_seed(name: str, seed: int, budget: dict[str, Any]) -> dict[str, Any]:
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab.bench import fit_predict_catboost_uncertainty
    from omnibias.tab.torch.boosting import fit_boosted_heteroscedastic
    from sklearn.model_selection import train_test_split as _split
    from sklearn.preprocessing import StandardScaler

    ds = _make_dataset(name, budget["n"], seed=seed)
    assert ds.s_true is not None
    Xtr_raw, Xte_raw, ytr, yte, _str, ste = _split(
        ds.X, ds.y, ds.s_true, test_size=0.3, random_state=seed
    )
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    _, log_scale_tr = fit_predict_catboost_uncertainty(
        Xtr, ytr, Xtr, seed=seed, iterations=budget["catboost_iterations"]
    )
    cfg = SoftTreeConfig(
        n_features=Xtr.shape[1], n_trees=budget["n_trees"], depth=budget["depth"],
        task="regression", n_outputs=1, beta_final=BETA_FINAL, seed=seed,
    )
    common = dict(
        n_stages=budget["n_stages"], learning_rate=budget["learning_rate"],
        inner_steps=budget["inner_steps"], inner_lr=budget["inner_lr"],
    )
    model_gls, _ = fit_boosted_heteroscedastic(
        Xtr, ytr, cfg, log_scale=log_scale_tr, weighting="gls", **common
    )
    model_shr, _ = fit_boosted_heteroscedastic(
        Xtr, ytr, cfg, log_scale=log_scale_tr, weighting="shrinkage", **common
    )
    pred_gls = model_gls.score(Xte)[:, 0]
    pred_shr = model_shr.score(Xte)[:, 0]
    rmse_top_gls = _top_decile_rmse((pred_gls - yte) ** 2, ste)
    rmse_top_shr = _top_decile_rmse((pred_shr - yte) ** 2, ste)
    ratio = rmse_top_gls / rmse_top_shr if rmse_top_shr > 0.0 else float("inf")
    row = {
        "seed": int(seed),
        "top_decile_rmse_gls": rmse_top_gls,
        "top_decile_rmse_shrinkage": rmse_top_shr,
        "ratio_gls_over_shrinkage": ratio,
    }
    print(
        f"  G3 {name} seed={seed}: gls={rmse_top_gls:.4f} shrinkage={rmse_top_shr:.4f} "
        f"ratio={ratio:.4f}",
        flush=True,
    )
    return row


def _run_g3_sweep(
    *, seeds: tuple[int, ...], budget: dict[str, Any], workers: int = 1
) -> dict[str, dict[str, Any]]:
    units = [(name, seed, budget) for name in DATASETS for seed in seeds]  # both synthetic sets
    rows = _parallel_map(_check_g3_one_seed, units, workers=workers)
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASETS}
    for (name, _seed, _budget), row in zip(units, rows, strict=True):
        by_name[name].append(row)

    by_dataset: dict[str, dict[str, Any]] = {}
    for name in DATASETS:
        per_seed = by_name[name]
        min_seeds = len(per_seed)
        gate = _soft_all_seeds(
            per_seed, key="ratio_gls_over_shrinkage", expected=1.0 - TAU_G3,
            direction="max", name=f"g3_absolute_{name}", min_seeds=min_seeds,
        )
        by_dataset[name] = {
            "dataset": name,
            "per_seed": per_seed,
            "gate_entry": gate,
            "g3_passed": bool(gate["passed"]),
        }
    return by_dataset


# --------------------------------------------------------------------------- #
# G4/G5: public-suite honesty report (win/loss table, no aggregate-only     #
# reporting) + no-regression-on-the-aggregate-metric check.                 #
# --------------------------------------------------------------------------- #


def _run_g4g5_one_seed(name: str, seed: int, budget: dict[str, Any]) -> dict[str, Any]:
    import torch
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab.bench import (
        NOISE_PUBLIC_MAX_ROWS,
        fit_predict_catboost,
        fit_predict_catboost_uncertainty,
        fit_predict_lightgbm,
        fit_predict_realmlp,
        fit_predict_tabm,
        load_dataset,
    )
    from omnibias.tab.torch.boosting import fit_boosted, fit_boosted_heteroscedastic
    from omnibias.tab.torch.model import SoftTreeEnsemble
    from omnibias.tab.torch.train import fit_second_order
    from sklearn.model_selection import train_test_split as _split
    from sklearn.preprocessing import StandardScaler

    cap = NOISE_PUBLIC_MAX_ROWS.get(name)
    row_cap = budget["max_rows"] if cap is None else min(budget["max_rows"], cap)
    try:
        ds = load_dataset(name, max_rows=row_cap, seed=seed)
    except RuntimeError as exc:
        print(f"  G4/G5 {name} seed={seed}: SKIPPED ({exc})", flush=True)
        return {"dataset": name, "seed": int(seed), "status": "skipped", "skip_reason": str(exc)}

    Xtr_raw, Xte_raw, ytr, yte = _split(ds.X, ds.y, test_size=0.3, random_state=seed)
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)
    cb_iters = budget["catboost_iterations"]

    # Model-based decile ranking on real data (no ground-truth noise) -- section 4(c)/10.
    _, log_scale_te = fit_predict_catboost_uncertainty(Xtr, ytr, Xte, seed=seed, iterations=cb_iters)
    s_hat_te = np.exp(log_scale_te)
    _, log_scale_tr = fit_predict_catboost_uncertainty(Xtr, ytr, Xtr, seed=seed, iterations=cb_iters)

    cfg = SoftTreeConfig(
        n_features=Xtr.shape[1], n_trees=budget["n_trees"], depth=budget["depth"],
        task="regression", n_outputs=1, beta_final=BETA_FINAL, seed=seed,
    )
    torch.manual_seed(seed)
    model_so = SoftTreeEnsemble(cfg)
    fit_second_order(model_so, Xtr, ytr, optimizer="trust_region", steps=budget["inner_steps"])
    pred_so = model_so.score(Xte, beta=BETA_FINAL)[:, 0]

    boost_common = dict(
        n_stages=budget["n_stages"], learning_rate=budget["learning_rate"],
        inner_steps=budget["inner_steps"], inner_lr=budget["inner_lr"],
    )
    model_boosted, _ = fit_boosted(Xtr, ytr, cfg, **boost_common)
    pred_boosted = model_boosted.score(Xte)[:, 0]

    # The predeclared "best Phase 1+2 arm" (fit_noise_aware is excluded -- G1 falsified it).
    model_omnibias, _ = fit_boosted_heteroscedastic(
        Xtr, ytr, cfg, log_scale=log_scale_tr, weighting="gls", **boost_common
    )
    pred_omnibias = model_omnibias.score(Xte)[:, 0]

    baseline_preds: dict[str, np.ndarray] = {}
    for bname in budget["baselines"]:
        if bname == "lightgbm":
            pred, _ = fit_predict_lightgbm(Xtr, ytr, Xte, task="regression", n_outputs=1, seed=seed)
        elif bname == "catboost":
            pred, _ = fit_predict_catboost(
                Xtr, ytr, Xte, task="regression", n_outputs=1, seed=seed, iterations=cb_iters
            )
        elif bname == "realmlp":
            pred, _ = fit_predict_realmlp(Xtr, ytr, Xte, seed=seed, n_epochs=budget.get("realmlp_epochs"))
        elif bname == "tabm":
            pred, _ = fit_predict_tabm(Xtr, ytr, Xte, seed=seed, n_epochs=budget.get("tabm_epochs"))
        else:
            raise ValueError(f"unknown baseline {bname!r}")
        baseline_preds[bname] = pred

    def _rmse(pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - yte) ** 2)))

    def _top(pred: np.ndarray) -> float:
        return _top_decile_rmse((pred - yte) ** 2, s_hat_te)

    omnibias_top = _top(pred_omnibias)
    vs_baselines = {
        bname: {
            "baseline_top_decile_rmse": _top(pred),
            "baseline_overall_rmse": _rmse(pred),
            "omnibias_top_decile_rmse": omnibias_top,
            "winner": "omnibias" if omnibias_top < _top(pred) else "baseline",
        }
        for bname, pred in baseline_preds.items()
    }
    row = {
        "dataset": name,
        "seed": int(seed),
        "status": "completed",
        "n_rows": int(ds.X.shape[0]),
        "n_features": int(ds.X.shape[1]),
        "omnibias_top_decile_rmse": omnibias_top,
        "omnibias_overall_rmse": _rmse(pred_omnibias),
        "fit_second_order_overall_rmse": _rmse(pred_so),
        "fit_boosted_overall_rmse": _rmse(pred_boosted),
        "vs_baselines": vs_baselines,
    }
    print(
        f"  G4/G5 {name} seed={seed}: omnibias_top={omnibias_top:.4f} "
        f"omnibias_overall={row['omnibias_overall_rmse']:.4f} vs "
        + ", ".join(f"{k}={v['winner']}" for k, v in vs_baselines.items()),
        flush=True,
    )
    return row


def _aggregate_g4g5_dataset(name: str, per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in per_seed if r["status"] == "completed"]
    if not completed:
        return {
            "dataset": name, "status": "skipped", "per_seed": per_seed,
            "skip_reason": next((r.get("skip_reason") for r in per_seed), "no seeds completed"),
        }

    def _seed_noise(vals: list[float]) -> float:
        return float(np.std(np.asarray(vals, dtype=np.float64), ddof=0))

    omnibias_overall = [float(r["omnibias_overall_rmse"]) for r in completed]
    so_overall = [float(r["fit_second_order_overall_rmse"]) for r in completed]
    boosted_overall = [float(r["fit_boosted_overall_rmse"]) for r in completed]

    baseline_names = sorted(completed[0]["vs_baselines"].keys())
    vs_table: dict[str, Any] = {}
    for bname in baseline_names:
        base_top = [float(r["vs_baselines"][bname]["baseline_top_decile_rmse"]) for r in completed]
        omni_top = [float(r["vs_baselines"][bname]["omnibias_top_decile_rmse"]) for r in completed]
        n_omni_wins = sum(1 for r in completed if r["vs_baselines"][bname]["winner"] == "omnibias")
        n = len(completed)
        vs_table[bname] = {
            "mean_omnibias_top_decile_rmse": float(np.mean(omni_top)),
            "mean_baseline_top_decile_rmse": float(np.mean(base_top)),
            "omnibias_wins": int(n_omni_wins),
            "n_seeds": n,
            "winner": "omnibias" if 2 * n_omni_wins > n else ("baseline" if 2 * n_omni_wins < n else "tie"),
        }

    g5_vs_so = bool(np.mean(omnibias_overall) <= np.mean(so_overall) + _seed_noise(so_overall))
    g5_vs_boosted = bool(np.mean(omnibias_overall) <= np.mean(boosted_overall) + _seed_noise(boosted_overall))
    return {
        "dataset": name,
        "status": "completed",
        "n_seeds": len(completed),
        "per_seed": per_seed,
        "vs_baselines": vs_table,
        "mean_omnibias_overall_rmse": float(np.mean(omnibias_overall)),
        "mean_fit_second_order_overall_rmse": float(np.mean(so_overall)),
        "mean_fit_boosted_overall_rmse": float(np.mean(boosted_overall)),
        "g5_not_worse_vs_fit_second_order": g5_vs_so,
        "g5_not_worse_vs_fit_boosted": g5_vs_boosted,
        "g5_dataset_passed": bool(g5_vs_so and g5_vs_boosted),
    }


def _run_g4g5_sweep(
    *, names: tuple[str, ...], seeds: tuple[int, ...], budget: dict[str, Any], workers: int = 1
) -> dict[str, dict[str, Any]]:
    units = [(name, seed, budget) for name in names for seed in seeds]
    rows = _parallel_map(_run_g4g5_one_seed, units, workers=workers)
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for (name, _seed, _budget), row in zip(units, rows, strict=True):
        by_name[name].append(row)

    by_dataset: dict[str, dict[str, Any]] = {}
    for name in names:
        by_dataset[name] = _aggregate_g4g5_dataset(name, by_name[name])
    return by_dataset


def _checkpoint_path(checkpoint_dir: Path, tier: str, name: str, seed: int) -> Path:
    return checkpoint_dir / f"{tier}_{name}_seed{seed}.json"


def _run_g1_one_seed_checkpointed(
    name: str,
    seed: int,
    budget: dict[str, int],
    *,
    checkpoint_dir: str | None,
    tier: str,
) -> dict[str, Any]:
    r"""``_run_g1_one_seed`` with an optional on-disk cache under ``checkpoint_dir``.

    A GPU-cluster job that gets requeued by the scheduler after a transient
    execution-host failure re-runs this script from ``argv[0]`` with a cold
    process pool -- every (dataset, seed) unit that already finished before
    the requeue would otherwise be silently redone at full cost.
    ``checkpoint_dir`` is a ``str`` (not a ``Path``) because this function is
    a ``ProcessPoolExecutor`` target and must be trivially picklable.
    """
    if checkpoint_dir is not None:
        path = _checkpoint_path(Path(checkpoint_dir), tier, name, seed)
        if path.is_file():
            print(f"  {name} seed={seed}: resumed from checkpoint {path.name}", flush=True)
            return dict(json.loads(path.read_text(encoding="utf-8")))
    row = _run_g1_one_seed(name, seed, budget)
    if checkpoint_dir is not None:
        path = _checkpoint_path(Path(checkpoint_dir), tier, name, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row), encoding="utf-8")
    return row


def _run_g1_sweep(
    *,
    seeds: tuple[int, ...],
    budget: dict[str, int],
    workers: int,
    checkpoint_dir: Path | None = None,
    tier: str = "full",
) -> dict[str, dict[str, Any]]:
    r"""Run every ``(dataset, seed)`` unit and group the rows back by dataset.

    ``workers <= 1`` runs serially in this process (the smoke-tier default --
    unchanged behavior, easiest to debug). ``workers > 1`` fans the *flat* list
    of ``len(DATASETS) * len(seeds)`` independent units (not just the seeds
    within one dataset) across a ``ProcessPoolExecutor``, since a single
    (dataset, seed) fit chain (1 CatBoost fit + 6 sequential omnibias fits) is
    the natural unit -- the 6 lam arms within it stay sequential (they reuse
    the same frozen ``log_scale_tr``, but are cheap relative to a full fit) and
    parallelizing further than one process per unit would oversubscribe.

    ``checkpoint_dir``, when given, makes the sweep resumable: each finished
    unit's row is cached to ``<checkpoint_dir>/<tier>_<name>_seed<seed>.json``
    and reloaded instead of recomputed on a subsequent invocation (see
    :func:`_run_g1_one_seed_checkpointed`).
    """
    units = [(name, seed) for name in DATASETS for seed in seeds]
    cdir_str = str(checkpoint_dir) if checkpoint_dir is not None else None
    if workers <= 1:
        rows = {
            (name, seed): _run_g1_one_seed_checkpointed(
                name, seed, budget, checkpoint_dir=cdir_str, tier=tier
            )
            for name, seed in units
        }
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # "spawn", not the Linux default "fork": torch's autograd engine spawns
        # reentrant-backward worker threads (eagerly once a CUDA device is
        # visible, e.g. under a GPU-cluster allocation, even if this code never
        # calls .cuda()) that a fork()'d child cannot safely inherit -- every
        # create_graph=True closure() call in _fit_omnibias would then raise
        # "Unable to handle autograd's threading in combination with
        # fork-based multiprocessing". spawn starts each worker as a fresh
        # interpreter instead, at the cost of re-importing torch per worker.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_pin_worker_threads,
            initargs=(1,),
        ) as ex:
            futures = {
                (name, seed): ex.submit(
                    _run_g1_one_seed_checkpointed,
                    name,
                    seed,
                    budget,
                    checkpoint_dir=cdir_str,
                    tier=tier,
                )
                for name, seed in units
            }
            rows = {k: f.result() for k, f in futures.items()}

    by_dataset: dict[str, dict[str, Any]] = {}
    for name in DATASETS:
        per_seed = [rows[(name, seed)] for seed in seeds]
        by_dataset[name] = _aggregate_g1_dataset(name, per_seed)
    return by_dataset


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="acceptance tier: 5 seeds, n=4000, steps=80, CatBoost 500 iterations",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "process-pool width across the (dataset, seed) units of the G1, G2, G3, "
            "and G4/G5 sweeps (default 1 = serial, matching earlier runs bit-for-bit "
            "since each unit is otherwise independent and seeded; >1 only changes "
            "wall-clock, sized to the cluster job's core allocation, not the "
            "reported numbers)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "cache each finished (dataset, seed) unit under "
            "$OMNIBIAS_SCRATCH/tabular_uncertainty/checkpoints/ and reuse it on a "
            "subsequent invocation -- a cheap insurance policy against a GPU "
            "job getting requeued by the scheduler after a transient execution-"
            "host failure and losing already-finished units; off by default so "
            "a plain re-run always recomputes everything"
        ),
    )
    args = parser.parse_args(argv)
    full = bool(args.full)
    workers = max(1, int(args.workers))
    tier = "full" if full else "smoke"
    budget = BUDGET[tier]
    seeds = SEEDS_FULL if full else SEEDS_SMOKE
    artifact_name = "tabular_uncertainty.json" if full else "tabular_uncertainty_smoke.json"
    checkpoint_dir = SCRATCH / "tabular_uncertainty" / "checkpoints" if args.resume else None

    t0 = time.perf_counter()
    print("checking G0 (plumbing identity: fit_noise_aware(lam=0) vs fit_second_order)...")
    g0 = _check_g0()
    print(f"  g0 passed={g0['passed']}")

    print("checking G1b (finite-differenced Hessian vs H_task + lam*Sigma_noise)...")
    g1b = _check_g1b()
    print(f"  g1b passed={g1b['passed']} worst_rel_error={g1b['worst_rel_error']:.3e}")

    print(
        f"running G1 sweep (tier={tier}, seeds={seeds}, n={budget['n']}, "
        f"steps={budget['steps']}, workers={workers}, "
        f"resume={'off' if checkpoint_dir is None else checkpoint_dir})..."
    )
    g1_by_dataset = _run_g1_sweep(
        seeds=seeds, budget=budget, workers=workers, checkpoint_dir=checkpoint_dir, tier=tier
    )
    g1_gate_entries: list[dict[str, Any]] = []
    for name in DATASETS:
        g1_gate_entries.extend(g1_by_dataset[name]["gate_entries"])
    g1_all_passed = all(bool(r["g1_passed"]) for r in g1_by_dataset.values())

    budget_g2g3 = BUDGET_G2G3[tier]
    print(f"running G2 sweep (tier={tier}, datasets={G2_DATASETS}, seeds={seeds})...")
    g2_by_dataset = _run_g2_sweep(seeds=seeds, budget=budget_g2g3, workers=workers)
    g2_gate_entries = [g2_by_dataset[name]["gate_entry"] for name in G2_DATASETS]
    g2_all_passed = all(bool(g2_by_dataset[name]["g2_passed"]) for name in G2_DATASETS)
    print(f"  g2_all_passed={g2_all_passed}")

    print(f"running G3 sweep (tier={tier}, datasets={DATASETS}, seeds={seeds})...")
    g3_by_dataset = _run_g3_sweep(seeds=seeds, budget=budget_g2g3, workers=workers)
    g3_gate_entries = [g3_by_dataset[name]["gate_entry"] for name in DATASETS]
    g3_all_passed = all(bool(g3_by_dataset[name]["g3_passed"]) for name in DATASETS)
    print(f"  g3_all_passed={g3_all_passed}")

    from omnibias.tab.bench import NOISE_PUBLIC_SUITE

    budget_g4g5 = BUDGET_G4G5[tier]
    g4g5_names = tuple(NOISE_PUBLIC_SUITE) if budget_g4g5["datasets"] is None else tuple(budget_g4g5["datasets"])
    print(
        f"running G4/G5 public-suite sweep (tier={tier}, datasets={g4g5_names}, "
        f"seeds={seeds}, baselines={budget_g4g5['baselines']})..."
    )
    g4g5_by_dataset = _run_g4g5_sweep(names=g4g5_names, seeds=seeds, budget=budget_g4g5, workers=workers)
    g4_completed = [b for b in g4g5_by_dataset.values() if b["status"] == "completed"]
    g4_skipped = [b for b in g4g5_by_dataset.values() if b["status"] == "skipped"]
    g4_entry = {
        "name": "g4_public_suite_honesty_report",
        # G4 is a REPORTING-COMPLETENESS gate (the full win/loss table exists for every
        # named baseline on >= G4_MIN_DATASETS sets), not a win/loss claim -- matching
        # 05-02 G3's "no aggregate-only reporting" rule (section 8).
        "passed": bool(full and len(g4_completed) >= G4_MIN_DATASETS),
        "n_completed": len(g4_completed),
        "n_skipped": len(g4_skipped),
        "min_required": G4_MIN_DATASETS,
    }
    g5_entry = {
        "name": "g5_no_regression_aggregate",
        "passed": bool(
            full
            and len(g4_completed) >= G4_MIN_DATASETS
            and all(bool(b["g5_dataset_passed"]) for b in g4_completed)
        ),
        "per_dataset": {
            b["dataset"]: {
                "vs_fit_second_order": bool(b["g5_not_worse_vs_fit_second_order"]),
                "vs_fit_boosted": bool(b["g5_not_worse_vs_fit_boosted"]),
            }
            for b in g4_completed
        },
    }
    print(f"  g4_entry passed={g4_entry['passed']} ({len(g4_completed)} completed) "
          f"g5_entry passed={g5_entry['passed']}")

    # G0/G1b are unconditional wiring gates (deterministic, true by construction /
    # validated math); G1-G5 only gate the exit code in --full, per the module
    # docstring -- a falsifiable scientific hypothesis is not decided by a
    # reduced-budget smoke run either way, and G4/G5 are structurally unsatisfiable
    # below the full >= 6-dataset suite regardless of tier.
    gated_entries = [g0, g1b] + (
        g1_gate_entries + g2_gate_entries + g3_gate_entries + [g4_entry, g5_entry] if full else []
    )
    gates = dict(gates_block(gated_entries))

    win_loss_table = [
        {
            "dataset": name,
            "status": g4g5_by_dataset[name]["status"],
            "vs_baselines": g4g5_by_dataset[name].get("vs_baselines"),
            "mean_omnibias_overall_rmse": g4g5_by_dataset[name].get("mean_omnibias_overall_rmse"),
            "mean_fit_second_order_overall_rmse": g4g5_by_dataset[name].get(
                "mean_fit_second_order_overall_rmse"
            ),
            "mean_fit_boosted_overall_rmse": g4g5_by_dataset[name].get("mean_fit_boosted_overall_rmse"),
            "g5_dataset_passed": g4g5_by_dataset[name].get("g5_dataset_passed"),
            "skip_reason": g4g5_by_dataset[name].get("skip_reason"),
        }
        for name in g4g5_names
    ]

    config = {
        "family": "tabular_uncertainty_noise_aware_newton",
        "tier": tier,
        "full": full,
        "smoke_is_wiring_gate": not full,
        "g1_gated_in_this_tier": full,
        "seeds": list(seeds),
        "lam_grid": list(LAM_GRID),
        "spearman_floor": SPEARMAN_FLOOR,
        "tau": TAU,
        "tau_g2": TAU_G2,
        "tau_g3": TAU_G3,
        "g4_min_datasets": G4_MIN_DATASETS,
        "budget": budget,
        "budget_g2g3": budget_g2g3,
        "budget_g4g5": budget_g4g5,
        "beta_final": BETA_FINAL,
        "datasets": list(DATASETS),
        "g2_datasets": list(G2_DATASETS),
        "g4g5_datasets": list(g4g5_names),
        "g4g5_baselines": list(budget_g4g5["baselines"]),
        "workers": workers,
        "resumed_from_checkpoint": checkpoint_dir is not None,
    }
    payload = provenance(schema="tabular-uncertainty-v1", config=config)
    payload.update(
        {
            "g0": g0,
            "g1b": g1b,
            "g1_by_dataset": g1_by_dataset,
            "g1_all_passed": bool(g1_all_passed),
            "g2_by_dataset": g2_by_dataset,
            "g2_all_passed": bool(g2_all_passed),
            "g3_by_dataset": g3_by_dataset,
            "g3_all_passed": bool(g3_all_passed),
            "g4g5_by_dataset": g4g5_by_dataset,
            "win_loss_table": win_loss_table,
            "g4": g4_entry,
            "g5": g5_entry,
            "gates": gates,
            "honesty": {
                "claim_rung": 1,
                "s_hat_guarantee_kind": "MODEL_BASED",
                "s_hat_is_sound_enclosure": False,
                "temperature_collapse": True,
                "bias_collapse": False,
                "no_aggregate_only_headline": True,
                "g0_earned": bool(g0["passed"]),
                "g1b_earned": bool(g1b["passed"]),
                "g1_earned": bool(g1_all_passed) if full else False,
                "g1_recorded_not_gated_in_smoke": not full,
                "g2_earned": bool(g2_all_passed) if full else False,
                "g3_earned": bool(g3_all_passed) if full else False,
                "g4_earned": bool(g4_entry["passed"]),
                "g5_earned": bool(g5_entry["passed"]),
                "phase2_gates_recorded_not_gated_in_smoke": not full,
                "falsifier_honestly_reported": True,
                "licensed_sentence": (
                    "noise-damped exact Newton (fit_noise_aware) beats plain exact "
                    "Newton on the true-noise top decile of test, on both synthetic "
                    "sets, over the full seed sweep"
                    if full and g1_all_passed
                    else (
                        "G1 not gated in the smoke tier -- see docs/benchmarks/"
                        "tabular_uncertainty.json (--full) for the acceptance verdict"
                        if not full
                        else (
                            "G1 failed on at least one dataset: Proposal A "
                            "(fit_noise_aware) is falsified at this budget; "
                            "fit_boosted_heteroscedastic (independent, section 4(d)) "
                            "is not affected"
                        )
                    )
                ),
                "phase2_licensed_sentence": (
                    "smoke wiring only for G2-G5 -- see docs/benchmarks/"
                    "tabular_uncertainty.json (--full) for the acceptance verdicts"
                    if not full
                    else (
                        f"G2 (local target consistency) {'passed' if g2_all_passed else 'did not pass'}; "
                        f"G3 (boosted heteroscedastic GLS win) "
                        f"{'passed' if g3_all_passed else 'did not pass'}; "
                        f"G4 (public-suite honesty report) is a reporting-completeness gate, "
                        f"reported in win_loss_table for {len(g4_completed)} datasets "
                        f"(not a win/loss claim); "
                        f"G5 (no aggregate regression) "
                        f"{'passed' if g5_entry['passed'] else 'did not pass'}"
                    )
                ),
                "theorem_prover_verified": False,
                "mathlib_verified": False,
            },
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )

    out = write_json(artifact_name, payload)
    print(
        f"wrote {out}  all_passed={gates['all_passed']}  g0={g0['passed']} g1b={g1b['passed']} "
        f"g1={g1_all_passed} g2={g2_all_passed} g3={g3_all_passed} "
        f"g4={g4_entry['passed']} g5={g5_entry['passed']} (gated={full})"
    )
    if full:
        scratch_dir = SCRATCH / "tabular_uncertainty"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        scratch_path = scratch_dir / artifact_name
        scratch_path.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied to {scratch_path}")
    if not gates["all_passed"]:
        raise SystemExit(1)
    return dict(payload)


if __name__ == "__main__":
    main()
