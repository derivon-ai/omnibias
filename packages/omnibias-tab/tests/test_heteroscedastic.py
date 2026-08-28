# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Theory 05-03 Phase 1: closed-form NLL correctness + the zero-gradient penalty math.

Four gates from the plan, in order:

1. **NLL gradient vs autograd** -- :func:`gaussian_nll_grad_hess`'s closed-form
   ``(grad_u, grad_v)`` matches ``torch.autograd.grad`` of the torch twin
   :func:`gaussian_nll_elementwise`.
2. **Fisher positive-definiteness** -- the returned Fisher block is
   ``diag(1/s**2, 2)``, positive-definite by construction; checked by eigenvalue.
3. **G0** -- ``fit_noise_aware(..., lam=0.0)`` is *bit-identical* to
   ``fit_second_order`` (same seed, same steps): the penalty term is the exact
   Python float ``0.0`` at every step, not merely small.
4. **G1b** -- the penalized closure's Hessian equals ``H + lam * Sigma_noise``
   to float tolerance, verified directly against
   ``Sigma_noise = (2/n) sum_i s_i**2 g_i g_i^T`` via ``torch.func``.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.tab._core.heteroscedastic import (
    gaussian_nll,
    gaussian_nll_grad_hess,
    log_scale_from_variance,
)
from omnibias.tab.bench import (
    fit_predict_catboost_uncertainty,
    mlp_heteroscedastic_20d,
    saw_wave_2d,
    saw_wave_f,
    saw_wave_s,
)

torch = pytest.importorskip("torch")

from omnibias.tab import SoftTreeConfig  # noqa: E402
from omnibias.tab.torch import SoftTreeEnsemble, fit_second_order  # noqa: E402
from omnibias.tab.torch.heteroscedastic import (  # noqa: E402
    HeteroscedasticHead,
    fit_heteroscedastic,
    fit_noise_aware,
    gaussian_nll_elementwise,
)

# --------------------------------------------------------------------------- #
# 1. NLL: closed-form grad + Fisher vs autograd / direct formula.             #
# --------------------------------------------------------------------------- #


def _random_uvy(seed: int = 0, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 2.0, size=n)
    v = rng.normal(0.0, 0.5, size=n)  # log-scale, kept modest to avoid overflow
    y = u + np.exp(v) * rng.standard_normal(n)
    return u, v, y


def test_gaussian_nll_grad_matches_torch_autograd() -> None:
    u, v, y = _random_uvy(seed=0)
    grad_u, grad_v, _fisher = gaussian_nll_grad_hess(u, v, y)

    ut = torch.tensor(u, dtype=torch.float64, requires_grad=True)
    vt = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.float64)
    nll = gaussian_nll_elementwise(ut, vt, yt).sum()
    (gu_ad,) = torch.autograd.grad(nll, ut, retain_graph=True)
    (gv_ad,) = torch.autograd.grad(nll, vt)

    assert np.allclose(grad_u, gu_ad.detach().numpy(), atol=1e-11, rtol=1e-9)
    assert np.allclose(grad_v, gv_ad.detach().numpy(), atol=1e-11, rtol=1e-9)


def test_gaussian_nll_numpy_matches_torch_twin() -> None:
    u, v, y = _random_uvy(seed=1)
    nll_np = gaussian_nll(u, v, y)
    nll_t = gaussian_nll_elementwise(
        torch.tensor(u, dtype=torch.float64),
        torch.tensor(v, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64),
    )
    assert np.allclose(nll_np, nll_t.numpy(), atol=1e-12, rtol=1e-10)


def test_fisher_is_positive_definite() -> None:
    u, v, y = _random_uvy(seed=2, n=64)
    _gu, _gv, fisher = gaussian_nll_grad_hess(u, v, y)
    assert fisher.shape == (64, 2, 2)
    eigvals = np.linalg.eigvalsh(fisher)
    assert np.all(eigvals > 0.0)


def test_fisher_matches_diag_1_over_s2_and_2() -> None:
    u, v, y = _random_uvy(seed=3, n=32)
    _gu, _gv, fisher = gaussian_nll_grad_hess(u, v, y)
    s2 = np.exp(2.0 * v)
    assert np.allclose(fisher[..., 0, 0], 1.0 / s2)
    assert np.allclose(fisher[..., 1, 1], 2.0)
    assert np.allclose(fisher[..., 0, 1], 0.0)
    assert np.allclose(fisher[..., 1, 0], 0.0)


def test_log_scale_from_variance_roundtrip() -> None:
    v = np.array([-1.0, 0.0, 0.3, 2.0])
    variance = np.exp(2.0 * v)
    assert np.allclose(log_scale_from_variance(variance), v, atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. G0: fit_noise_aware(lam=0.0) is bit-identical to fit_second_order.       #
# --------------------------------------------------------------------------- #


def _regression_problem(seed: int = 0, n: int = 120, d: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.05 * rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize("optimizer", ["trust_region", "cubic"])
def test_g0_lam_zero_is_bit_identical_to_fit_second_order(optimizer: str) -> None:
    X, y = _regression_problem(seed=0)
    cfg = SoftTreeConfig(
        n_features=5, n_trees=4, depth=1, task="regression", n_outputs=1, beta_final=4.0, seed=1
    )
    log_scale = np.zeros(X.shape[0])  # any finite array -- lam=0 must ignore it exactly

    torch.manual_seed(0)
    m_base = SoftTreeEnsemble(cfg)
    r_base = fit_second_order(m_base, X, y, optimizer=optimizer, steps=10, anneal=False)

    torch.manual_seed(0)
    m_noise = SoftTreeEnsemble(cfg)
    r_noise = fit_noise_aware(
        m_noise, X, y, log_scale=log_scale, lam=0.0, optimizer=optimizer, steps=10, anneal=False
    )

    assert r_base.history == r_noise.history  # exact float equality, not approx
    assert torch.equal(m_base.W, m_noise.W)
    assert torch.equal(m_base.t, m_noise.t)
    assert torch.equal(m_base.leaves, m_noise.leaves)
    assert torch.equal(m_base.b0, m_noise.b0)


def test_g0_holds_even_with_extreme_log_scale_since_lam_is_exactly_zero() -> None:
    r"""``lam=0.0`` must zero the penalty even when ``s**2`` is huge -- ``0.0 * finite ==
    0.0`` exactly in IEEE float64, so G0 does not depend on ``log_scale`` being tame."""
    X, y = _regression_problem(seed=4)
    cfg = SoftTreeConfig(
        n_features=5, n_trees=3, depth=1, task="regression", n_outputs=1, beta_final=4.0, seed=2
    )
    extreme_log_scale = np.full(X.shape[0], 20.0)  # s**2 = exp(40) -- large but finite

    torch.manual_seed(0)
    m_base = SoftTreeEnsemble(cfg)
    r_base = fit_second_order(m_base, X, y, optimizer="trust_region", steps=6, anneal=False)

    torch.manual_seed(0)
    m_noise = SoftTreeEnsemble(cfg)
    r_noise = fit_noise_aware(
        m_noise, X, y, log_scale=extreme_log_scale, lam=0.0,
        optimizer="trust_region", steps=6, anneal=False,
    )
    assert r_base.history == r_noise.history


# --------------------------------------------------------------------------- #
# 3. G1b: the penalized closure's Hessian is H + lam * Sigma_noise exactly.   #
# --------------------------------------------------------------------------- #


def _flat_params(model: SoftTreeEnsemble) -> tuple[torch.Tensor, list[str], list[torch.Size]]:
    names = [n for n, _ in model.named_parameters()]
    shapes = [p.shape for _, p in model.named_parameters()]
    theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    return theta0, names, shapes


def _make_flat_forward(model: SoftTreeEnsemble, Xt: torch.Tensor, names, shapes):
    from torch.func import functional_call

    numels = [int(np.prod(sh)) for sh in shapes]

    def flat_forward(theta: torch.Tensor) -> torch.Tensor:
        pieces = {}
        idx = 0
        for name, sh, k in zip(names, shapes, numels, strict=True):
            pieces[name] = theta[idx : idx + k].reshape(sh)
            idx += k
        out = functional_call(model, pieces, (Xt,))
        return out.reshape(-1)  # (n,), n_outputs == 1

    return flat_forward


def test_g1b_penalized_hessian_equals_base_hessian_plus_lam_sigma_noise() -> None:
    n, d = 14, 3
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, d))
    y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(n)
    cfg = SoftTreeConfig(
        n_features=d, n_trees=2, depth=1, task="regression", n_outputs=1, beta_final=3.0, seed=5
    )
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    model.set_beta(cfg.beta_final)

    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    log_scale = rng.normal(0.0, 0.4, size=n)
    s2 = torch.tensor(np.exp(2.0 * log_scale), dtype=torch.float64)
    lam = 2.5

    theta0, names, shapes = _flat_params(model)
    flat_forward = _make_flat_forward(model, Xt, names, shapes)

    def base_loss(theta: torch.Tensor) -> torch.Tensor:
        F = flat_forward(theta)
        return ((F - yt) ** 2).mean()

    def penalized_loss(theta: torch.Tensor) -> torch.Tensor:
        F = flat_forward(theta)
        base = ((F - yt) ** 2).mean()
        resid = F - F.detach()
        penalty = (lam / n) * (s2 * resid * resid).sum()
        return base + penalty

    H0 = torch.func.hessian(base_loss)(theta0)
    H1 = torch.func.hessian(penalized_loss)(theta0)

    # Sigma_noise = (2/n) sum_i s_i**2 g_i g_i^T ,  g_i = dF_i/dtheta  (the (n, P) Jacobian).
    J = torch.func.jacrev(flat_forward)(theta0)  # (n, P)
    sigma_noise = (2.0 / n) * (J * s2[:, None]).T @ J

    diff = H1 - H0
    expected = lam * sigma_noise
    assert diff.shape == expected.shape
    assert torch.allclose(diff, expected, atol=1e-9, rtol=1e-6)
    # the penalty is a positive proximal term away from theta0, hence PSD
    assert torch.linalg.eigvalsh(sigma_noise).min() >= -1e-9


def test_g1b_penalty_contributes_zero_to_gradient_at_theta0() -> None:
    r"""The zero-gradient half of the claim: ``d(penalty)/d(theta)|theta0 == 0`` exactly."""
    n, d = 10, 2
    rng = np.random.default_rng(8)
    X = rng.standard_normal((n, d))
    y = rng.standard_normal(n)
    cfg = SoftTreeConfig(n_features=d, n_trees=2, depth=1, task="regression", seed=6)
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    model.set_beta(cfg.beta_final)
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    log_scale = rng.normal(0.0, 0.3, size=n)
    s2 = torch.tensor(np.exp(2.0 * log_scale), dtype=torch.float64)
    lam = 1.7

    F = model(Xt)
    base = ((F.reshape(-1) - yt) ** 2).mean()
    resid = F.reshape(-1) - F.reshape(-1).detach()
    penalty = (lam / n) * (s2 * resid * resid).sum()
    loss = base + penalty

    assert float(penalty.detach()) == 0.0
    g_loss = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    g_base = torch.autograd.grad(base, list(model.parameters()))
    for gl, gb in zip(g_loss, g_base, strict=True):
        assert torch.allclose(gl, gb, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4. Smokes: the trainers actually run and reduce their objective.           #
# --------------------------------------------------------------------------- #


def test_fit_noise_aware_lam_positive_runs_and_reduces_loss() -> None:
    X, y = _regression_problem(seed=9, n=150)
    cfg = SoftTreeConfig(n_features=5, n_trees=6, depth=1, task="regression", beta_final=4.0, seed=3)
    log_scale = np.zeros(X.shape[0])
    torch.manual_seed(0)
    model = SoftTreeEnsemble(cfg)
    r = fit_noise_aware(
        model, X, y, log_scale=log_scale, lam=1.5, optimizer="trust_region", steps=20, anneal=False
    )
    assert r.history[-1] < r.history[0]
    assert np.isfinite(r.train_loss)


def test_fit_heteroscedastic_runs_and_reduces_nll() -> None:
    rng = np.random.default_rng(10)
    n, d = 160, 4
    X = rng.standard_normal((n, d))
    s_true = 0.3 + 0.2 * np.abs(X[:, 0])
    y = X[:, 0] * 1.1 + s_true * rng.standard_normal(n)
    cfg = SoftTreeConfig(n_features=d, n_trees=4, depth=1, task="regression", beta_final=4.0, seed=1)
    torch.manual_seed(0)
    model = HeteroscedasticHead(cfg)
    r = fit_heteroscedastic(model, X, y, optimizer="trust_region", steps=20, anneal=False)
    assert r.history[-1] < r.history[0]  # NLL decreases
    assert np.isfinite(r.train_loss)


def test_fit_noise_aware_rejects_classification_task() -> None:
    cfg = SoftTreeConfig(n_features=3, n_trees=2, depth=1, task="binary", seed=0)
    model = SoftTreeEnsemble(cfg)
    X = np.zeros((5, 3))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="regression-only"):
        fit_noise_aware(model, X, y, log_scale=np.zeros(5), lam=1.0)


def test_fit_noise_aware_rejects_wrong_length_log_scale() -> None:
    cfg = SoftTreeConfig(n_features=3, n_trees=2, depth=1, task="regression", seed=0)
    model = SoftTreeEnsemble(cfg)
    X = np.zeros((5, 3))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="length"):
        fit_noise_aware(model, X, y, log_scale=np.zeros(3), lam=1.0)


def test_fit_noise_aware_rejects_kfac() -> None:
    cfg = SoftTreeConfig(n_features=3, n_trees=2, depth=1, task="regression", seed=0)
    model = SoftTreeEnsemble(cfg)
    X = np.zeros((5, 3))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="trust_region"):
        fit_noise_aware(model, X, y, log_scale=np.zeros(5), lam=1.0, optimizer="kfac")


def test_heteroscedastic_head_rejects_non_regression() -> None:
    cfg = SoftTreeConfig(n_features=3, n_trees=2, depth=1, task="binary", seed=0)
    with pytest.raises(ValueError, match="regression-only"):
        HeteroscedasticHead(cfg)


# --------------------------------------------------------------------------- #
# 5. bench.py: the new synthetic generators + the frozen CatBoost estimator. #
# --------------------------------------------------------------------------- #


def test_saw_wave_2d_matches_known_formula() -> None:
    ds = saw_wave_2d(500, seed=0)
    assert ds.task == "regression" and ds.n_outputs == 1
    assert ds.f_clean is not None and ds.s_true is not None
    x1, x2 = ds.X[:, 0], ds.X[:, 1]
    assert np.all((x1 >= 0.0) & (x1 <= 1.0))
    assert np.all((x2 >= 0.0) & (x2 <= 10.0))
    assert np.array_equal(ds.f_clean, saw_wave_f(x1, x2))
    assert np.allclose(ds.s_true, saw_wave_s(x2))
    assert set(np.unique(ds.f_clean).tolist()).issubset({0.0, 1.0})
    # y - f_clean is s_true * N(0, 1) -- the standardized residual should be ~unit scale
    # wherever the noise is non-negligible (low-x2 rows have s_true ~ 0 and are excluded).
    informative = ds.s_true > 1e-3
    standardized = (ds.y[informative] - ds.f_clean[informative]) / ds.s_true[informative]
    assert 0.7 < np.std(standardized) < 1.4


def test_saw_wave_s_grows_with_x2() -> None:
    x2 = np.array([0.0, 2.0, 5.0, 10.0])
    s = saw_wave_s(x2)
    assert s[0] == 0.0
    assert s[1] == pytest.approx(2.0**6 / 62500.0)
    assert np.all(np.diff(s) > 0.0)  # strictly increasing noise with x2


def test_mlp_heteroscedastic_20d_shape_and_positive_noise() -> None:
    ds = mlp_heteroscedastic_20d(300, seed=0)
    assert ds.X.shape == (300, 20)
    assert ds.f_clean is not None and ds.s_true is not None
    assert np.all(ds.s_true > 0.0)  # s = exp(g(x)) is always positive


def test_fit_predict_catboost_uncertainty_correlates_with_true_noise() -> None:
    pytest.importorskip("catboost")
    ds = saw_wave_2d(1500, seed=0)
    assert ds.s_true is not None
    Xtr, Xte = ds.X[:1000], ds.X[1000:]
    ytr = ds.y[:1000]
    ste = ds.s_true[1000:]
    _mean, log_scale = fit_predict_catboost_uncertainty(Xtr, ytr, Xte, seed=0, iterations=150)
    assert log_scale.shape == (Xte.shape[0],)
    from scipy.stats import spearmanr

    rho, _ = spearmanr(np.exp(log_scale), ste)
    assert rho > 0.5  # a weak but real floor -- reference-validity sanity, not gate G1
