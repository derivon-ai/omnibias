# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Contract + behavioural tests for :mod:`omnibias.curvature.torch.sharpness`.

Validates the multi-layer, matrix-free torch curvature stack:

1. **Exactness** -- HVP equals the dense Hessian action; ``dense_hessian``
   equals ``torch.func.hessian`` of the loss; power iteration recovers the
   dense top eigenvalue (including an indefinite Hessian); Hutchinson
   estimators are unbiased; and the differentiable penalty gradient (a
   reverse-over-reverse-over-reverse pass that rides on the closed-form
   ``sigma'''``) matches a central finite difference.
2. **The SAM surrogate is right** -- the exact second-order gap upper-bounds
   the sampled worst case in a parameter ball.
3. **It does what it promises** -- curvature-regularised training reaches a
   strictly flatter minimum than plain MSE on a multi-layer net.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.curvature.torch import sharpness as S  # noqa: E402
from omnibias.torch.architectures import JetMLP  # noqa: E402

torch.manual_seed(0)


def _small_problem(in_dim=2, hidden=3, depth=2, n=16, seed=0):
    torch.manual_seed(seed)
    net = JetMLP(in_dim, hidden, 1, depth=depth, base="tanh").double()
    X = torch.randn(n, in_dim, dtype=torch.float64)
    Y = torch.randn(n, dtype=torch.float64)
    params = [p for p in net.parameters() if p.requires_grad]
    return net, X, Y, params


def _mse(net, X, Y):
    return ((net(X).squeeze(-1) - Y) ** 2).mean()


def _flat_loss_fn(net, X, Y):
    """A differentiable loss as a function of a flat parameter vector, via
    ``functional_call`` (unlike ``vector_to_parameters``, this keeps the graph)."""
    from torch.func import functional_call
    names = [n for n, _ in net.named_parameters()]
    shapes = [p.shape for _, p in net.named_parameters()]
    numels = [p.numel() for _, p in net.named_parameters()]

    def flat_loss(theta):
        pieces = {}
        idx = 0
        for name, sh, k in zip(names, shapes, numels, strict=True):
            pieces[name] = theta[idx:idx + k].reshape(sh)
            idx += k
        out = functional_call(net, pieces, (X,)).squeeze(-1)
        return ((out - Y) ** 2).mean()

    theta0 = torch.cat([p.detach().reshape(-1) for p in net.parameters()])
    return flat_loss, theta0


# ---------------------------------------------------------------------------
# 1. Exactness
# ---------------------------------------------------------------------------


def test_dense_hessian_matches_torch_func_hessian():
    net, X, Y, params = _small_problem()
    Hd = S.dense_hessian(_mse(net, X, Y), params)
    flat_loss, theta0 = _flat_loss_fn(net, X, Y)
    Ho = torch.func.hessian(flat_loss)(theta0)
    assert Hd.shape == Ho.shape == (theta0.numel(), theta0.numel())
    assert torch.allclose(Hd, Hd.T, atol=1e-9), "dense Hessian not symmetric"
    assert float((Hd - Ho).abs().max()) < 1e-7


def test_hvp_matches_dense_action():
    net, X, Y, params = _small_problem(seed=1)
    Hd = S.dense_hessian(_mse(net, X, Y), params)
    v = S._rand_like(params, generator=torch.Generator().manual_seed(3))
    Hv = S.hvp(_mse(net, X, Y), params, v)
    Hv_flat = torch.cat([h.reshape(-1) for h in Hv]).detach()
    v_flat = torch.cat([vi.reshape(-1) for vi in v])
    assert float((Hv_flat - Hd @ v_flat).abs().max()) < 1e-10


def test_power_iteration_matches_dense_top_eigenvalue():
    net, X, Y, params = _small_problem(seed=2)
    Hd = S.dense_hessian(_mse(net, X, Y), params)
    ev = torch.linalg.eigvalsh(0.5 * (Hd + Hd.T))
    lam = float(S.top_eigenvalue(_mse(net, X, Y), params, iters=80,
                                 generator=torch.Generator().manual_seed(0)))
    assert abs(lam - float(ev[-1])) < 1e-5
    lo, hi = S.hessian_eigenvalue_extremes(_mse(net, X, Y), params, iters=120,
                                           generator=torch.Generator().manual_seed(0))
    assert abs(hi - float(ev[-1])) < 1e-5
    assert abs(lo - float(ev[0])) < 1e-3  # min converges slower; looser


def test_power_iteration_handles_indefinite_hessian():
    """Away from a minimum the Hessian is indefinite; the two-phase power
    iteration must still return the algebraically largest eigenvalue."""
    net, X, Y, params = _small_problem(seed=5)
    # Perturb params so the loss landscape is indefinite here.
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn_like(p))
    Hd = S.dense_hessian(_mse(net, X, Y), params)
    ev = torch.linalg.eigvalsh(0.5 * (Hd + Hd.T))
    assert float(ev[0]) < 0 < float(ev[-1]), "test setup not indefinite"
    lam = float(S.top_eigenvalue(_mse(net, X, Y), params, iters=120,
                                 generator=torch.Generator().manual_seed(1)))
    assert abs(lam - float(ev[-1])) < 1e-3


def test_hutchinson_trace_and_frobenius_unbiased():
    net, X, Y, params = _small_problem(seed=4)
    Hd = S.dense_hessian(_mse(net, X, Y), params)
    tr_exact = float(torch.trace(Hd))
    fr_exact = float((Hd * Hd).sum())
    gen = torch.Generator().manual_seed(0)
    tr_est = float(S.hutchinson_trace(_mse(net, X, Y), params, n_samples=2000, generator=gen))
    fr_est = float(S.hutchinson_frobenius_sq(_mse(net, X, Y), params, n_samples=2000, generator=gen))
    # Unbiased estimators; at 2000 probes (seed-fixed => deterministic) the
    # relative error is well under 5%.
    assert abs(tr_est - tr_exact) < 0.05 * (1 + abs(tr_exact))
    assert abs(fr_est - fr_exact) < 0.05 * (1 + abs(fr_exact))


def test_sigma_third_derivative_exact_in_torch():
    """d/dz sigma''(z) == sigma'''(z) in closed form -- the mechanism the
    sharpness-penalty gradient rides on."""
    from omnibias.torch import get_activation
    for act in ("tanh", "sigmoid", "softplus", "gaussian", "exp"):
        spec = get_activation(act)
        z = torch.linspace(-2.0, 2.0, 9, dtype=torch.float64, requires_grad=True)
        sigma_pp = spec.fastpath(z, 2)
        (d_pp,) = torch.autograd.grad(sigma_pp.sum(), z, create_graph=False)
        sigma_ppp = spec.fastpath(z, 3)
        assert torch.allclose(d_pp, sigma_ppp, atol=1e-9), f"sigma''' mismatch for {act}"


@pytest.mark.parametrize("measure", ["trace", "frobenius"])
def test_penalty_gradient_matches_finite_difference(measure):
    """The differentiable sharpness penalty gradient equals a central finite
    difference of the same (fixed-probe) estimator -- exact triple-backward."""
    net, X, Y, params = _small_problem(seed=7)
    seed = 123

    def pen_value():
        gen = torch.Generator().manual_seed(seed)
        return float(S.curvature_sharpness(_mse(net, X, Y), params, measure=measure,
                                           n_samples=4, generator=gen, differentiable=False))

    gen = torch.Generator().manual_seed(seed)
    pen = S.curvature_sharpness(_mse(net, X, Y), params, measure=measure,
                               n_samples=4, generator=gen, differentiable=True)
    net.zero_grad()
    pen.backward()
    grads = [p.grad.detach().clone() for p in params]

    eps = 1e-6
    # check a couple of coordinates in the first and last parameter tensors
    for k in (0, len(params) - 1):
        flat = params[k].reshape(-1)
        for j in (0, flat.numel() - 1):
            with torch.no_grad():
                flat[j] += eps
            plus = pen_value()
            with torch.no_grad():
                flat[j] -= 2 * eps
            minus = pen_value()
            with torch.no_grad():
                flat[j] += eps
            fd = (plus - minus) / (2 * eps)
            ad = float(grads[k].reshape(-1)[j])
            assert abs(fd - ad) <= 1e-3 * (1 + abs(ad)), (
                f"{measure} param {k} coord {j}: fd={fd} ad={ad}"
            )


def test_unknown_measure_raises():
    net, X, Y, params = _small_problem()
    with pytest.raises(ValueError, match="unknown sharpness measure"):
        S.curvature_sharpness(_mse(net, X, Y), params, measure="nope")


def test_parity_torch_hvp_vs_jax_closed_form_one_layer():
    """The torch matrix-free Hessian and the JAX *closed-form* one-layer
    Hessian describe the same field -- their spectra / trace / Frobenius match
    to float precision (bit-identical-by-construction, across backends)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.curvature.sharpness import mse_loss_hessian as jax_loss_hessian

    rng = np.random.default_rng(0)
    D, H, B = 3, 4, 8
    Wn = rng.normal(scale=0.3, size=(H, D))
    bn = rng.normal(scale=0.2, size=(H,))
    cn = rng.normal(scale=0.4, size=(H,))
    b0 = 0.13
    Xn = rng.normal(size=(B, D))
    Yn = rng.normal(size=(B,))

    Hj, _ = jax_loss_hessian(
        jnp.asarray(Xn), jnp.asarray(Yn), jnp.asarray(Wn), jnp.asarray(bn),
        jnp.asarray(cn), jnp.asarray(b0), "tanh",
    )
    Hj = np.asarray(0.5 * (Hj + Hj.T))
    ev_jax = np.sort(np.linalg.eigvalsh(Hj))

    W = torch.tensor(Wn, requires_grad=True)
    beta = torch.tensor(bn, requires_grad=True)
    c = torch.tensor(cn, requires_grad=True)
    b = torch.tensor(b0, requires_grad=True)
    X = torch.tensor(Xn)
    Y = torch.tensor(Yn)

    def loss():
        f = b + torch.tanh(X @ W.T + beta) @ c
        return ((f - Y) ** 2).mean()

    Ht = S.dense_hessian(loss(), [W, beta, c, b])
    ev_torch = np.sort(np.linalg.eigvalsh(np.asarray((0.5 * (Ht + Ht.T)).detach())))

    tr_j, tr_t = float(np.trace(Hj)), float(torch.trace(Ht))
    fr_j, fr_t = float((Hj * Hj).sum()), float((Ht * Ht).sum())
    assert np.allclose(ev_jax, ev_torch, atol=1e-8), (ev_jax, ev_torch)
    assert abs(tr_j - tr_t) <= 1e-8 * (1 + abs(tr_j)), (tr_j, tr_t)
    assert abs(fr_j - fr_t) <= 1e-8 * (1 + abs(fr_j)), (fr_j, fr_t)


# ---------------------------------------------------------------------------
# 2. The SAM surrogate is right
# ---------------------------------------------------------------------------


def test_sam_gap_upper_bounds_sampled_worst_case():
    net, X, Y, params = _small_problem(in_dim=1, hidden=2, depth=1, n=8, seed=9)
    rho = 0.08
    gap2 = float(S.sam_sharpness_gap(_mse(net, X, Y), params, rho=rho, iters=100,
                                     generator=torch.Generator().manual_seed(0)))
    assert gap2 > 0.0

    rng = np.random.default_rng(0)
    flats = [p.reshape(-1) for p in params]
    total = sum(f.numel() for f in flats)
    worst = 0.0
    with torch.no_grad():
        L0 = float(_mse(net, X, Y))
        for _ in range(1500):
            d = rng.normal(size=total)
            d = rho * d / np.linalg.norm(d)
            d_t = torch.tensor(d, dtype=torch.float64)
            idx = 0
            for f in flats:
                k = f.numel()
                f += d_t[idx:idx + k]
                idx += k
            worst = max(worst, float(_mse(net, X, Y)) - L0)
            idx = 0
            for f in flats:
                k = f.numel()
                f -= d_t[idx:idx + k]
                idx += k
    assert gap2 >= worst - 1e-3, f"gap2={gap2} under-shoots sampled worst={worst}"
    assert gap2 <= 2.0 * worst + 1e-6


# ---------------------------------------------------------------------------
# 3. It does what it promises (multi-layer)
# ---------------------------------------------------------------------------


def test_all_measures_finite_on_deep_net():
    net, X, Y, params = _small_problem(in_dim=3, hidden=8, depth=3, n=24, seed=11)
    P = sum(p.numel() for p in params)
    assert P > 100  # genuinely multi-layer / larger than one-layer
    for measure in ("trace", "frobenius", "top_eig"):
        val = S.curvature_sharpness(_mse(net, X, Y), params, measure=measure,
                                    n_samples=8, iters=30,
                                    generator=torch.Generator().manual_seed(0))
        assert torch.isfinite(val)


def _teacher_student(seed, D, H_teacher, n_train, noise):
    torch.manual_seed(seed)
    teacher = JetMLP(D, H_teacher, 1, depth=2, base="tanh").double()
    Xtr = torch.randn(n_train, D, dtype=torch.float64)
    Xte = torch.randn(300, D, dtype=torch.float64)
    with torch.no_grad():
        Ytr = teacher(Xtr).squeeze(-1) + noise * torch.randn(n_train, dtype=torch.float64)
        Yte = teacher(Xte).squeeze(-1)
    return Xtr, Ytr, Xte, Yte


def _train(net, Xtr, Ytr, *, lam, steps, lr, seed):
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        opt.zero_grad()
        loss = _mse(net, Xtr, Ytr)
        if lam > 0.0:
            obj = S.sharpness_aware_loss(loss, params, lam=lam,
                                         measure="frobenius", n_samples=2, generator=gen)
        else:
            obj = loss
        obj.backward()
        opt.step()
    return params


def test_sharpness_aware_training_finds_flatter_minimum():
    D, H_teacher, H_student, n_train = 3, 3, 6, 30
    Xtr, Ytr, Xte, Yte = _teacher_student(17, D, H_teacher, n_train, noise=0.1)

    torch.manual_seed(100)
    net_plain = JetMLP(D, H_student, 1, depth=2, base="tanh").double()
    net_flat = JetMLP(D, H_student, 1, depth=2, base="tanh").double()
    net_flat.load_state_dict(net_plain.state_dict())  # same init

    p_plain = _train(net_plain, Xtr, Ytr, lam=0.0, steps=200, lr=8e-3, seed=1)
    p_flat = _train(net_flat, Xtr, Ytr, lam=3e-3, steps=200, lr=8e-3, seed=1)

    def report(net, params):
        tr = float(_mse(net, Xtr, Ytr).detach())
        te = float(_mse(net, Xte, Yte).detach())
        curv = float(S.top_eigenvalue(_mse(net, Xtr, Ytr), params, iters=60,
                                      generator=torch.Generator().manual_seed(0)))
        return tr, te, curv

    tr_p, te_p, curv_p = report(net_plain, p_plain)
    tr_f, te_f, curv_f = report(net_flat, p_flat)

    assert tr_p < 0.05 and tr_f < 0.08, (tr_p, tr_f)
    assert curv_f < 0.75 * curv_p, f"curv flat={curv_f:.3f} vs plain={curv_p:.3f}"
    # exact-curvature regularisation also improved generalisation here
    assert te_f <= te_p * 1.05, f"test mse flat={te_f:.4f} vs plain={te_p:.4f}"
