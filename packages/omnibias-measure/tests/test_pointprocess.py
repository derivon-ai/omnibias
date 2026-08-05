# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Temporal-point-process / survival likelihoods on the measure integral.

Validates against analytic oracles (homogeneous Poisson, exponential /
polynomial-Weibull hazards), Gauss-Legendre exactness on polynomial intensities,
the closed-form antiderivative compensator vs quadrature, cross-backend parity,
end-to-end autograd (train-through), an accuracy-vs-coarse-Riemann gate, and the
error paths.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

from omnibias.measure.jax import pointprocess as J  # noqa: E402
from omnibias.measure.torch import pointprocess as T  # noqa: E402


# --------------------------------------------------------------------------- #
# Closed-form antiderivative compensator (the omnibias-native route)
# --------------------------------------------------------------------------- #
def test_closed_form_compensator_exp_matches_analytic() -> None:
    w, b, t0, t1 = 0.7, -0.3, 0.0, 2.5
    ana = (math.exp(w * t1 + b) - math.exp(w * t0 + b)) / w
    cf = float(T.closed_form_compensator("exp", w, b, t0, t1))
    assert cf == pytest.approx(ana, rel=1e-12)


def test_closed_form_compensator_sigmoid_matches_quadrature() -> None:
    # sigma's antiderivative is softplus; check the exact window vs a high-order
    # Gauss-Legendre quadrature of the same intensity 2*sigmoid(1.3 t - 0.5).
    cf = float(T.closed_form_compensator("sigmoid", 1.3, -0.5, 0.0, 3.0, scale=2.0))
    q = float(T.compensator(lambda t: 2.0 * torch.sigmoid(1.3 * t - 0.5), 0.0, 3.0, num=128))
    assert cf == pytest.approx(q, rel=1e-10)


def test_closed_form_compensator_gradients_autodiff_exact() -> None:
    w = torch.tensor(0.9, requires_grad=True)
    b = torch.tensor(-0.2, requires_grad=True)
    scale = torch.tensor(1.5, requires_grad=True)
    out = T.closed_form_compensator("sigmoid", w, b, 0.0, 2.0, scale=scale)
    out.backward()
    for g in (w.grad, b.grad, scale.grad):
        assert g is not None and torch.isfinite(g).all()


# --------------------------------------------------------------------------- #
# Quadrature compensator exactness
# --------------------------------------------------------------------------- #
def test_compensator_gauss_legendre_exact_on_polynomial() -> None:
    # 3 t^2 + 2 t + 1 integrates to t^3 + t^2 + t; GL is exact for polynomials.
    val = float(T.compensator(lambda t: 3.0 * t**2 + 2.0 * t + 1.0, 0.0, 2.0, num=8))
    assert val == pytest.approx(2.0**3 + 2.0**2 + 2.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# Poisson process likelihood vs analytic
# --------------------------------------------------------------------------- #
def test_poisson_nll_homogeneous_matches_analytic() -> None:
    mu, horizon = 1.7, 4.0
    events = torch.tensor([0.3, 1.1, 2.7, 3.5])
    nll = float(T.poisson_nll(lambda t: mu * torch.ones_like(t), events, 0.0, horizon, num=32))
    ana = mu * horizon - len(events) * math.log(mu)
    assert nll == pytest.approx(ana, rel=1e-10)


def test_log_likelihood_is_negative_nll() -> None:
    class Lam(torch.nn.Module):
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(0.5 * t + 0.1)

    tpp = T.TemporalPointProcess(Lam(), num=48)
    ev = torch.tensor([0.2, 0.9, 1.8])
    assert float(tpp.log_likelihood(ev, 0.0, 2.0)) == pytest.approx(
        -float(tpp.nll(ev, 0.0, 2.0)), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# Survival / hazard likelihood vs analytic
# --------------------------------------------------------------------------- #
def test_survival_nll_exponential_matches_analytic() -> None:
    lam = 0.9
    d = torch.tensor([0.5, 1.2, 2.0, 0.8])
    obs = torch.tensor([1.0, 0.0, 1.0, 1.0])
    snll = float(T.survival_nll(lambda t: lam * torch.ones_like(t), d, obs, num=16))
    ana = -float((obs * math.log(lam) - lam * d).sum())
    assert snll == pytest.approx(ana, rel=1e-10)


def test_survival_nll_polynomial_weibull_matches_analytic() -> None:
    # Weibull shape k=2: hazard h(t)=2 t, cumulative H(t)=t^2 (polynomial -> GL exact).
    d = torch.tensor([0.5, 1.5, 2.0])
    obs = torch.tensor([1.0, 1.0, 0.0])
    snll = float(T.survival_nll(lambda t: 2.0 * t, d, obs, num=16))
    log_h = torch.log(2.0 * d)
    ana = -float((obs * log_h - d**2).sum())
    assert snll == pytest.approx(ana, rel=1e-10)


# --------------------------------------------------------------------------- #
# Cross-backend parity (torch <-> jax)
# --------------------------------------------------------------------------- #
def test_closed_form_and_quadrature_parity_backends() -> None:
    ct = float(T.closed_form_compensator("sigmoid", 1.3, -0.5, 0.0, 3.0, scale=2.0))
    cj = float(J.closed_form_compensator("sigmoid", 1.3, -0.5, 0.0, 3.0, scale=2.0))
    assert ct == pytest.approx(cj, rel=1e-12)
    qt = float(T.compensator(lambda t: 2.0 * torch.sigmoid(1.3 * t - 0.5), 0.0, 3.0, num=96))
    qj = float(J.compensator(lambda t: 2.0 * jax.nn.sigmoid(1.3 * t - 0.5), 0.0, 3.0, num=96))
    assert qt == pytest.approx(qj, rel=1e-12)


def test_poisson_and_survival_nll_parity_backends() -> None:
    ev_t, ev_j = torch.tensor([0.3, 1.1, 2.7]), jnp.array([0.3, 1.1, 2.7])
    nt = float(T.poisson_nll(lambda t: 2.0 * torch.sigmoid(1.3 * t - 0.5), ev_t, 0.0, 3.0, num=96))
    nj = float(J.poisson_nll(lambda t: 2.0 * jax.nn.sigmoid(1.3 * t - 0.5), ev_j, 0.0, 3.0, num=96))
    assert nt == pytest.approx(nj, rel=1e-12)

    d_t, o_t = torch.tensor([0.5, 1.2, 2.0]), torch.tensor([1.0, 0.0, 1.0])
    d_j, o_j = jnp.array([0.5, 1.2, 2.0]), jnp.array([1.0, 0.0, 1.0])
    st = float(T.survival_nll(lambda t: torch.nn.functional.softplus(t - 0.3), d_t, o_t, num=64))
    sj = float(J.survival_nll(lambda t: jax.nn.softplus(t - 0.3), d_j, o_j, num=64))
    assert st == pytest.approx(sj, rel=1e-11)


# --------------------------------------------------------------------------- #
# Autograd (train-through) + parity of the gradient
# --------------------------------------------------------------------------- #
def test_train_through_gradients_parity_backends() -> None:
    ev_t, ev_j = torch.tensor([0.3, 1.1, 2.7]), jnp.array([0.3, 1.1, 2.7])
    theta = torch.tensor([1.0, 0.0], requires_grad=True)
    loss = T.poisson_nll(
        lambda t: torch.nn.functional.softplus(theta[0] * t + theta[1]), ev_t, 0.0, 3.0, num=64
    )
    loss.backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()

    def loss_j(p: jax.Array) -> jax.Array:
        return J.poisson_nll(
            lambda t: jax.nn.softplus(p[0] * t + p[1]), ev_j, 0.0, 3.0, num=64
        )

    g = jax.grad(loss_j)(jnp.array([1.0, 0.0]))
    assert float(jnp.max(jnp.abs(g - jnp.asarray(theta.grad.tolist())))) < 1e-9


def test_temporal_point_process_module_trains() -> None:
    # Fit a log-linear intensity softplus(a t + c) to events drawn from a known
    # increasing rate; train-through NLL must strictly decrease.
    torch.manual_seed(0)

    class Intensity(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(0.0))
            self.c = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.a * t + self.c)

    tpp = T.TemporalPointProcess(Intensity(), num=64)
    events = torch.tensor([0.4, 1.3, 1.9, 2.2, 2.6, 2.8])
    opt = torch.optim.Adam(tpp.parameters(), lr=0.1)
    nll0 = float(tpp.nll(events, 0.0, 3.0).detach())
    for _ in range(60):
        opt.zero_grad()
        loss = tpp(events, 0.0, 3.0)
        loss.backward()
        opt.step()
    nll1 = float(tpp.nll(events, 0.0, 3.0).detach())
    assert nll1 < nll0 - 1e-3
    # events cluster late -> a positive (increasing intensity) is recovered.
    assert float(tpp.intensity.a.detach()) > 0.0


# --------------------------------------------------------------------------- #
# Empirical gate: exact/GL compensator beats a coarse left-Riemann baseline
# --------------------------------------------------------------------------- #
def test_closed_form_beats_coarse_riemann_baseline() -> None:
    # Named baseline: left-Riemann sum (the usual TPP compensator approximation).
    w, b, t0, t1, n = 1.1, -0.4, 0.0, 4.0, 16
    ana = (math.exp(w * t1 + b) - math.exp(w * t0 + b)) / w
    grid = torch.linspace(t0, t1, n + 1)[:-1]
    dx = (t1 - t0) / n
    riemann = float((torch.exp(w * grid + b) * dx).sum())
    gl = float(T.compensator(lambda t: torch.exp(w * t + b), t0, t1, num=n))
    cf = float(T.closed_form_compensator("exp", w, b, t0, t1))
    err_riemann = abs(riemann - ana)
    err_gl = abs(gl - ana)
    err_cf = abs(cf - ana)
    # Gauss-Legendre (same node budget) and the closed form are orders of
    # magnitude more accurate than the coarse Riemann baseline.
    assert err_gl < 1e-6 * err_riemann
    assert err_cf <= 1e-12


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #
def test_error_paths() -> None:
    with pytest.raises(ValueError, match="t0 < t1"):
        T.compensator(lambda t: torch.ones_like(t), 1.0, 0.0)
    with pytest.raises(ValueError, match="gauss_legendre.*monte_carlo"):
        T.compensator(lambda t: torch.ones_like(t), 0.0, 1.0, rule="simpson")
    with pytest.raises(ValueError, match="w != 0"):
        T.closed_form_compensator("sigmoid", 0.0, 0.1, 0.0, 1.0)
    with pytest.raises(ValueError, match="no closed-form antiderivative"):
        T.closed_form_compensator("silu", 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="must match"):
        T.survival_nll(
            lambda t: torch.ones_like(t), torch.tensor([1.0, 2.0]), torch.tensor([1.0])
        )
