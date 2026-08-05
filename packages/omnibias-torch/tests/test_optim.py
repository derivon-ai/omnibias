# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Second-order PINN optimisation (torch): Gauss-Newton / LM + self-adaptive weights.

The headline check is that the exact omnibias residual map lets Gauss-Newton drive a
PINN loss far below what Adam reaches in the same wall-clock regime -- the "energy
natural gradient" behaviour. The rest pins down the LM core (exact on linear least
squares, push-through primal/dual identity, monotone accept/reject) and the gradient-norm
loss balancer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from omnibias.torch.architectures import JetMLP
from omnibias.torch.optim import (
    CubicRegularizedGaussNewton,
    CubicRegularizedNewton,
    GaussNewton,
    GradNormBalancer,
    JetLBFGS,
    cgls,
    conjugate_gradient,
    cubic_regularized_newton_step,
    functional_residual_fn,
    gauss_newton_direction,
    gauss_newton_direction_cg,
    gauss_newton_direction_cgls,
    hvp,
    lanczos_tridiag,
    lstsq_gauss_newton_direction,
    quadrature_loss,
    taylor_line_min,
    weighted_residual_fn,
)
from torch.func import jacrev
from torch.nn.utils import vector_to_parameters


def _gauss_legendre_01(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes/weights mapped to [0, 1] (numpy primitive, no fields dep)."""
    t, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (t + 1.0)
    return (
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(0.5 * w, dtype=torch.float64),
    )


@pytest.fixture(autouse=True)
def _use_float64() -> object:
    """Exercise the optimiser oracles in double precision, test-locally.

    Scoped to this module so the float64 default never leaks into dtype-sensitive
    suites (e.g. ``test_fastpath_stability``) during collection.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# --- Gauss-Newton direction core -----------------------------------------


def test_gn_direction_solves_linear_least_squares_in_one_step() -> None:
    """For a linear residual r(p)=A p - b, one undamped GN step is the exact lstsq min."""
    torch.manual_seed(0)
    a = torch.randn(20, 5, dtype=torch.float64)
    b = torch.randn(20, dtype=torch.float64)
    p0 = torch.zeros(5, dtype=torch.float64)
    res = a @ p0 - b
    delta = gauss_newton_direction(a, res, damping=1e-10)
    p_star = torch.linalg.lstsq(a, b).solution
    assert torch.allclose(p0 + delta, p_star, atol=1e-6)


def test_gn_direction_primal_dual_pushthrough_identity() -> None:
    """Primal (P<=N) and dual (P>N) forms agree: (JtJ+uI)^-1 Jt = Jt (JJt+uI)^-1."""
    torch.manual_seed(1)
    # over-parameterised: P=12 > N=6 -> function uses the dual form
    jac = torch.randn(6, 12, dtype=torch.float64)
    res = torch.randn(6, dtype=torch.float64)
    mu = 1e-1
    d_impl = gauss_newton_direction(jac, res, damping=mu)
    p = jac.shape[1]
    d_primal = torch.linalg.solve(
        jac.T @ jac + mu * torch.eye(p, dtype=torch.float64), -(jac.T @ res)
    )
    assert torch.allclose(d_impl, d_primal, atol=1e-9)


def test_gn_direction_shape_validation() -> None:
    with pytest.raises(ValueError):
        gauss_newton_direction(torch.zeros(3), torch.zeros(3), 1e-3)
    with pytest.raises(ValueError):
        gauss_newton_direction(torch.zeros(4, 2), torch.zeros(3), 1e-3)


# --- LM optimiser on nonlinear least squares ------------------------------


def test_gauss_newton_recovers_nonlinear_least_squares() -> None:
    """Exponential model fit: LM recovers the true parameters to high precision."""
    t = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    true = torch.tensor([2.0, -0.7], dtype=torch.float64)
    y = true[0] * torch.exp(true[1] * t)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.exp(p[1] * t) - y

    opt = GaussNewton(damping=1e-3)
    p0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    p, history = opt.minimize(residual_fn, p0, steps=30)
    assert history[-1] < 1e-16
    assert torch.allclose(p, true, atol=1e-6)
    # accepted steps never increase the loss
    assert all(history[i + 1] <= history[i] + 1e-18 for i in range(len(history) - 1))


def test_gauss_newton_step_rejects_when_no_improvement() -> None:
    """At a minimiser the step cannot improve, so it is rejected and params are kept."""
    t = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)
    y = 1.5 * torch.exp(-0.3 * t)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.exp(p[1] * t) - y

    p_star = torch.tensor([1.5, -0.3], dtype=torch.float64)
    opt = GaussNewton(damping=1e-6)
    new_p, info = opt.step(residual_fn, p_star)
    assert not info.accepted
    assert torch.allclose(new_p, p_star, atol=0.0)  # unchanged on rejection


# --- PINN: GN beats Adam by orders of magnitude ---------------------------


class _PoissonResidual(nn.Module):
    """Residual of u'' = f, u(0)=u(1)=0 (manufactured u*=sin(pi x)); forward = residual."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = JetMLP(1, hidden, 1, depth=2, base="tanh")

    def forward(self, x_int: torch.Tensor, x_bc: torch.Tensor, f_int: torch.Tensor) -> torch.Tensor:
        _v, _g, h = self.net.value_grad_hessian(x_int)
        res_pde = h[:, 0, 0, 0] - f_int
        u_bc = self.net.value(x_bc).reshape(-1)
        return torch.cat([res_pde, u_bc])


def _poisson_points() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_int = torch.linspace(0.0, 1.0, 40, dtype=torch.float64)[1:-1].reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    f_int = (-math.pi**2) * torch.sin(math.pi * x_int).reshape(-1)
    return x_int, x_bc, f_int


def test_gauss_newton_beats_adam_on_poisson_pinn() -> None:
    x_int, x_bc, f_int = _poisson_points()

    torch.manual_seed(0)
    m_gn = _PoissonResidual(16).double()
    flat0, residual_fn = functional_residual_fn(m_gn, x_int, x_bc, f_int)
    _, history = GaussNewton(damping=1e-2).minimize(residual_fn, flat0, steps=40)
    gn_loss = history[-1]

    torch.manual_seed(0)
    m_adam = _PoissonResidual(16).double()
    adam = torch.optim.Adam(m_adam.parameters(), lr=1e-3)
    for _ in range(800):
        adam.zero_grad()
        r = m_adam(x_int, x_bc, f_int)
        (0.5 * (r**2).mean()).backward()
        adam.step()
    adam_loss = 0.5 * (m_adam(x_int, x_bc, f_int) ** 2).mean().item()

    assert gn_loss < 5e-4
    assert adam_loss > 2e-3
    assert gn_loss < adam_loss / 5.0


def test_functional_residual_fn_matches_module_forward() -> None:
    x_int, x_bc, f_int = _poisson_points()
    torch.manual_seed(3)
    m = _PoissonResidual(8).double()
    flat0, residual_fn = functional_residual_fn(m, x_int, x_bc, f_int)
    assert torch.allclose(residual_fn(flat0), m(x_int, x_bc, f_int), atol=1e-12)
    # writing flat0 back reproduces the same network
    m2 = _PoissonResidual(8).double()
    vector_to_parameters(flat0, m2.parameters())
    assert torch.allclose(m2(x_int, x_bc, f_int), m(x_int, x_bc, f_int), atol=1e-12)


# --- Matrix-free (conjugate-gradient) Gauss-Newton ------------------------


def test_conjugate_gradient_solves_spd_system() -> None:
    """CG recovers the exact solution of a dense SPD system, matrix-free."""
    torch.manual_seed(0)
    mat = torch.randn(12, 12, dtype=torch.float64)
    a = mat @ mat.T + 0.5 * torch.eye(12, dtype=torch.float64)  # SPD
    x_true = torch.randn(12, dtype=torch.float64)
    b = a @ x_true
    x = conjugate_gradient(lambda v: a @ v, b, max_iter=200, tol=1e-14)
    assert torch.allclose(x, x_true, atol=1e-8)


def test_conjugate_gradient_zero_rhs_returns_zero() -> None:
    x = conjugate_gradient(lambda v: v, torch.zeros(5, dtype=torch.float64))
    assert torch.count_nonzero(x).item() == 0


def test_conjugate_gradient_warm_start() -> None:
    """A warm start near the solution still converges to the same point."""
    torch.manual_seed(1)
    mat = torch.randn(8, 8, dtype=torch.float64)
    a = mat @ mat.T + torch.eye(8, dtype=torch.float64)
    x_true = torch.randn(8, dtype=torch.float64)
    b = a @ x_true
    x = conjugate_gradient(lambda v: a @ v, b, max_iter=200, tol=1e-14, x0=x_true + 1e-3)
    assert torch.allclose(x, x_true, atol=1e-8)


def test_gauss_newton_direction_cg_matches_dense() -> None:
    """Matrix-free CG direction equals the dense LM direction (primal form)."""
    torch.manual_seed(0)
    a = torch.randn(30, 8, dtype=torch.float64)
    b = torch.randn(30, dtype=torch.float64)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return torch.tanh(a @ p) - b

    p0 = torch.randn(8, dtype=torch.float64) * 0.1
    mu = 1e-2
    jac = jacrev(residual_fn)(p0)
    res = residual_fn(p0)
    d_dense = gauss_newton_direction(jac, res, mu)
    d_cg = gauss_newton_direction_cg(residual_fn, p0, mu, cg_max_iter=200, cg_tol=1e-12)
    assert torch.allclose(d_dense, d_cg, atol=1e-8)


def test_gauss_newton_cg_recovers_nonlinear_least_squares() -> None:
    """The matrix-free solver reproduces the dense LM fit to machine precision."""
    t = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    true = torch.tensor([2.0, -0.7], dtype=torch.float64)
    y = true[0] * torch.exp(true[1] * t)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.exp(p[1] * t) - y

    opt = GaussNewton(damping=1e-3, solver="cg", cg_tol=1e-12, cg_max_iter=50)
    p0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    p, history = opt.minimize(residual_fn, p0, steps=30)
    assert history[-1] < 1e-16
    assert torch.allclose(p, true, atol=1e-6)


def test_gauss_newton_cg_matches_dense_first_step_on_poisson() -> None:
    """On the over-parameterised PINN (P > N) CG matches the dense dual step."""
    x_int, x_bc, f_int = _poisson_points()
    torch.manual_seed(0)
    m = _PoissonResidual(16).double()
    flat0, residual_fn = functional_residual_fn(m, x_int, x_bc, f_int)
    p_dense, info_d = GaussNewton(damping=1e-2, solver="dense").step(residual_fn, flat0.clone())
    p_cg, info_c = GaussNewton(
        damping=1e-2, solver="cg", cg_tol=1e-12, cg_max_iter=500
    ).step(residual_fn, flat0.clone())
    assert info_d.accepted and info_c.accepted
    assert torch.allclose(p_dense, p_cg, atol=1e-6)


def test_gauss_newton_cg_validation() -> None:
    with pytest.raises(ValueError):
        GaussNewton(solver="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GaussNewton(damping_strategy="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GaussNewton(cg_max_iter=0)
    with pytest.raises(ValueError):
        GaussNewton(cg_tol=0.0)


# --- Ill-conditioning-safe solvers (QR least-squares + CGLS) ---------------


def test_cgls_matches_lstsq_overdetermined() -> None:
    """Undamped CGLS reproduces the QR least-squares solution."""
    torch.manual_seed(0)
    a = torch.randn(30, 8, dtype=torch.float64)
    b = torch.randn(30, dtype=torch.float64)
    x = cgls(lambda v: a @ v, lambda u: a.T @ u, b, damp=0.0, max_iter=100, tol=1e-14)
    x_ref = torch.linalg.lstsq(a, b.unsqueeze(1)).solution.reshape(-1)
    assert torch.allclose(x, x_ref, atol=1e-8)


def test_cgls_damped_matches_normal_equations() -> None:
    """Damped CGLS solves ``(A^T A + damp^2 I) x = A^T b``."""
    torch.manual_seed(1)
    a = torch.randn(20, 6, dtype=torch.float64)
    b = torch.randn(20, dtype=torch.float64)
    lam = 0.3
    x = cgls(lambda v: a @ v, lambda u: a.T @ u, b, damp=math.sqrt(lam), max_iter=100, tol=1e-14)
    x_ref = torch.linalg.solve(a.T @ a + lam * torch.eye(6, dtype=torch.float64), a.T @ b)
    assert torch.allclose(x, x_ref, atol=1e-8)


def test_lstsq_gn_direction_matches_dense_wellconditioned() -> None:
    """On a well-conditioned residual, QR and normal-equations directions agree."""
    torch.manual_seed(0)
    a = torch.randn(30, 8, dtype=torch.float64)
    b = torch.randn(30, dtype=torch.float64)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return torch.tanh(a @ p) - b

    p0 = torch.randn(8, dtype=torch.float64) * 0.1
    mu = 1e-2
    jac = jacrev(residual_fn)(p0)
    res = residual_fn(p0)
    assert torch.allclose(
        gauss_newton_direction(jac, res, mu), lstsq_gauss_newton_direction(jac, res, mu), atol=1e-8
    )


def test_gauss_newton_direction_cgls_matches_dense() -> None:
    """Matrix-free CGLS direction equals the dense LM direction on a benign problem."""
    torch.manual_seed(0)
    a = torch.randn(30, 8, dtype=torch.float64)
    b = torch.randn(30, dtype=torch.float64)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return torch.tanh(a @ p) - b

    p0 = torch.randn(8, dtype=torch.float64) * 0.1
    mu = 1e-2
    jac = jacrev(residual_fn)(p0)
    res = residual_fn(p0)
    d_dense = gauss_newton_direction(jac, res, mu)
    d_cgls = gauss_newton_direction_cgls(residual_fn, p0, mu, cgls_max_iter=200, cgls_tol=1e-12)
    assert torch.allclose(d_dense, d_cgls, atol=1e-8)


def test_qr_beats_normal_equations_on_ill_conditioned_jacobian() -> None:
    """The headline fix: forming ``J^T J`` squares kappa; QR keeps kappa(J).

    Built from a known SVD ``J = U diag(s) V^T`` with ``kappa(J)=1e7`` so the exact damped
    LM step ``V diag(s/(s^2+mu)) U^T b`` is available in closed form. The QR solve stays near
    ``kappa(J)*eps``; the normal-equations solve degrades to ``~kappa(J)^2*eps``.
    """
    torch.manual_seed(0)
    p_dim, n_dim = 6, 20
    u = torch.linalg.qr(torch.randn(n_dim, p_dim, dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(p_dim, p_dim, dtype=torch.float64))[0]
    s = torch.logspace(0, -7, p_dim, dtype=torch.float64)  # kappa(J) = 1e7
    jac = u @ torch.diag(s) @ v.T
    b = torch.randn(n_dim, dtype=torch.float64)
    res0 = -b  # r(0) = J*0 - b
    mu = 1e-14
    x_exact = v @ (s * (u.T @ b) / (s**2 + mu))
    xn = float(torch.linalg.vector_norm(x_exact))
    rel_normal = float(torch.linalg.vector_norm(gauss_newton_direction(jac, res0, mu) - x_exact)) / xn
    rel_qr = float(torch.linalg.vector_norm(lstsq_gauss_newton_direction(jac, res0, mu) - x_exact)) / xn
    assert rel_qr < 1e-6
    assert rel_qr < rel_normal


def test_gauss_newton_qr_recovers_nonlinear_least_squares() -> None:
    t = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    true = torch.tensor([2.0, -0.7], dtype=torch.float64)
    y = true[0] * torch.exp(true[1] * t)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.exp(p[1] * t) - y

    opt = GaussNewton(damping=1e-3, solver="qr")
    p, history = opt.minimize(residual_fn, torch.tensor([1.0, 0.0], dtype=torch.float64), steps=30)
    assert history[-1] < 1e-16
    assert torch.allclose(p, true, atol=1e-6)


def test_gauss_newton_cgls_recovers_nonlinear_least_squares() -> None:
    t = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    true = torch.tensor([2.0, -0.7], dtype=torch.float64)
    y = true[0] * torch.exp(true[1] * t)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.exp(p[1] * t) - y

    opt = GaussNewton(damping=1e-3, solver="cgls", cg_tol=1e-13, cg_max_iter=50)
    p, history = opt.minimize(residual_fn, torch.tensor([1.0, 0.0], dtype=torch.float64), steps=30)
    assert history[-1] < 1e-12
    assert torch.allclose(p, true, atol=1e-5)


# --- Nielsen (gain-ratio trust-region) damping ----------------------------


def test_gauss_newton_nielsen_recovers_nonlinear_least_squares() -> None:
    t = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    true = torch.tensor([2.0, -0.7], dtype=torch.float64)
    y = true[0] * torch.exp(true[1] * t)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.exp(p[1] * t) - y

    opt = GaussNewton(damping=1e-3, damping_strategy="nielsen")
    p, history = opt.minimize(residual_fn, torch.tensor([1.0, 0.0], dtype=torch.float64), steps=30)
    assert history[-1] < 1e-16
    assert torch.allclose(p, true, atol=1e-6)
    assert all(history[i + 1] <= history[i] + 1e-18 for i in range(len(history) - 1))


def test_gauss_newton_nielsen_rejects_at_minimum() -> None:
    t = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)
    y = 1.5 * torch.exp(-0.3 * t)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.exp(p[1] * t) - y

    p_star = torch.tensor([1.5, -0.3], dtype=torch.float64)
    new_p, info = GaussNewton(damping=1e-6, damping_strategy="nielsen").step(residual_fn, p_star)
    assert not info.accepted
    assert torch.allclose(new_p, p_star, atol=0.0)


# --- Integral (function-space) loss + integral-weighted natural gradient ---


def test_quadrature_loss_matches_analytic_integral() -> None:
    r"""Gauss quadrature of sin^2 recovers ``\int_0^1 sin^2(pi x) dx = 1/2`` exactly."""
    x, w = _gauss_legendre_01(32)
    r = torch.sin(math.pi * x)
    assert quadrature_loss(r, w).item() == pytest.approx(0.5, abs=1e-10)


def test_quadrature_loss_validation() -> None:
    with pytest.raises(ValueError):
        quadrature_loss(torch.zeros(3, 1), torch.zeros(3))
    with pytest.raises(ValueError):
        quadrature_loss(torch.zeros(3), torch.zeros(4))


def test_weighted_residual_fn_scaling_and_loss() -> None:
    """``weighted_residual_fn`` is sqrt(w) scaling and its sq-norm is the integral loss."""
    x, w = _gauss_legendre_01(16)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.sin(math.pi * x) + p[1]

    wrapped = weighted_residual_fn(residual_fn, w)
    p = torch.tensor([1.3, -0.2], dtype=torch.float64)
    assert torch.allclose(wrapped(p), residual_fn(p) * torch.sqrt(w), atol=1e-14)
    assert (wrapped(p) ** 2).sum().item() == pytest.approx(
        quadrature_loss(residual_fn(p), w).item(), abs=1e-12
    )


def test_weighted_residual_fn_validation() -> None:
    with pytest.raises(ValueError):
        weighted_residual_fn(lambda v: v, torch.zeros(3, 1))
    with pytest.raises(ValueError):
        weighted_residual_fn(lambda v: v, torch.tensor([-1.0, 1.0], dtype=torch.float64))


def test_integral_weighted_gn_matrix_is_l2_metric() -> None:
    r"""The weighted Gauss-Newton matrix is the discretised L2 metric ``J^T diag(w) J``."""
    x, w = _gauss_legendre_01(12)
    torch.manual_seed(0)
    mat = torch.randn(12, 4, dtype=torch.float64)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mat @ p * x)

    wrapped = weighted_residual_fn(residual_fn, w)
    p0 = torch.randn(4, dtype=torch.float64)
    jac_w = jacrev(wrapped)(p0)
    jac = jacrev(residual_fn)(p0)
    assert torch.allclose(jac_w.T @ jac_w, jac.T @ torch.diag(w) @ jac, atol=1e-10)


# --- Self-adaptive loss weights -------------------------------------------


def test_grad_norm_balancer_equalises_weighted_norms() -> None:
    x_int, x_bc, f_int = _poisson_points()
    torch.manual_seed(1)
    m = _PoissonResidual(16).double()
    n = f_int.numel()

    # one forward feeds the balancer (its backward frees this graph)
    r_a = m(x_int, x_bc, f_int)
    weights = bal_weights(m, r_a, n)

    # an independent forward feeds the manual reference norms
    r_b = m(x_int, x_bc, f_int)
    loss_pde = (r_b[:n] ** 2).mean()
    loss_bc = 1000.0 * (r_b[n:] ** 2).mean()
    params = list(m.parameters())
    g_pde = torch.autograd.grad(loss_pde, params, retain_graph=True)
    g_bc = torch.autograd.grad(loss_bc, params)
    norm_pde = torch.sqrt(sum((g**2).sum() for g in g_pde))
    norm_bc = torch.sqrt(sum((g**2).sum() for g in g_bc))
    norms = torch.stack([norm_pde, norm_bc])

    weighted = weights * norms
    assert torch.allclose(weighted, weighted[0] * torch.ones_like(weighted), rtol=1e-6)
    assert weights[0].item() == pytest.approx(1.0, abs=1e-9)  # reference term keeps weight 1


def bal_weights(m: nn.Module, r: torch.Tensor, n: int) -> torch.Tensor:
    loss_pde = (r[:n] ** 2).mean()
    loss_bc = 1000.0 * (r[n:] ** 2).mean()
    return GradNormBalancer(2, alpha=0.0, ref_index=0).update([loss_pde, loss_bc], m.parameters())


def test_grad_norm_balancer_ema_and_validation() -> None:
    bal = GradNormBalancer(3, alpha=0.5, ref_index=0)
    assert torch.allclose(bal.weights, torch.ones(3))
    with pytest.raises(ValueError):
        GradNormBalancer(0)
    with pytest.raises(ValueError):
        GradNormBalancer(2, alpha=1.5)
    with pytest.raises(ValueError):
        GradNormBalancer(2, ref_index=5)
    with pytest.raises(ValueError):
        bal.update([torch.zeros(())], [torch.zeros(1, requires_grad=True)])


# --- Cubic-regularised Newton (ARC) + exact Taylor line search --------------


def _spd_quadratic(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randn(n, n, dtype=torch.float64)
    mat = a @ a.T + n * torch.eye(n, dtype=torch.float64)
    vec = torch.randn(n, dtype=torch.float64)
    return mat, vec


def test_hvp_matches_dense_hessian_quadratic() -> None:
    mat, vec = _spd_quadratic(6)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ (mat @ x) + vec @ x

    x0 = torch.randn(6, dtype=torch.float64)
    v = torch.randn(6, dtype=torch.float64)
    assert torch.allclose(hvp(loss, x0, v), mat @ v, atol=1e-9)


def test_hvp_matches_autograd_nonquadratic() -> None:
    torch.manual_seed(0)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.sin(x) * torch.exp(0.1 * x))

    x0 = torch.randn(5, dtype=torch.float64) * 0.5
    v = torch.randn(5, dtype=torch.float64)
    dense = torch.func.hessian(loss)(x0)
    assert torch.allclose(hvp(loss, x0, v), dense @ v, atol=1e-9)


def test_lanczos_reconstructs_operator() -> None:
    torch.manual_seed(0)
    n = 8
    a = torch.randn(n, n, dtype=torch.float64)
    sym = a + a.T  # symmetric, indefinite
    b = torch.randn(n, dtype=torch.float64)
    q_basis, tri = lanczos_tridiag(lambda v: sym @ v, b, n)
    m = q_basis.shape[1]
    assert torch.allclose(q_basis.T @ q_basis, torch.eye(m, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(q_basis[:, 0], b / torch.linalg.vector_norm(b), atol=1e-10)
    assert torch.allclose(q_basis.T @ (sym @ q_basis), tri, atol=1e-7)


def test_cubic_regularized_newton_convex_quadratic() -> None:
    mat, vec = _spd_quadratic(10)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ (mat @ x) + vec @ x

    x_star = torch.linalg.solve(mat, -vec)
    opt = CubicRegularizedNewton(sigma=1.0, krylov_dim=10)
    x, hist = opt.minimize(loss, torch.zeros(10, dtype=torch.float64), steps=30)
    assert torch.allclose(x, x_star, atol=1e-6)
    assert all(hist[i + 1] <= hist[i] + 1e-12 for i in range(len(hist) - 1))


def test_cubic_regularized_newton_rosenbrock() -> None:
    def loss(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    opt = CubicRegularizedNewton(sigma=1.0, krylov_dim=2)
    x, hist = opt.minimize(loss, torch.tensor([-1.2, 1.0], dtype=torch.float64), steps=150)
    assert torch.allclose(x, torch.ones(2, dtype=torch.float64), atol=1e-4)
    assert hist[-1] < 1e-10


def test_cubic_regularized_newton_escapes_saddle() -> None:
    # f = (x^2 - 1)^2 + y^2: minima at (+-1, 0); the origin is an indefinite saddle
    # (Hessian diag(-4, 2)), where gradient methods stall. CRN uses the negative-curvature
    # direction from the cubic model to escape.
    def loss(x: torch.Tensor) -> torch.Tensor:
        return (x[0] ** 2 - 1.0) ** 2 + x[1] ** 2

    opt = CubicRegularizedNewton(sigma=1.0, krylov_dim=2)
    x, hist = opt.minimize(loss, torch.tensor([1e-3, 1e-3], dtype=torch.float64), steps=60)
    assert hist[-1] < 1e-10
    assert abs(abs(float(x[0])) - 1.0) < 1e-4
    assert abs(float(x[1])) < 1e-4


def test_cubic_regularized_newton_step_is_descent() -> None:
    mat, vec = _spd_quadratic(6)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ (mat @ x) + vec @ x

    x0 = torch.randn(6, dtype=torch.float64)
    s = cubic_regularized_newton_step(loss, x0, sigma=1.0, krylov_dim=6)
    assert float(loss(x0 + s)) < float(loss(x0))


def test_cubic_newton_validation() -> None:
    with pytest.raises(ValueError):
        CubicRegularizedNewton(sigma=0.0)
    with pytest.raises(ValueError):
        CubicRegularizedNewton(eta_accept=0.9, eta_success=0.1)
    with pytest.raises(ValueError):
        CubicRegularizedNewton(sigma_increase=1.0)
    with pytest.raises(ValueError):
        CubicRegularizedNewton(krylov_dim=0)


def test_taylor_line_min_quadratic_is_exact_newton_step() -> None:
    mat, vec = _spd_quadratic(5)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ (mat @ x) + vec @ x

    torch.manual_seed(1)
    p = torch.randn(5, dtype=torch.float64)
    d = torch.randn(5, dtype=torch.float64)
    g = mat @ p + vec
    a_exact = -float(g @ d) / float(d @ (mat @ d))
    a_star, _ = taylor_line_min(loss, p, d, order=2, bracket=(-10.0, 10.0))
    assert a_star == pytest.approx(a_exact, abs=1e-8)


def test_taylor_line_min_order3_recovers_cubic_minimizer() -> None:
    def loss(x: torch.Tensor) -> torch.Tensor:
        return x[0] ** 3 - x[0]  # along e0: phi(a) = a^3 - a, minimum at 1/sqrt(3)

    p = torch.zeros(1, dtype=torch.float64)
    d = torch.ones(1, dtype=torch.float64)
    a_star, _ = taylor_line_min(loss, p, d, order=3, bracket=(0.0, 1.0))
    assert a_star == pytest.approx(1.0 / math.sqrt(3.0), abs=1e-6)


def test_taylor_line_min_validation() -> None:
    def loss(x: torch.Tensor) -> torch.Tensor:
        return x @ x

    p = torch.zeros(2, dtype=torch.float64)
    d = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        taylor_line_min(loss, p, d, order=5)
    with pytest.raises(ValueError):
        taylor_line_min(loss, p, d, bracket=(1.0, 0.0))


def test_cubic_gauss_newton_linear_least_squares_minimizer() -> None:
    # inconsistent overdetermined system: CRGN must find the exact LS minimiser
    torch.manual_seed(0)
    n, p_dim = 20, 5
    mat = torch.randn(n, p_dim, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return mat @ p - b

    p_star = torch.linalg.lstsq(mat, b).solution
    opt = CubicRegularizedGaussNewton(sigma=1.0, krylov_dim=p_dim)
    p, _hist = opt.minimize(residual_fn, torch.zeros(p_dim, dtype=torch.float64), steps=40)
    assert torch.allclose(p, p_star, atol=1e-6)


def test_cubic_gauss_newton_recovers_nonlinear_least_squares() -> None:
    t = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    true = torch.tensor([2.0, -0.7], dtype=torch.float64)
    y = true[0] * torch.exp(true[1] * t)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.exp(p[1] * t) - y

    opt = CubicRegularizedGaussNewton(sigma=1.0, krylov_dim=2)
    p, history = opt.minimize(residual_fn, torch.tensor([1.0, 0.0], dtype=torch.float64), steps=60)
    assert history[-1] < 1e-12
    assert torch.allclose(p, true, atol=1e-5)
    assert all(history[i + 1] <= history[i] + 1e-14 for i in range(len(history) - 1))


def test_cubic_gauss_newton_rejects_at_minimum() -> None:
    t = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)
    y = 1.5 * torch.exp(-0.3 * t)

    def residual_fn(p: torch.Tensor) -> torch.Tensor:
        return p[0] * torch.exp(p[1] * t) - y

    p_star = torch.tensor([1.5, -0.3], dtype=torch.float64)
    new_p, info = CubicRegularizedGaussNewton(sigma=1e-4, krylov_dim=2).step(residual_fn, p_star)
    assert not info.accepted
    assert torch.allclose(new_p, p_star, atol=0.0)


def test_cubic_gauss_newton_inherits_validation() -> None:
    with pytest.raises(ValueError):
        CubicRegularizedGaussNewton(sigma=0.0)
    with pytest.raises(ValueError):
        CubicRegularizedGaussNewton(krylov_dim=0)


# --- JetLBFGS (L-BFGS lifted by exact-jet line search + curvature scale) ----


def test_jet_lbfgs_convex_quadratic() -> None:
    mat, vec = _spd_quadratic(10)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ (mat @ x) + vec @ x

    x_star = torch.linalg.solve(mat, -vec)
    opt = JetLBFGS(history_size=10)
    x, hist = opt.minimize(loss, torch.zeros(10, dtype=torch.float64), steps=40)
    assert torch.allclose(x, x_star, atol=1e-6)
    assert all(hist[i + 1] <= hist[i] + 1e-12 for i in range(len(hist) - 1))


def test_jet_lbfgs_exact_h0_makes_first_step_unit() -> None:
    # with gamma = <g,g>/<g,Hg>, the exact order-2 line search lands at a = 1 by construction
    mat, vec = _spd_quadratic(6, seed=3)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ (mat @ x) + vec @ x

    x0 = torch.randn(6, dtype=torch.float64)
    _new, info = JetLBFGS(exact_h0=True, line_search_order=2, max_step=10.0).step(loss, x0)
    assert info.step_size == pytest.approx(1.0, abs=1e-9)


def test_jet_lbfgs_rosenbrock() -> None:
    def loss(x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    opt = JetLBFGS(history_size=10, line_search_order=3, max_step=5.0)
    x, hist = opt.minimize(loss, torch.tensor([-1.2, 1.0], dtype=torch.float64), steps=200)
    assert torch.allclose(x, torch.ones(2, dtype=torch.float64), atol=1e-3)
    assert hist[-1] < 1e-8


def test_jet_lbfgs_scalar_h0_also_converges() -> None:
    mat, vec = _spd_quadratic(8, seed=2)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x @ (mat @ x) + vec @ x

    x_star = torch.linalg.solve(mat, -vec)
    opt = JetLBFGS(history_size=8, exact_h0=False)
    x, _hist = opt.minimize(loss, torch.zeros(8, dtype=torch.float64), steps=60)
    assert torch.allclose(x, x_star, atol=1e-5)


def test_jet_lbfgs_validation() -> None:
    with pytest.raises(ValueError):
        JetLBFGS(history_size=0)
    with pytest.raises(ValueError):
        JetLBFGS(line_search_order=1)
    with pytest.raises(ValueError):
        JetLBFGS(max_step=0.0)


# --- Certified-conditioning (eps -> 0 collapse) damping floor -------------


def test_gauss_newton_target_condition_certifies_and_floors_damping() -> None:
    """target_condition floors the LM damping so kappa(J^T J + eps I) is provably <= T."""
    from omnibias.core.proof.certificate import verify_certificate_digest

    torch.manual_seed(0)
    n, p = 12, 5
    u, _ = torch.linalg.qr(torch.randn(n, p, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(p, p, dtype=torch.float64))
    svals = torch.tensor([1e4, 1e3, 1e2, 1e1, 1.0], dtype=torch.float64)  # kappa(J^T J) ~ 1e8
    a = (u * svals) @ v.T
    b = torch.randn(n, dtype=torch.float64)

    def residual_fn(pp: torch.Tensor) -> torch.Tensor:
        return a @ pp - b

    target = 100.0
    opt = GaussNewton(damping=1e-8, solver="dense", target_condition=target)
    _, info = opt.step(residual_fn, torch.zeros(p, dtype=torch.float64))
    assert opt.last_certificate is not None
    assert verify_certificate_digest(opt.last_certificate)
    eps = opt.last_certificate["payload"]["certified_damping"]
    normal = (a.T @ a).numpy()
    assert np.linalg.cond(normal + eps * np.eye(p)) <= target + 1e-6
    assert eps > 1e-8  # the certified floor genuinely raised the tiny LM damping
    assert info.accepted


def test_gauss_newton_no_target_condition_has_no_certificate() -> None:
    t = torch.linspace(0.0, 1.0, 16, dtype=torch.float64)
    y = 1.5 * torch.exp(-0.3 * t)

    def residual_fn(pp: torch.Tensor) -> torch.Tensor:
        return pp[0] * torch.exp(pp[1] * t) - y

    opt = GaussNewton(damping=1e-3)
    opt.step(residual_fn, torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert opt.last_certificate is None


def test_gauss_newton_target_condition_validation() -> None:
    with pytest.raises(ValueError, match="target_condition"):
        GaussNewton(target_condition=1.0)
