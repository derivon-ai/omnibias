# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Drop-in ``torch.optim.Optimizer`` curvature optimisers (torch).

These pin down the *bridge*: the drop-in front ends must reproduce the functional
exact-curvature cores step for step (same ARC math, same exact Hessian), the diagonal
preconditioner must read the *exact* curvature (Gauss-Newton diagonal exact, Hutchinson
diagonal unbiased), and the whole family must be a genuine one-line replacement for Adam
that beats it on a stiff PINN residual. Adam itself is never restricted -- it is the
baseline every test compares against.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn as nn
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.architectures import JetMLP
from omnibias.torch.optim import (
    KFAC,
    ConformalSymplectic,
    CubicGaussNewton,
    CubicNewton,
    CubicRegularizedNewton,
    DiagonalCurvature,
    FrugalCurvature,
    JetLBFGSOptimizer,
    JetSubspaceTensor,
    NaturalGradient,
    StochasticNewtonCG,
    TrustRegionNewtonCG,
    functional_residual_fn,
    gauss_newton_fisher,
    gauss_newton_fisher_matvec,
    natural_gradient_direction,
    solve_subspace_trust_region,
    steihaug_cg,
    taylor_subspace_model,
)
from torch.nn.utils import parameters_to_vector


@pytest.fixture(autouse=True)
def _use_float64() -> object:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# --- test problems --------------------------------------------------------


class _Quadratic(nn.Module):
    """Strictly convex quadratic ``0.5 theta^T A theta - b^T theta`` (min at ``A^{-1} b``)."""

    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(a.shape[0], dtype=torch.float64))
        self.register_buffer("a", a)
        self.register_buffer("b", b)

    def loss(self) -> torch.Tensor:
        return 0.5 * self.theta @ (self.a @ self.theta) - self.b @ self.theta


def _spd(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, n, generator=g, dtype=torch.float64)
    return m @ m.T + n * torch.eye(n, dtype=torch.float64)


class _PoissonResidual(nn.Module):
    """Residual of ``u'' = f``, ``u(0)=u(1)=0`` (manufactured ``u*=sin(pi x)``)."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = JetMLP(1, hidden, 1, depth=2, base="tanh")

    def forward(self, x_int: torch.Tensor, x_bc: torch.Tensor, f_int: torch.Tensor) -> torch.Tensor:
        _v, _g, h = self.net.value_grad_hessian(x_int)
        res_pde = h[:, 0, 0, 0] - f_int
        u_bc = self.net.value(x_bc).reshape(-1)
        return torch.cat([res_pde, u_bc])


def _poisson_points(n: int = 40) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_int = torch.linspace(0.0, 1.0, n, dtype=torch.float64)[1:-1].reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    f_int = (-math.pi**2) * torch.sin(math.pi * x_int).reshape(-1)
    return x_int, x_bc, f_int


def _poisson_loss(model: _PoissonResidual, pts: tuple[torch.Tensor, ...]) -> float:
    with torch.no_grad():
        r = model(*pts)
    return 0.5 * float((r**2).mean())


# --- bridge fidelity: drop-in == functional core --------------------------


def test_cubic_newton_dropin_matches_functional_core() -> None:
    """The drop-in ``CubicNewton`` reproduces ``CubicRegularizedNewton`` step for step."""
    x_int, x_bc, f_int = _poisson_points(16)

    torch.manual_seed(0)
    m_func = _PoissonResidual(8).double()
    flat0, residual_fn = functional_residual_fn(m_func, x_int, x_bc, f_int)

    def loss_fn(vec: torch.Tensor) -> torch.Tensor:
        r = residual_fn(vec)
        return 0.5 * (r**2).mean()

    final_func, _hist = CubicRegularizedNewton(krylov_dim=10).minimize(loss_fn, flat0, steps=6)

    torch.manual_seed(0)
    m_drop = _PoissonResidual(8).double()

    def closure() -> torch.Tensor:
        r = m_drop(x_int, x_bc, f_int)
        return 0.5 * (r**2).mean()

    opt = CubicNewton(m_drop.parameters(), krylov_dim=10)
    for _ in range(6):
        opt.step(closure)
    final_drop = parameters_to_vector(m_drop.parameters())

    assert torch.allclose(final_func, final_drop, atol=1e-7, rtol=0.0)


# --- Newton exactness on a quadratic --------------------------------------


def test_cubic_newton_solves_quadratic() -> None:
    """Cubic-regularised Newton drives a strictly convex quadratic to its exact minimum."""
    a = _spd(6, 1)
    b = torch.randn(6, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    model = _Quadratic(a, b)
    theta_star = torch.linalg.solve(a, b)

    opt = CubicNewton(model.parameters())
    for _ in range(25):
        opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_star, atol=1e-6)


def test_jet_lbfgs_dropin_solves_quadratic() -> None:
    """Exact-curvature L-BFGS reaches the quadratic minimum (exact H0 + line search)."""
    a = _spd(6, 3)
    b = torch.randn(6, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    model = _Quadratic(a, b)
    theta_star = torch.linalg.solve(a, b)

    opt = JetLBFGSOptimizer(model.parameters())
    for _ in range(40):
        opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_star, atol=1e-6)


# --- Steihaug truncated-CG (trust-region subproblem solver) ---------------


def test_steihaug_cg_interior_solves_spd() -> None:
    """With a large radius, Steihaug-CG returns the exact Newton step ``-H^{-1} g`` (interior)."""
    a = _spd(6, 21)
    g = torch.randn(6, generator=torch.Generator().manual_seed(22), dtype=torch.float64)
    p, hit = steihaug_cg(lambda v: a @ v, g, radius=1e6, max_iter=100, tol=1e-12)
    assert not hit
    assert torch.allclose(a @ p, -g, atol=1e-6)


def test_steihaug_cg_hits_boundary_small_radius() -> None:
    """A tight radius truncates CG on the trust-region boundary (``||p|| == radius``)."""
    a = _spd(6, 21)
    g = torch.randn(6, generator=torch.Generator().manual_seed(22), dtype=torch.float64)
    radius = 0.01
    p, hit = steihaug_cg(lambda v: a @ v, g, radius=radius, max_iter=100, tol=1e-12)
    assert hit
    assert math.isclose(float(torch.linalg.vector_norm(p)), radius, rel_tol=1e-9)
    model = float(g @ p) + 0.5 * float(p @ (a @ p))
    assert model < 0.0  # the boundary step still reduces the quadratic model


def test_steihaug_cg_negative_curvature_to_boundary() -> None:
    """On an indefinite Hessian, CG rides a negative-curvature direction to the boundary."""
    h = torch.diag(torch.tensor([3.0, -1.0], dtype=torch.float64))
    g = torch.tensor([1.0, 1.0], dtype=torch.float64)
    radius = 2.0
    p, hit = steihaug_cg(lambda v: h @ v, g, radius=radius, max_iter=50, tol=1e-12)
    assert hit
    assert math.isclose(float(torch.linalg.vector_norm(p)), radius, rel_tol=1e-9)
    model = float(g @ p) + 0.5 * float(p @ (h @ p))
    assert model < 0.0


def test_trust_region_newton_cg_solves_quadratic() -> None:
    """Trust-region Newton-CG drives a strictly convex quadratic to its exact minimum."""
    a = _spd(6, 23)
    b = torch.randn(6, generator=torch.Generator().manual_seed(24), dtype=torch.float64)
    model = _Quadratic(a, b)
    theta_star = torch.linalg.solve(a, b)

    opt = TrustRegionNewtonCG(model.parameters())
    for _ in range(30):
        opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_star, atol=1e-6)


def test_newton_cg_reports_last_cg_iters() -> None:
    """Both Newton-CG drop-ins record a sane ``last_cg_iters`` (1..cg_max_iter) after a step."""
    a = _spd(6, 41)
    b = torch.randn(6, generator=torch.Generator().manual_seed(42), dtype=torch.float64)
    for make in (
        lambda m: TrustRegionNewtonCG(m.parameters(), cg_max_iter=10),
        lambda m: StochasticNewtonCG(m.parameters(), damping=1e-3, cg_max_iter=10),
    ):
        model = _Quadratic(a, b)
        opt = make(model)
        assert opt.last_cg_iters == 0  # nothing solved yet
        opt.step(model.loss)
        assert 1 <= opt.last_cg_iters <= 10


@pytest.mark.slow
def test_trust_region_newton_cg_beats_adam_on_poisson() -> None:
    """The matrix-free trust-region Newton drop-in beats Adam's plateau on the Poisson PINN."""
    x_int, x_bc, f_int = _poisson_points(16)

    torch.manual_seed(0)
    m_tr = _PoissonResidual(8).double()

    def closure() -> torch.Tensor:
        r = m_tr(x_int, x_bc, f_int)
        return 0.5 * (r**2).mean()

    opt = TrustRegionNewtonCG(m_tr.parameters(), cg_max_iter=25)
    for _ in range(20):
        opt.step(closure)
    tr_loss = _poisson_loss(m_tr, (x_int, x_bc, f_int))

    torch.manual_seed(0)
    m_adam = _PoissonResidual(8).double()
    adam = torch.optim.Adam(m_adam.parameters(), lr=1e-3)
    for _ in range(500):
        adam.zero_grad()
        r = m_adam(x_int, x_bc, f_int)
        (0.5 * (r**2).mean()).backward()
        adam.step()
    adam_loss = _poisson_loss(m_adam, (x_int, x_bc, f_int))

    assert tr_loss < adam_loss


# --- stochastic / subsampled Newton-CG ------------------------------------


class _SubsampledLeastSquares(nn.Module):
    """Consistent LS ``y = M theta*``; every row-subset shares the same root ``theta*``."""

    def __init__(self, m: torch.Tensor, theta_true: torch.Tensor, subset: int, seed: int) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(m.shape[1], dtype=torch.float64))
        self.register_buffer("m", m)
        self.register_buffer("y", m @ theta_true)
        self.subset = subset
        self._gen = torch.Generator().manual_seed(seed)
        self._idx = torch.arange(subset)

    def resample(self) -> None:
        self._idx = torch.randperm(self.m.shape[0], generator=self._gen)[: self.subset]

    def loss(self) -> torch.Tensor:
        r = self.m[self._idx] @ self.theta - self.y[self._idx]
        return 0.5 * (r**2).mean()


def test_stochastic_newton_cg_solves_quadratic() -> None:
    """With no subsampling, Levenberg-damped Newton-CG relaxes damping and solves the quadratic."""
    a = _spd(6, 33)
    b = torch.randn(6, generator=torch.Generator().manual_seed(34), dtype=torch.float64)
    model = _Quadratic(a, b)
    theta_star = torch.linalg.solve(a, b)

    opt = StochasticNewtonCG(model.parameters(), damping=1e-3)
    for _ in range(40):
        opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_star, atol=1e-6)


def test_stochastic_newton_cg_converges_under_subsampling() -> None:
    """Subsampled Newton-CG with a per-step resample hook still finds the shared consistent root."""
    g = torch.Generator().manual_seed(31)
    m = torch.randn(20, 4, generator=g, dtype=torch.float64)
    theta_true = torch.randn(4, generator=g, dtype=torch.float64)
    model = _SubsampledLeastSquares(m, theta_true, subset=10, seed=99)

    opt = StochasticNewtonCG(model.parameters(), damping=1e-2, resample=model.resample)
    for _ in range(60):
        opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_true, atol=1e-4)


def test_stochastic_newton_cg_curvature_closure_path() -> None:
    """The ``|S_H| < |S_g|`` split (separate curvature closure) runs and reduces the loss."""
    g = torch.Generator().manual_seed(41)
    m = torch.randn(24, 5, generator=g, dtype=torch.float64)
    theta_true = torch.randn(5, generator=g, dtype=torch.float64)
    model = _SubsampledLeastSquares(m, theta_true, subset=16, seed=7)

    def curvature_closure() -> torch.Tensor:  # curvature on a smaller subset of the same batch
        r = model.m[model._idx[:8]] @ model.theta - model.y[model._idx[:8]]
        return 0.5 * (r**2).mean()

    opt = StochasticNewtonCG(
        model.parameters(), damping=1e-2, resample=model.resample, curvature_closure=curvature_closure
    )
    f_start = float(model.loss().detach())
    for _ in range(40):
        opt.step(model.loss)
    assert float(model.loss().detach()) < 1e-3 * (1.0 + f_start)


# --- least-squares (PINN) drop-in vs Adam ---------------------------------


@pytest.mark.slow
def test_cubic_gauss_newton_dropin_beats_adam_on_poisson() -> None:
    """The GN drop-in drives the Poisson residual far below Adam in far fewer steps."""
    pts = _poisson_points(24)

    torch.manual_seed(0)
    m_gn = _PoissonResidual(12).double()
    opt = CubicGaussNewton(m_gn.parameters(), krylov_dim=15)

    def residual_closure() -> torch.Tensor:
        return m_gn(*pts)

    for _ in range(30):
        opt.step(residual_closure)
    gn_loss = _poisson_loss(m_gn, pts)

    torch.manual_seed(0)
    m_adam = _PoissonResidual(12).double()
    adam = torch.optim.Adam(m_adam.parameters(), lr=1e-3)
    for _ in range(500):
        adam.zero_grad()
        r = m_adam(*pts)
        (0.5 * (r**2).mean()).backward()
        adam.step()
    adam_loss = _poisson_loss(m_adam, pts)

    assert gn_loss < 5e-4
    assert gn_loss < adam_loss / 5.0


# --- diagonal preconditioner: exact curvature -----------------------------


class _LinearResidual(nn.Module):
    """Affine residual ``r = M theta - y`` (constant Jacobian ``J = M``)."""

    def __init__(self, m: torch.Tensor, y: torch.Tensor) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(m.shape[1], dtype=torch.float64))
        self.register_buffer("m", m)
        self.register_buffer("y", y)

    def forward(self) -> torch.Tensor:
        return self.m @ self.theta - self.y


def test_diagonal_gauss_newton_is_exact() -> None:
    """The Gauss-Newton diagonal equals the true ``diag(J^T J)/N`` (constant Jacobian)."""
    g = torch.Generator().manual_seed(5)
    m = torch.randn(9, 4, generator=g, dtype=torch.float64)
    y = torch.randn(9, generator=g, dtype=torch.float64)
    model = _LinearResidual(m, y)
    n_res = m.shape[0]

    opt = DiagonalCurvature(model.parameters(), curvature="gauss_newton", curvature_every=1)
    opt.step(model.forward)
    exact = (m**2).sum(0) / n_res
    assert opt._d is not None
    # the bias-corrected running diagonal recovers the exact diag(J^T J)/N after one step
    d_hat = opt._d / (1.0 - opt.beta2**opt._d_updates)
    assert torch.allclose(d_hat, exact, atol=1e-10)


def test_diagonal_hutchinson_unbiased() -> None:
    """The Hutchinson Hessian diagonal averages to the true ``diag(H)`` (SPD quadratic)."""
    a = _spd(5, 7)
    b = torch.zeros(5, dtype=torch.float64)
    model = _Quadratic(a, b)

    opt = DiagonalCurvature(
        model.parameters(), curvature="hutchinson", curvature_every=1, hutchinson_samples=4000
    )
    opt.step(model.loss)
    assert opt._d is not None
    exact = torch.diagonal(a).abs()
    d_hat = opt._d / (1.0 - opt.beta2**opt._d_updates)
    rel = (d_hat - exact).abs() / (exact.abs() + 1e-12)
    assert float(rel.max()) < 0.1


@pytest.mark.slow
def test_diagonal_curvature_beats_adam_on_poisson() -> None:
    """The exact-diagonal Adam substitute beats Adam on the stiff Poisson residual."""
    pts = _poisson_points(16)

    torch.manual_seed(0)
    m_diag = _PoissonResidual(12).double()
    opt = DiagonalCurvature(
        m_diag.parameters(), lr=5e-2, curvature="gauss_newton", curvature_every=10, clip=1.0
    )

    def residual_closure() -> torch.Tensor:
        return m_diag(*pts)

    for _ in range(400):
        opt.step(residual_closure)
    diag_loss = _poisson_loss(m_diag, pts)

    torch.manual_seed(0)
    m_adam = _PoissonResidual(12).double()
    adam = torch.optim.Adam(m_adam.parameters(), lr=1e-3)
    for _ in range(400):
        adam.zero_grad()
        r = m_adam(*pts)
        (0.5 * (r**2).mean()).backward()
        adam.step()
    adam_loss = _poisson_loss(m_adam, pts)

    assert diag_loss < adam_loss


@pytest.mark.slow
def test_diagonal_curvature_hardened_is_stable_on_poisson() -> None:
    """The hardened diagonal (rel-floor + safeguard) descends monotonically -- no wandering.

    Regression for the benchmark failure where the raw GN diagonal stalled/diverged (best
    rel-L2 ~ 0.5). With the condition-number floor and the monotone safeguard the training
    loss is finite and non-increasing at every step.
    """
    x_int, x_bc, f_int = _poisson_points(12)

    torch.manual_seed(0)
    m = _PoissonResidual(8).double()

    def residual_closure() -> torch.Tensor:
        return m(x_int, x_bc, f_int)

    opt = DiagonalCurvature(
        m.parameters(),
        lr=5e-2,
        curvature="gauss_newton",
        curvature_every=10,
        clip=1.0,
        rel_floor=1e-2,
        safeguard=True,
    )
    f0 = _poisson_loss(m, (x_int, x_bc, f_int))
    losses = []
    for _ in range(80):
        opt.step(residual_closure)
        losses.append(_poisson_loss(m, (x_int, x_bc, f_int)))

    assert all(math.isfinite(val) for val in losses)
    assert losses[-1] < f0  # genuine progress
    # the safeguard forbids any loss-increasing step (monotone up to fp noise)
    assert all(b <= a + 1e-10 for a, b in zip(losses[:-1], losses[1:], strict=True))


# --- K-FAC (hook-based Kronecker curvature) -------------------------------


class _OneLayer(nn.Module):
    """Scalar one-layer net ``f = c . tanh(W x + beta) + b`` (matches kfac_kron_factors)."""

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(d, h)
        self.out = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.hidden(x)))


class _MLP(nn.Module):
    """Plain 2-hidden-layer tanh MLP with a standard (non-jet) forward, for K-FAC."""

    def __init__(self, d: int, h: int, out: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(d, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)


def test_kfac_factors_match_closed_form() -> None:
    """K-FAC's hook-captured ``A``/``G`` equal the closed-form ``kfac_kron_factors`` (exact)."""
    d, h, batch = 3, 5, 32
    torch.manual_seed(0)
    net = _OneLayer(d, h).double()
    x = torch.randn(batch, d, dtype=torch.float64)

    opt = KFAC(net)
    opt.zero_grad(set_to_none=True)
    (net(x).sum()).backward()  # L = sum(f) => dL/ds = c (.) sigma'(z), the closed-form g
    opt._accumulate()

    w = net.hidden.weight.detach()
    beta = net.hidden.bias.detach()
    c = net.out.weight.detach()[0]
    z = x @ w.t() + beta
    sigma_p = get_activation("tanh").fastpath(z, 1)
    grad_pre = sigma_p * c
    a_cf = (x.t() @ x) / batch
    g_cf = (grad_pre.t() @ grad_pre) / batch

    a_kfac = opt._a_cov[net.hidden]  # bias-augmented (d+1, d+1)
    g_kfac = opt._g_cov[net.hidden]  # (h, h)
    assert torch.allclose(a_kfac[:d, :d], a_cf, atol=1e-10)
    assert torch.allclose(g_kfac, g_cf, atol=1e-10)
    opt.remove_hooks()


def test_kfac_reduces_loss_and_beats_sgd() -> None:
    """K-FAC's preconditioning drives a realizable regression below plain SGD in equal steps."""
    torch.manual_seed(0)
    d, h = 4, 16
    teacher = _MLP(d, h, 1).double()
    x = torch.randn(96, d, dtype=torch.float64)
    y = teacher(x).detach()

    student_kfac = _MLP(d, h, 1).double()
    student_sgd = _MLP(d, h, 1).double()
    student_sgd.load_state_dict(student_kfac.state_dict())  # identical init

    def mse(net: nn.Module) -> torch.Tensor:
        return ((net(x) - y) ** 2).mean()

    f_start = float(mse(student_kfac).detach())
    opt = KFAC(student_kfac, lr=5e-2, damping=1e-1, refresh_every=5)
    for _ in range(60):
        opt.step(lambda: mse(student_kfac))
    kfac_loss = float(mse(student_kfac).detach())
    opt.remove_hooks()

    sgd = torch.optim.SGD(student_sgd.parameters(), lr=1e-1)
    for _ in range(60):
        sgd.zero_grad()
        mse(student_sgd).backward()
        sgd.step()
    sgd_loss = float(mse(student_sgd).detach())

    assert kfac_loss < 0.1 * f_start  # genuine, large reduction
    assert kfac_loss < sgd_loss  # curvature preconditioning beats plain SGD at equal steps


def test_kfac_precondition_modules_restricts_hooked_layers() -> None:
    """``precondition_modules`` hooks exactly the named Linears; the rest land in ``_other``."""
    torch.manual_seed(0)
    net = _MLP(4, 8, 1).double()
    opt = KFAC(net, precondition_modules=[net.l2])
    assert opt._layers == [net.l2]
    assert len(opt._handles) == 1  # only l2 is hooked
    other_ids = {id(p) for p in opt._other}
    assert other_ids == {id(net.l1.weight), id(net.l1.bias), id(net.l3.weight), id(net.l3.bias)}

    x = torch.randn(32, 4, dtype=torch.float64)
    y = torch.randn(32, 1, dtype=torch.float64)

    def mse() -> torch.Tensor:
        return ((net(x) - y) ** 2).mean()

    f0 = float(mse().detach())
    for _ in range(40):
        opt.step(mse)
    assert float(mse().detach()) < f0  # KFAC(l2) + SGD(l1,l3) still trains
    opt.remove_hooks()


def test_kfac_default_preconditions_all_linears() -> None:
    """Backward-compat: no new args => every ``nn.Linear`` preconditioned, ``_other`` empty."""
    torch.manual_seed(0)
    net = _MLP(4, 8, 1).double()
    opt = KFAC(net)
    assert {id(m) for m in opt._layers} == {id(net.l1), id(net.l2), id(net.l3)}
    assert opt._other == []
    assert opt.other_optimizer is None
    opt.remove_hooks()


def test_kfac_hybrid_delegates_other_params_to_other_optimizer() -> None:
    """The first hybrid step moves the non-preconditioned params by exactly ``other_optimizer``'s rule.

    At the shared initial point the gradient of the non-preconditioned params is identical whether
    or not KFAC preconditions the other layers, so a one-step AdamW-on-``l3`` must match a
    standalone AdamW-on-``l3`` bit for bit -- proving the delegation feeds the right grads and does
    not double-apply.
    """
    torch.manual_seed(0)
    d, h = 4, 8
    x = torch.randn(48, d, dtype=torch.float64)
    y = torch.randn(48, 1, dtype=torch.float64)

    def mse(net: nn.Module) -> torch.Tensor:
        return ((net(x) - y) ** 2).mean()

    net = _MLP(d, h, 1).double()
    ref = _MLP(d, h, 1).double()
    ref.load_state_dict(net.state_dict())  # identical init

    ref_adam = torch.optim.AdamW([ref.l3.weight, ref.l3.bias], lr=1e-2)
    ref_adam.zero_grad(set_to_none=True)
    mse(ref).backward()
    ref_adam.step()

    hyb_adam = torch.optim.AdamW([net.l3.weight, net.l3.bias], lr=1e-2)
    opt = KFAC(net, precondition_modules=[net.l1, net.l2], other_optimizer=hyb_adam)
    opt.step(lambda: mse(net))

    assert torch.allclose(net.l3.weight, ref.l3.weight, atol=1e-12)
    assert torch.allclose(net.l3.bias, ref.l3.bias, atol=1e-12)
    assert not torch.allclose(net.l1.weight, ref.l1.weight)  # KFAC did move l1
    opt.remove_hooks()


def test_kfac_other_optimizer_replaces_sgd_fallback() -> None:
    """With ``other_optimizer`` set, the non-preconditioned params take that step, not KFAC's SGD."""
    torch.manual_seed(0)
    d, h = 4, 8
    x = torch.randn(48, d, dtype=torch.float64)
    y = torch.randn(48, 1, dtype=torch.float64)

    def mse(net: nn.Module) -> torch.Tensor:
        return ((net(x) - y) ** 2).mean()

    net_sgd = _MLP(d, h, 1).double()
    net_hyb = _MLP(d, h, 1).double()
    net_hyb.load_state_dict(net_sgd.state_dict())

    opt_sgd = KFAC(net_sgd, precondition_modules=[net_sgd.l1, net_sgd.l2])  # l3 -> SGD fallback
    opt_sgd.step(lambda: mse(net_sgd))
    opt_sgd.remove_hooks()

    hyb_adam = torch.optim.AdamW([net_hyb.l3.weight, net_hyb.l3.bias], lr=1e-2)
    opt_hyb = KFAC(net_hyb, precondition_modules=[net_hyb.l1, net_hyb.l2], other_optimizer=hyb_adam)
    opt_hyb.step(lambda: mse(net_hyb))
    opt_hyb.remove_hooks()

    assert not torch.allclose(net_sgd.l3.weight, net_hyb.l3.weight)  # different rules => different l3


def test_kfac_rejects_overlap_and_non_linear_modules() -> None:
    """Overlap between ``other_optimizer`` and preconditioned params (or non-Linear entries) is rejected."""
    torch.manual_seed(0)
    net = _MLP(4, 8, 1).double()
    with pytest.raises(ValueError, match="double-update"):
        KFAC(
            net,
            precondition_modules=[net.l1],
            other_optimizer=torch.optim.SGD(net.parameters(), lr=0.1),  # owns l1 too
        )
    with pytest.raises(ValueError, match="nn.Linear"):
        KFAC(net, precondition_modules=[net])  # a whole MLP is not an nn.Linear


def test_kfac_cholesky_solver_matches_eigh() -> None:
    """The ``cholesky`` solver takes the same damped natural-gradient step as the default ``eigh``.

    Both solve the identical damped system ``(G + damp_g I)^{-1} grad (A + damp_a I)^{-1}`` (the
    trace-normalised ``pi`` is the mean eigenvalue = mean diagonal, so it matches exactly); with
    ``refresh_every=1`` both refactor the same EMA factors each step, so the trajectories coincide.
    """
    torch.manual_seed(0)
    d, h = 4, 12
    teacher = _MLP(d, h, 1).double()
    x = torch.randn(64, d, dtype=torch.float64)
    y = teacher(x).detach()

    net_e = _MLP(d, h, 1).double()
    net_c = _MLP(d, h, 1).double()
    net_c.load_state_dict(net_e.state_dict())  # identical init

    def mse(net: nn.Module) -> torch.Tensor:
        return ((net(x) - y) ** 2).mean()

    opt_e = KFAC(net_e, lr=5e-2, damping=1e-2, refresh_every=1, solver="eigh")
    opt_c = KFAC(net_c, lr=5e-2, damping=1e-2, refresh_every=1, solver="cholesky")
    for _ in range(30):
        opt_e.step(lambda: mse(net_e))
        opt_c.step(lambda: mse(net_c))
    opt_e.remove_hooks()
    opt_c.remove_hooks()

    ve = parameters_to_vector(net_e.parameters())
    vc = parameters_to_vector(net_c.parameters())
    assert torch.allclose(ve, vc, atol=1e-7, rtol=0.0)


def test_kfac_adaptive_damping_rejects_bad_step_and_accepts_good_step() -> None:
    """Adaptive damping rolls back a loss-increasing step (grows damping) and keeps a good one."""
    torch.manual_seed(0)
    d, h = 4, 8
    x = torch.randn(48, d, dtype=torch.float64)
    y = torch.randn(48, 1, dtype=torch.float64)

    net = _MLP(d, h, 1).double()

    def mse() -> torch.Tensor:
        return ((net(x) - y) ** 2).mean()

    # An enormous LR makes the natural-gradient step massively overshoot -> loss increases -> reject.
    opt = KFAC(net, lr=1e6, damping=1e-3, refresh_every=1, adaptive_damping=True)
    before = parameters_to_vector(net.parameters()).clone()
    d0 = opt.current_damping
    opt.step(mse)
    assert torch.allclose(before, parameters_to_vector(net.parameters()))  # rolled back
    assert opt.current_damping > d0  # damping grew after the rejected step
    opt.remove_hooks()

    # A realizable teacher target + small LR -> the step reduces the loss -> accept (damping shrinks).
    teacher = _MLP(d, h, 1).double()
    yt = teacher(x).detach()
    net2 = _MLP(d, h, 1).double()

    def mse2() -> torch.Tensor:
        return ((net2(x) - yt) ** 2).mean()

    # Heavy damping + modest LR -> the natural step is close to a (small) scaled gradient step, a
    # guaranteed descent direction, so the first step is accepted.
    opt2 = KFAC(net2, lr=5e-2, damping=1.0, refresh_every=1, adaptive_damping=True)
    b2 = parameters_to_vector(net2.parameters()).detach().clone()
    d0_2 = opt2.current_damping
    f0 = float(mse2().detach())
    opt2.step(mse2)
    assert not torch.allclose(b2, parameters_to_vector(net2.parameters()).detach())  # accepted -> moved
    assert float(mse2().detach()) < f0  # loss decreased
    assert opt2.current_damping < d0_2  # damping relaxed after the accepted step
    opt2.remove_hooks()


def test_kfac_adaptive_damping_is_monotone_and_does_not_spike() -> None:
    """Under an aggressive (spiky) setting, adaptive damping stays monotone; fixed damping spikes."""
    torch.manual_seed(0)
    d, h = 4, 16
    teacher = _MLP(d, h, 1).double()
    x = torch.randn(96, d, dtype=torch.float64)
    y = teacher(x).detach()

    net_ad = _MLP(d, h, 1).double()
    net_fx = _MLP(d, h, 1).double()
    net_fx.load_state_dict(net_ad.state_dict())  # identical init + identical aggressive hypers

    def mse(net: nn.Module) -> torch.Tensor:
        return ((net(x) - y) ** 2).mean()

    kw = dict(lr=1.0, damping=1e-4, refresh_every=3)
    opt_ad = KFAC(net_ad, adaptive_damping=True, **kw)  # type: ignore[arg-type]
    opt_fx = KFAC(net_fx, adaptive_damping=False, **kw)  # type: ignore[arg-type]
    f_start = float(mse(net_ad).detach())
    ad_losses = [float(opt_ad.step(lambda: mse(net_ad))) for _ in range(50)]
    fx_losses = [float(opt_fx.step(lambda: mse(net_fx))) for _ in range(50)]
    opt_ad.remove_hooks()
    opt_fx.remove_hooks()

    # adaptive: the (start-of-step) loss sequence never rises, and it makes real progress.
    assert all(math.isfinite(v) for v in ad_losses)
    assert all(ad_losses[i + 1] <= ad_losses[i] + 1e-9 for i in range(len(ad_losses) - 1))
    assert float(mse(net_ad).detach()) < 0.5 * f_start
    # fixed damping at the same aggressive setting is not monotone (it spikes / diverges).
    fixed_spikes = any(not math.isfinite(v) for v in fx_losses) or any(
        fx_losses[i + 1] > fx_losses[i] + 1e-6 for i in range(len(fx_losses) - 1)
    )
    assert fixed_spikes


def test_kfac_max_step_norm_caps_the_update() -> None:
    """``max_step_norm`` scales the whole natural-gradient step down to a global norm cap."""
    torch.manual_seed(0)
    d, h = 4, 8
    x = torch.randn(48, d, dtype=torch.float64)
    y = torch.randn(48, 1, dtype=torch.float64)

    def mse(net: nn.Module) -> torch.Tensor:
        return ((net(x) - y) ** 2).mean()

    net_free = _MLP(d, h, 1).double()
    net_cap = _MLP(d, h, 1).double()
    net_cap.load_state_dict(net_free.state_dict())  # identical init -> identical step direction

    v0 = parameters_to_vector(net_free.parameters()).detach().clone()
    opt_free = KFAC(net_free, lr=1.0, damping=1e-2)
    opt_free.step(lambda: mse(net_free))
    free_norm = float(torch.linalg.vector_norm(parameters_to_vector(net_free.parameters()).detach() - v0))
    opt_free.remove_hooks()

    cap = free_norm / 5.0
    opt_cap = KFAC(net_cap, lr=1.0, damping=1e-2, max_step_norm=cap)
    opt_cap.step(lambda: mse(net_cap))
    cap_norm = float(torch.linalg.vector_norm(parameters_to_vector(net_cap.parameters()).detach() - v0))
    opt_cap.remove_hooks()

    assert free_norm > cap  # the uncapped step overshoots the cap
    assert cap_norm == pytest.approx(cap, rel=1e-4)  # the capped step sits exactly on the cap


# --- JetSubspaceTensor: exact third-order tensor method in a subspace -----


def _orthonormal_basis(p: int, k: int, seed: int) -> torch.Tensor:
    """A random ``(p, k)`` matrix with orthonormal columns (a subspace basis ``Q``)."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(p, k, generator=g, dtype=torch.float64))
    return q


def test_taylor_subspace_model_matches_dense_quadratic() -> None:
    """On a quadratic the reduced model is exactly ``(Q^T g, Q^T A Q, 0)`` -- no third derivative."""
    p, k = 6, 3
    a = _spd(p, 31)
    b = torch.randn(p, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
    theta0 = torch.randn(p, generator=torch.Generator().manual_seed(33), dtype=torch.float64)
    q = _orthonormal_basis(p, k, 34)

    def loss_fn(theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * theta @ (a @ theta) - b @ theta

    c, hess, tensor3 = taylor_subspace_model(loss_fn, theta0, q, order=3)
    g = a @ theta0 - b
    assert torch.allclose(c, q.T @ g, atol=1e-10)
    assert torch.allclose(hess, q.T @ a @ q, atol=1e-10)
    assert tensor3 is not None
    assert float(tensor3.abs().max()) < 1e-8  # a quadratic has no third derivative


def test_taylor_subspace_model_recovers_cubic_third_derivative() -> None:
    """For ``f = sum theta_i^3 / 6`` the reduced tensor is the exact projection of ``grad^3 f``."""
    p, k = 5, 2
    theta0 = torch.randn(p, generator=torch.Generator().manual_seed(41), dtype=torch.float64)
    q = _orthonormal_basis(p, k, 42)

    def loss_fn(theta: torch.Tensor) -> torch.Tensor:
        return (theta**3).sum() / 6.0

    c, hess, tensor3 = taylor_subspace_model(loss_fn, theta0, q, order=3)
    assert torch.allclose(c, q.T @ (theta0**2 / 2.0), atol=1e-9)  # grad = theta^2 / 2
    assert torch.allclose(hess, q.T @ torch.diag(theta0) @ q, atol=1e-9)  # hess = diag(theta)
    assert tensor3 is not None
    expected = torch.einsum("pi,pj,pl->ijl", q, q, q)  # grad^3 is the diagonal all-ones tensor
    assert torch.allclose(tensor3, expected, atol=1e-9)


def test_solve_subspace_trust_region_quadratic_interior_and_boundary() -> None:
    """``tensor3=None`` gives the exact TR step: interior Newton for a big radius, on the ball for a small one."""
    k = 4
    a = _spd(k, 51)
    c = torch.randn(k, generator=torch.Generator().manual_seed(52), dtype=torch.float64)
    s_interior = solve_subspace_trust_region(c, a, None, radius=1e6)
    assert torch.allclose(s_interior, torch.linalg.solve(a, -c), atol=1e-8)
    s_boundary = solve_subspace_trust_region(c, a, None, radius=0.1)
    assert float(torch.linalg.vector_norm(s_boundary)) == pytest.approx(0.1, rel=1e-4)


def test_solve_subspace_trust_region_cubic_matches_gridsearch() -> None:
    """On a 2-D cubic model the solver stays in the ball and beats a dense grid search of it."""
    k = 2
    gen = torch.Generator().manual_seed(0)
    c = torch.randn(k, generator=gen, dtype=torch.float64)
    hraw = torch.randn(k, k, generator=gen, dtype=torch.float64)
    hess = 0.5 * (hraw + hraw.T)  # (possibly indefinite)
    traw = torch.randn(k, k, k, generator=gen, dtype=torch.float64)
    tensor3 = sum(traw.permute(*perm) for perm in itertools.permutations(range(3))) / 6.0
    radius = 1.0

    def model(a: torch.Tensor) -> float:
        return (
            float(c @ a)
            + 0.5 * float(a @ (hess @ a))
            + (1.0 / 6.0) * float(torch.einsum("ijl,i,j,l->", tensor3, a, a, a))
        )

    a_star = solve_subspace_trust_region(c, hess, tensor3, radius=radius)
    assert float(torch.linalg.vector_norm(a_star)) <= radius + 1e-8
    best = 0.0  # model(0) = 0
    grid = torch.linspace(-radius, radius, 121, dtype=torch.float64)
    for xi in grid:
        for yi in grid:
            pt = torch.stack([xi, yi])
            if float(torch.linalg.vector_norm(pt)) <= radius:
                best = min(best, model(pt))
    assert model(a_star) <= best + 1e-3


def test_jet_subspace_tensor_solves_quadratic() -> None:
    """Both the order-2 (subspace Newton) and order-3 (tensor) variants reach the exact minimum."""
    a = _spd(6, 61)
    b = torch.randn(6, generator=torch.Generator().manual_seed(62), dtype=torch.float64)
    theta_star = torch.linalg.solve(a, b)
    for order in (2, 3):
        model = _Quadratic(a, b)
        opt = JetSubspaceTensor(model.parameters(), subspace_dim=6, order=order, radius=10.0)
        for _ in range(25):
            opt.step(model.loss)
        assert torch.allclose(model.theta.detach(), theta_star, atol=1e-6)


def test_jet_subspace_tensor_k1_reduces_quadratic() -> None:
    """A one-dimensional subspace (k=1) is a valid exact line step and still reduces the loss."""
    a = _spd(5, 63)
    b = torch.randn(5, generator=torch.Generator().manual_seed(64), dtype=torch.float64)
    model = _Quadratic(a, b)
    opt = JetSubspaceTensor(model.parameters(), subspace_dim=1, order=3, radius=10.0)
    f0 = float(model.loss().detach())
    for _ in range(30):
        opt.step(model.loss)
    assert float(model.loss().detach()) < f0


@pytest.mark.slow
def test_jet_subspace_tensor_dropin_beats_adam_on_regression() -> None:
    """Iso-step: at an equal step budget the tensor method is orders of magnitude below Adam.

    This is the regime the method targets (a smooth full-batch objective) and the documented claim
    (the *iso-step* curvature win): with the same number of steps the exact third-order subspace model
    reaches far below a first-order method. A plain MLP (no PINN spatial Hessian) keeps the third-order
    autodiff shallow and cheap.
    """
    d, h = 4, 8
    torch.manual_seed(0)
    teacher = _MLP(d, h, 1).double()
    xg = torch.Generator().manual_seed(7)
    x = torch.randn(64, d, generator=xg, dtype=torch.float64)
    y = teacher(x).detach()

    torch.manual_seed(1)
    m_jst = _MLP(d, h, 1).double()
    torch.manual_seed(1)
    m_adam = _MLP(d, h, 1).double()  # identical initialisation to m_jst

    def mse(net: nn.Module) -> torch.Tensor:
        return ((net(x) - y) ** 2).mean()

    steps = 60
    opt = JetSubspaceTensor(m_jst.parameters(), subspace_dim=6, order=3)
    for _ in range(steps):
        opt.step(lambda: mse(m_jst))
    jst_loss = float(mse(m_jst).detach())

    adam = torch.optim.Adam(m_adam.parameters(), lr=1e-2)  # a strong first-order baseline, same budget
    for _ in range(steps):
        adam.zero_grad()
        mse(m_adam).backward()
        adam.step()
    adam_loss = float(mse(m_adam).detach())

    assert jst_loss < adam_loss / 10.0  # iso-step: an order of magnitude below Adam at equal steps


def test_step_returns_loss_and_reduces_it() -> None:
    """Smoke: every drop-in returns a finite scalar loss and reduces it from the start."""
    a = _spd(5, 11)
    b = torch.randn(5, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    for make_opt in (
        lambda m: CubicNewton(m.parameters()),
        lambda m: JetLBFGSOptimizer(m.parameters()),
        lambda m: JetSubspaceTensor(m.parameters()),
        lambda m: JetSubspaceTensor(m.parameters(), order=2),
        lambda m: TrustRegionNewtonCG(m.parameters()),
        lambda m: StochasticNewtonCG(m.parameters(), damping=1e-2),
        lambda m: DiagonalCurvature(m.parameters(), lr=1e-1, curvature="hutchinson"),
        lambda m: ConformalSymplectic(m.parameters(), lr=1e-1, mass="identity"),
        lambda m: ConformalSymplectic(m.parameters(), lr=1e-1, mass="hutchinson"),
        lambda m: FrugalCurvature(m.parameters(), lr=1e-1, curvature="hutchinson", curvature_every=1),
    ):
        model = _Quadratic(a, b)
        opt = make_opt(model)
        f_start = float(model.loss().detach())
        last = None
        for _ in range(20):
            last = opt.step(model.loss)
        assert last is not None and math.isfinite(float(last))
        assert float(model.loss().detach()) < f_start


# --- Conformal-Symplectic Descent (CSD) -----------------------------------


def test_csd_solves_quadratic() -> None:
    """Conformal-symplectic heavy-ball (identity mass) converges to the quadratic minimum."""
    a = _spd(4, 31)
    b = torch.randn(4, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
    model = _Quadratic(a, b)
    theta_star = torch.linalg.solve(a, b)

    opt = ConformalSymplectic(model.parameters(), lr=5e-2, momentum=0.9, mass="identity")
    for _ in range(4000):
        opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_star, atol=1e-5)


def test_csd_hutchinson_mass_converges() -> None:
    """The exact-curvature (Hutchinson) mass preconditions CSD to the quadratic minimum."""
    a = _spd(4, 33)
    b = torch.randn(4, generator=torch.Generator().manual_seed(34), dtype=torch.float64)
    model = _Quadratic(a, b)
    theta_star = torch.linalg.solve(a, b)

    opt = ConformalSymplectic(
        model.parameters(), lr=5e-2, momentum=0.9, mass="hutchinson", hutchinson_samples=8
    )
    for _ in range(4000):
        opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_star, atol=1e-3)


def test_csd_dissipative_converges_and_frictionless_is_stable() -> None:
    """Physics of the integrator: with friction CSD dissipates to the minimum (the velocity dies
    out); frictionless (``gamma=0``) it is symplectic and the trajectory stays bounded -- never
    the explicit-Euler blow-up."""
    a = _spd(5, 41)
    b = torch.zeros(5, dtype=torch.float64)  # minimum at the origin, L(0) = 0
    theta0 = torch.ones(5, dtype=torch.float64)
    loss0 = 0.5 * float(theta0 @ (a @ theta0))

    def run(gamma: float, steps: int) -> tuple[ConformalSymplectic, list[float]]:
        model = _Quadratic(a, b)
        with torch.no_grad():
            model.theta.copy_(theta0)
        opt = ConformalSymplectic(model.parameters(), lr=1e-2, gamma=gamma, mass="identity")
        losses = [float(model.loss().detach())]
        for _ in range(steps):
            opt.step(model.loss)
            losses.append(float(model.loss().detach()))
        return opt, losses

    opt_d, losses_d = run(gamma=5.0, steps=800)
    assert opt_d._v is not None
    assert losses_d[-1] < 1e-4  # rolled to the bottom
    assert float(torch.linalg.vector_norm(opt_d._v)) < 1e-2  # velocity dissipated

    _opt_c, losses_c = run(gamma=0.0, steps=800)
    assert all(math.isfinite(v) for v in losses_c)
    assert max(losses_c) < 10.0 * loss0  # symplectic: bounded, no drift / blow-up
    assert losses_c[-1] < 10.0 * loss0


def test_csd_line_search_reduces_loss() -> None:
    """The exact directional-jet line search (order 2 and 3) drives the loss down."""
    for order in (2, 3):
        a = _spd(5, 51)
        b = torch.randn(5, generator=torch.Generator().manual_seed(52), dtype=torch.float64)
        model = _Quadratic(a, b)
        f0 = float(model.loss().detach())
        opt = ConformalSymplectic(model.parameters(), lr=1e-1, mass="identity", line_search=order)
        for _ in range(30):
            opt.step(model.loss)
        assert float(model.loss().detach()) < f0


def test_csd_thermostat_is_stochastic_and_reduces_loss() -> None:
    """The Langevin thermostat keeps the step finite, makes the trajectory seed-dependent, and
    still drives the loss below the start."""
    a = _spd(5, 61)
    b = torch.randn(5, generator=torch.Generator().manual_seed(62), dtype=torch.float64)

    def final_theta(seed: int) -> torch.Tensor:
        model = _Quadratic(a, b)
        opt = ConformalSymplectic(
            model.parameters(), lr=2e-2, momentum=0.9, mass="identity", temperature=1e-3, seed=seed
        )
        for _ in range(200):
            opt.step(model.loss)
        return model.theta.detach().clone()

    t1 = final_theta(0)
    t2 = final_theta(1)
    assert torch.all(torch.isfinite(t1))
    assert not torch.allclose(t1, t2)  # different noise seeds -> different trajectories

    model = _Quadratic(a, b)
    with torch.no_grad():
        model.theta.copy_(t1)
    assert float(model.loss().detach()) < 0.0  # still descended below L(0) = 0


def test_csd_validation() -> None:
    """The constructor rejects out-of-range hyper-parameters."""
    p = [torch.zeros(2, requires_grad=True)]
    with pytest.raises(ValueError):
        ConformalSymplectic(p, lr=0.0)
    with pytest.raises(ValueError):
        ConformalSymplectic(p, momentum=1.0)
    with pytest.raises(ValueError):
        ConformalSymplectic(p, gamma=-1.0)
    with pytest.raises(ValueError):
        ConformalSymplectic(p, mass="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ConformalSymplectic(p, line_search=1)
    with pytest.raises(ValueError):
        ConformalSymplectic(p, temperature=-1.0)


@pytest.mark.slow
def test_csd_beats_adam_on_poisson() -> None:
    """CSD (Gauss-Newton mass + conformal momentum) beats tuned Adam on the stiff Poisson residual."""
    pts = _poisson_points(16)

    torch.manual_seed(0)
    m_csd = _PoissonResidual(12).double()
    opt = ConformalSymplectic(
        m_csd.parameters(),
        lr=5e-2,
        momentum=0.9,
        mass="gauss_newton",
        curvature_every=10,
        clip=1.0,
        safeguard=True,
    )

    def csd_closure() -> torch.Tensor:
        return m_csd(*pts)

    for _ in range(400):
        opt.step(csd_closure)
    csd_loss = _poisson_loss(m_csd, pts)

    torch.manual_seed(0)
    m_adam = _PoissonResidual(12).double()
    adam = torch.optim.Adam(m_adam.parameters(), lr=1e-3)
    for _ in range(400):
        adam.zero_grad()
        r = m_adam(*pts)
        (0.5 * (r**2).mean()).backward()
        adam.step()
    adam_loss = _poisson_loss(m_adam, pts)

    assert csd_loss < adam_loss


@pytest.mark.slow
def test_csd_beats_diagonal_curvature_ablation_on_poisson() -> None:
    """Ablation isolating the integrator. Both optimisers use the *identical* exact Gauss-Newton
    curvature, the same ``clip`` and monotone ``safeguard``, and an equal step budget -- the only
    difference is the update: CSD's conformal-symplectic momentum vs :class:`DiagonalCurvature`'s
    Adam-style ``m``/``d`` EMA. The symplectic integrator reaches a markedly lower residual
    (~40x here), and -- because the safeguard makes both runs deterministic -- reproducibly so."""
    pts = _poisson_points(16)

    torch.manual_seed(0)
    m_csd = _PoissonResidual(12).double()
    csd = ConformalSymplectic(
        m_csd.parameters(),
        lr=5e-2,
        momentum=0.9,
        mass="gauss_newton",
        curvature_every=10,
        clip=1.0,
        safeguard=True,
    )

    def csd_closure() -> torch.Tensor:
        return m_csd(*pts)

    for _ in range(400):
        csd.step(csd_closure)
    csd_loss = _poisson_loss(m_csd, pts)

    torch.manual_seed(0)
    m_diag = _PoissonResidual(12).double()
    diag = DiagonalCurvature(
        m_diag.parameters(),
        lr=5e-2,
        curvature="gauss_newton",
        curvature_every=10,
        clip=1.0,
        safeguard=True,
    )

    def diag_closure() -> torch.Tensor:
        return m_diag(*pts)

    for _ in range(400):
        diag.step(diag_closure)
    diag_loss = _poisson_loss(m_diag, pts)

    assert csd_loss < diag_loss


# --- FrugalCurvature (memory-lean adaptivity) -----------------------------


def test_frugal_curvature_solves_quadratic() -> None:
    """The lean per-tensor exact-curvature preconditioner drives the quadratic to its minimum."""
    a = _spd(4, 71)
    b = torch.randn(4, generator=torch.Generator().manual_seed(72), dtype=torch.float64)
    model = _Quadratic(a, b)
    theta_star = torch.linalg.solve(a, b)

    opt = FrugalCurvature(
        model.parameters(), lr=5e-2, curvature="hutchinson", hutchinson_samples=8, curvature_every=1
    )
    for _ in range(6000):
        opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_star, atol=1e-2)


def test_frugal_curvature_multi_tensor_reduces_loss() -> None:
    """With several parameter tensors the per-tensor curvature preconditions each block; on a
    realisable (teacher) target the student fits it to a small residual."""
    torch.manual_seed(3)
    teacher = JetMLP(3, 5, 1, depth=2, base="tanh").double()
    x = torch.randn(16, 3, dtype=torch.float64)
    with torch.no_grad():
        y = teacher(x).squeeze(-1)
    net = JetMLP(3, 5, 1, depth=2, base="tanh").double()  # different init (RNG advanced)

    def closure() -> torch.Tensor:
        return ((net(x).squeeze(-1) - y) ** 2).mean()

    f0 = float(closure().detach())
    opt = FrugalCurvature(net.parameters(), lr=1e-2, curvature="hutchinson", curvature_every=5)
    for _ in range(400):
        opt.step(closure)
    assert float(closure().detach()) < 0.25 * f0


def test_frugal_curvature_gauss_newton_reduces_residual() -> None:
    """The exact Gauss-Newton per-tensor diagonal preconditions the stiff Poisson residual."""
    torch.manual_seed(0)
    model = _PoissonResidual(10).double()
    pts = _poisson_points(16)
    f0 = _poisson_loss(model, pts)

    opt = FrugalCurvature(
        model.parameters(), lr=5e-2, curvature="gauss_newton", curvature_every=5, clip=1.0
    )

    def closure() -> torch.Tensor:
        return model(*pts)

    for _ in range(200):
        opt.step(closure)
    assert _poisson_loss(model, pts) < f0


def test_frugal_curvature_memory_is_lean() -> None:
    """One O(P) momentum buffer + O(#tensors) curvature scalars, vs Adam's two O(P) buffers."""
    torch.manual_seed(0)
    net = JetMLP(3, 6, 1, depth=2, base="tanh").double()
    params = [p for p in net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    n_tensors = len(params)
    x = torch.randn(8, 3, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)

    def closure() -> torch.Tensor:
        return ((net(x).squeeze(-1) - y) ** 2).mean()

    opt = FrugalCurvature(net.parameters(), lr=1e-2, curvature="hutchinson", curvature_every=1)
    for _ in range(3):
        opt.step(closure)

    assert opt._d is None  # the O(P) curvature diagonal is never stored
    assert opt._m is not None and opt._m.numel() == n_params
    assert opt._c is not None and opt._c.numel() == n_tensors
    frugal_state = opt._m.numel() + opt._c.numel()

    net_a = JetMLP(3, 6, 1, depth=2, base="tanh").double()
    adam = torch.optim.Adam(net_a.parameters(), lr=1e-2)
    for _ in range(3):
        adam.zero_grad()
        ((net_a(x).squeeze(-1) - y) ** 2).mean().backward()
        adam.step()
    adam_state = sum(s["exp_avg"].numel() + s["exp_avg_sq"].numel() for s in adam.state.values())

    assert adam_state == 2 * n_params
    assert frugal_state == n_params + n_tensors
    assert frugal_state < adam_state


def test_frugal_curvature_sign_momentum_reduces_loss() -> None:
    """The Lion-like sign-momentum variant (scaled per tensor by exact curvature) descends."""
    torch.manual_seed(4)
    net = JetMLP(3, 5, 1, depth=2, base="tanh").double()
    x = torch.randn(16, 3, dtype=torch.float64)
    y = torch.randn(16, dtype=torch.float64)

    def closure() -> torch.Tensor:
        return ((net(x).squeeze(-1) - y) ** 2).mean()

    f0 = float(closure().detach())
    opt = FrugalCurvature(
        net.parameters(), lr=2e-3, curvature="hutchinson", curvature_every=5, sign_momentum=True
    )
    for _ in range(200):
        opt.step(closure)
    assert float(closure().detach()) < f0


def test_frugal_curvature_validation() -> None:
    """The constructor rejects out-of-range hyper-parameters."""
    p = [torch.zeros(2, requires_grad=True)]
    with pytest.raises(ValueError):
        FrugalCurvature(p, lr=0.0)
    with pytest.raises(ValueError):
        FrugalCurvature(p, curvature="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FrugalCurvature(p, beta1=1.0)
    with pytest.raises(ValueError):
        FrugalCurvature(p, beta2=1.0)
    with pytest.raises(ValueError):
        FrugalCurvature(p, eps=0.0)
    with pytest.raises(ValueError):
        FrugalCurvature(p, clip=0.0)
    with pytest.raises(ValueError):
        FrugalCurvature(p, curvature_every=0)
    with pytest.raises(ValueError):
        FrugalCurvature(p, hutchinson_samples=0)
    with pytest.raises(ValueError):
        FrugalCurvature(p, reduce="bogus")  # type: ignore[arg-type]


# --- Natural-gradient / Riemannian optimiser --------------------


class _LinearLeastSquares(nn.Module):
    """``0.5 mean((X theta - y)^2)`` -- a residual **linear** in ``theta`` (GN Fisher == Hessian)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float64))
        self.register_buffer("x", x)
        self.register_buffer("y", y)

    def residual(self) -> torch.Tensor:
        return self.x @ self.theta - self.y

    def loss(self) -> torch.Tensor:
        return 0.5 * (self.residual() ** 2).mean()


def _lstsq_design(n: int, p: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    theta_star = torch.linalg.lstsq(x, y).solution
    return x, y, theta_star


def test_natural_gradient_direction_dense_matches_damped_solve() -> None:
    """The dense metric path solves ``(M + mu I) delta = g`` exactly (== torch.linalg.solve)."""
    m = _spd(6, 21)
    g = torch.randn(6, generator=torch.Generator().manual_seed(22), dtype=torch.float64)
    for mu in (0.0, 1e-3, 1.0):
        delta = natural_gradient_direction(m, g, damping=mu)
        ref = torch.linalg.solve(m + mu * torch.eye(6, dtype=torch.float64), g)
        assert torch.allclose(delta, ref, atol=1e-10, rtol=0.0)


def test_natural_gradient_direction_matrix_free_matches_dense() -> None:
    """The matrix-free (CG) path reproduces the dense solve on an SPD metric."""
    m = _spd(7, 23)
    g = torch.randn(7, generator=torch.Generator().manual_seed(24), dtype=torch.float64)
    dense = natural_gradient_direction(m, g, damping=1e-2)
    free = natural_gradient_direction(lambda v: m @ v, g, damping=1e-2, cg_max_iter=200, cg_tol=1e-12)
    assert torch.allclose(dense, free, atol=1e-8, rtol=0.0)


def test_natural_gradient_direction_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        natural_gradient_direction(torch.eye(3, dtype=torch.float64), torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError):
        natural_gradient_direction(torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), damping=-1.0)
    with pytest.raises(ValueError):
        natural_gradient_direction(torch.eye(3, dtype=torch.float64), torch.zeros((3, 1), dtype=torch.float64))


def test_gauss_newton_fisher_equals_newton_on_linear_least_squares() -> None:
    """For a residual linear in ``theta`` the GN Fisher is the Hessian: one natural step recovers lstsq."""
    x, y, theta_star = _lstsq_design(20, 5, 30)

    def residual_fn(theta: torch.Tensor) -> torch.Tensor:
        return x @ theta - y

    theta0 = torch.zeros(5, dtype=torch.float64)
    fisher, g = gauss_newton_fisher(residual_fn, theta0)
    # Fisher == (1/N) X^T X, gradient == (1/N) X^T r  -- the exact least-squares normal operator.
    assert torch.allclose(fisher, (x.T @ x) / 20.0, atol=1e-10, rtol=0.0)
    assert torch.allclose(g, (x.T @ (x @ theta0 - y)) / 20.0, atol=1e-10, rtol=0.0)
    delta = natural_gradient_direction(fisher, g, damping=0.0)
    assert torch.allclose(theta0 - delta, theta_star, atol=1e-8)


def test_gauss_newton_fisher_matvec_matches_dense() -> None:
    """The matrix-free Fisher operator reproduces ``F v`` and the gradient of the dense builder."""
    x, y, _ = _lstsq_design(15, 4, 31)

    def residual_fn(theta: torch.Tensor) -> torch.Tensor:
        return x @ theta - y

    theta0 = torch.randn(4, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
    fisher, g = gauss_newton_fisher(residual_fn, theta0)
    _res, g_free, matvec = gauss_newton_fisher_matvec(residual_fn, theta0)
    assert torch.allclose(g_free, g, atol=1e-10, rtol=0.0)
    v = torch.randn(4, generator=torch.Generator().manual_seed(33), dtype=torch.float64)
    assert torch.allclose(matvec(v), fisher @ v, atol=1e-9, rtol=0.0)


def test_natural_gradient_dropin_recovers_quadratic_in_one_step() -> None:
    """With the exact Hessian as metric, one damping-free natural step lands on the minimiser."""
    a = _spd(6, 40)
    b = torch.randn(6, generator=torch.Generator().manual_seed(41), dtype=torch.float64)
    model = _Quadratic(a, b)
    theta_star = torch.linalg.solve(a, b)

    opt = NaturalGradient(model.parameters(), metric=lambda _th: a, lr=1.0, damping=0.0)
    opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_star, atol=1e-8)
    assert opt.n_iter == 1


def test_natural_gradient_dropin_with_fisher_provider_recovers_regression() -> None:
    """The Fisher provider wired through the drop-in recovers a linear regression in one step."""
    x, y, theta_star = _lstsq_design(24, 5, 42)
    model = _LinearLeastSquares(x, y)

    def residual_fn(theta: torch.Tensor) -> torch.Tensor:
        return x @ theta - y

    opt = NaturalGradient(
        model.parameters(),
        metric=lambda flat: gauss_newton_fisher(residual_fn, flat)[0],
        lr=1.0,
        damping=0.0,
    )
    opt.step(model.loss)
    assert torch.allclose(model.theta.detach(), theta_star, atol=1e-8)


def test_natural_gradient_identity_metric_reduces_loss() -> None:
    """metric=None is backtracked gradient descent: it monotonically reduces the loss."""
    a = _spd(5, 43)
    b = torch.randn(5, generator=torch.Generator().manual_seed(44), dtype=torch.float64)
    model = _Quadratic(a, b)
    opt = NaturalGradient(model.parameters(), metric=None, lr=1.0)
    f_prev = float(model.loss().detach())
    for _ in range(30):
        opt.step(model.loss)
        f_now = float(model.loss().detach())
        assert f_now <= f_prev + 1e-12  # monotone (backtracking guarantees descent)
        f_prev = f_now
    # a strictly convex quadratic is driven well below its starting value
    assert f_prev < 0.0


def test_natural_gradient_arbitrary_spd_metric_reduces_loss() -> None:
    """Any SPD metric provider (not the Hessian) still yields a descent step -- pluggability."""
    a = _spd(5, 45)
    b = torch.randn(5, generator=torch.Generator().manual_seed(46), dtype=torch.float64)
    model = _Quadratic(a, b)
    # A pullback-shaped metric g = C^T C + I (constant "chart" Jacobian C), unrelated to the Hessian.
    c = torch.randn(7, 5, generator=torch.Generator().manual_seed(47), dtype=torch.float64)
    metric = c.T @ c + torch.eye(5, dtype=torch.float64)
    f_start = float(model.loss().detach())
    opt = NaturalGradient(model.parameters(), metric=lambda _th: metric, lr=0.5, damping=1e-6)
    for _ in range(50):
        opt.step(model.loss)
    assert float(model.loss().detach()) < f_start


def test_natural_gradient_step_validation() -> None:
    a = _spd(3, 48)
    b = torch.zeros(3, dtype=torch.float64)
    model = _Quadratic(a, b)
    opt = NaturalGradient(model.parameters())
    with pytest.raises(ValueError):
        opt.step()  # no closure
    with pytest.raises(ValueError):
        opt.step(lambda: model.theta)  # non-scalar "loss"
    with pytest.raises(ValueError):
        NaturalGradient(model.parameters(), lr=0.0)
    with pytest.raises(ValueError):
        NaturalGradient(model.parameters(), damping=-1.0)
    with pytest.raises(ValueError):
        NaturalGradient(model.parameters(), max_line_search=0)


def test_natural_gradient_target_condition_certifies_and_floors_damping() -> None:
    """target_condition floors the damping so kappa(M + eps I) is provably <= T (eps -> 0 collapse)."""
    import numpy as np
    from omnibias.core.proof.certificate import verify_certificate_digest

    b = torch.zeros(5, dtype=torch.float64)
    metric = torch.diag(torch.tensor([1e6, 1e3, 1.0, 1e-1, 1e-3], dtype=torch.float64))  # kappa ~ 1e9
    model = _Quadratic(metric, b)  # loss 0.5 theta^T M theta (min at 0)
    model.theta.data = torch.ones(5, dtype=torch.float64)  # move off the minimum
    target = 1e3
    opt = NaturalGradient(
        model.parameters(), metric=lambda _th: metric, lr=1.0, damping=1e-12, target_condition=target
    )
    f0 = float(model.loss().detach())
    opt.step(model.loss)
    assert opt.last_certificate is not None
    assert verify_certificate_digest(opt.last_certificate)
    eps = opt.last_certificate["payload"]["certified_damping"]
    m = metric.numpy()
    assert np.linalg.cond(m + eps * np.eye(5)) <= target + 1e-6
    assert eps > 1e-12  # the certified floor genuinely raised the tiny damping
    assert float(model.loss().detach()) <= f0 + 1e-9  # still a descent step


def test_natural_gradient_target_condition_none_has_no_certificate() -> None:
    a = _spd(4, 60)
    b = torch.zeros(4, dtype=torch.float64)
    model = _Quadratic(a, b)
    opt = NaturalGradient(model.parameters(), metric=lambda _th: a, lr=1.0)
    opt.step(model.loss)
    assert opt.last_certificate is None


def test_natural_gradient_target_condition_validation() -> None:
    a = _spd(3, 61)
    b = torch.zeros(3, dtype=torch.float64)
    model = _Quadratic(a, b)
    with pytest.raises(ValueError, match="target_condition"):
        NaturalGradient(model.parameters(), target_condition=1.0)
