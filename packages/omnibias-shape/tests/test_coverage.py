# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Soft coverage: soft-OR properties and closed-form energy grad/Hessian vs autodiff (torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.shape.torch import ops as shape  # noqa: E402


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _problem(seed: int = 0):
    m, n, side, beta = 7, 8, 3.0, 2.0
    axes = (torch.arange(m, dtype=torch.float64), torch.arange(n, dtype=torch.float64))
    torch.manual_seed(seed)
    centers = torch.rand(3, 2) * torch.tensor([float(m), float(n)])
    gates = torch.tensor([0.9, 0.7, 0.8])
    ones = (torch.rand(m, n) > 0.4).to(torch.float64)
    return axes, centers, gates, ones, side, beta


def test_soft_or_in_unit_interval_and_monotone_in_gates():
    axes, centers, gates, _ones, side, beta = _problem()
    occ = shape.soft_box(axes, centers, side, beta)
    cov_lo, _ = shape.soft_or_coverage(occ, gates * 0.1)
    cov_hi, _ = shape.soft_or_coverage(occ, gates)
    assert torch.all(cov_hi > 0.0) and torch.all(cov_hi < 1.0)
    assert float(cov_hi.sum()) >= float(cov_lo.sum())


@pytest.mark.parametrize("loss", ["softplus", "sq_hinge"])
def test_coverage_energy_grad_and_hessian_match_autodiff(loss: str):
    axes, centers, gates, ones, side, beta = _problem(1)
    centers = centers.clone().requires_grad_(True)

    def energy(c: torch.Tensor) -> torch.Tensor:
        occ = shape.soft_box(axes, c, side, beta)
        return shape.coverage_energy(occ, gates, ones, loss=loss, kappa=1.5)

    g_auto = torch.autograd.functional.jacobian(energy, centers).reshape(-1)
    g_cf = shape.coverage_energy_grad(axes, centers.detach(), side, beta, gates, ones,
                                      loss=loss, kappa=1.5)
    h_auto = torch.autograd.functional.hessian(energy, centers).reshape(6, 6)
    h_cf = shape.coverage_energy_hessian(axes, centers.detach(), side, beta, gates, ones,
                                         loss=loss, kappa=1.5)
    assert torch.allclose(g_cf, g_auto, atol=1e-9)
    assert torch.allclose(h_cf, h_auto, atol=1e-9)
    assert torch.allclose(h_cf, h_cf.T, atol=1e-12)


@pytest.mark.parametrize("loss", ["softplus", "sq_hinge"])
@pytest.mark.parametrize("lam", [0.0, 0.5])
def test_coverage_energy_grad_hessian_wrt_all_matches_autodiff(loss: str, lam: float):
    axes, centers, gates, ones, side, beta = _problem(5)
    k = centers.shape[0]
    logits = torch.log(gates / (1.0 - gates))
    params = torch.cat([centers.reshape(-1), logits]).clone().requires_grad_(True)

    def energy(p: torch.Tensor) -> torch.Tensor:
        c = p[: 2 * k].reshape(k, 2)
        g = torch.sigmoid(p[2 * k :])
        occ = shape.soft_box(axes, c, side, beta)
        return shape.coverage_energy(occ, g, ones, loss=loss, kappa=1.5, lam=lam)

    g_auto = torch.autograd.functional.jacobian(energy, params).reshape(-1)
    h_auto = torch.autograd.functional.hessian(energy, params)
    g_cf = shape.coverage_energy_grad(
        axes, centers, side, beta, gates, ones, loss=loss, kappa=1.5, lam=lam, wrt="all"
    )
    h_cf = shape.coverage_energy_hessian(
        axes, centers, side, beta, gates, ones, loss=loss, kappa=1.5, lam=lam, wrt="all"
    )
    assert g_cf.shape == (2 * k + k,)
    assert h_cf.shape == (2 * k + k, 2 * k + k)
    assert torch.allclose(g_cf, g_auto, atol=1e-9)
    assert torch.allclose(h_cf, h_auto, atol=1e-8)
    assert torch.allclose(h_cf, h_cf.T, atol=1e-12)


def test_wrt_all_center_subblock_matches_centers_only():
    axes, centers, gates, ones, side, beta = _problem(6)
    g_c = shape.coverage_energy_grad(axes, centers, side, beta, gates, ones)
    h_c = shape.coverage_energy_hessian(axes, centers, side, beta, gates, ones)
    g_a = shape.coverage_energy_grad(axes, centers, side, beta, gates, ones, wrt="all")
    h_a = shape.coverage_energy_hessian(axes, centers, side, beta, gates, ones, wrt="all")
    kd = centers.numel()
    assert torch.allclose(g_c, g_a[:kd], atol=1e-14)
    assert torch.allclose(h_c, h_a[:kd, :kd], atol=1e-14)


@pytest.mark.parametrize("loss", ["softplus", "sq_hinge"])
@pytest.mark.parametrize("wrt", ["centers", "all"])
def test_background_term_grad_hessian_match_autodiff(loss: str, wrt: str):
    axes, centers, gates, ones, side, beta = _problem(8)
    bg = 1.0 - ones
    k, mu, lam = centers.shape[0], 0.7, 0.3
    logits = torch.log(gates / (1.0 - gates))

    if wrt == "all":
        params = torch.cat([centers.reshape(-1), logits]).clone().requires_grad_(True)

        def energy(p: torch.Tensor) -> torch.Tensor:
            c = p[: 2 * k].reshape(k, 2)
            g = torch.sigmoid(p[2 * k :])
            occ = shape.soft_box(axes, c, side, beta)
            return shape.coverage_energy(
                occ, g, ones, loss=loss, kappa=1.5, lam=lam, bg_mask=bg, mu=mu
            )
    else:
        params = centers.reshape(-1).clone().requires_grad_(True)

        def energy(p: torch.Tensor) -> torch.Tensor:
            c = p.reshape(k, 2)
            occ = shape.soft_box(axes, c, side, beta)
            return shape.coverage_energy(
                occ, gates, ones, loss=loss, kappa=1.5, lam=lam, bg_mask=bg, mu=mu
            )

    g_auto = torch.autograd.functional.jacobian(energy, params).reshape(-1)
    h_auto = torch.autograd.functional.hessian(energy, params)
    g_cf = shape.coverage_energy_grad(
        axes, centers, side, beta, gates, ones, loss=loss, kappa=1.5, lam=lam, bg_mask=bg, mu=mu, wrt=wrt
    )
    h_cf = shape.coverage_energy_hessian(
        axes, centers, side, beta, gates, ones, loss=loss, kappa=1.5, lam=lam, bg_mask=bg, mu=mu, wrt=wrt
    )
    assert torch.allclose(g_cf, g_auto, atol=1e-9)
    assert torch.allclose(h_cf, h_auto, atol=1e-8)


def test_background_mu_zero_reproduces_no_background():
    axes, centers, gates, ones, side, beta = _problem(9)
    bg = 1.0 - ones
    h0 = shape.coverage_energy_hessian(axes, centers, side, beta, gates, ones)
    h_mu0 = shape.coverage_energy_hessian(axes, centers, side, beta, gates, ones, bg_mask=bg, mu=0.0)
    assert torch.allclose(h0, h_mu0, atol=1e-14)


def test_gauss_newton_wrt_all_is_psd():
    axes, centers, gates, ones, side, beta = _problem(7)
    h_gn = shape.coverage_energy_hessian(
        axes, centers, side, beta, gates, ones, loss="sq_hinge", gauss_newton=True, wrt="all"
    )
    eig = torch.linalg.eigvalsh(0.5 * (h_gn + h_gn.T))
    assert float(eig.min()) >= -1e-9


@pytest.mark.parametrize("loss", ["softplus", "sq_hinge"])
@pytest.mark.parametrize("wrt", ["centers", "all"])
def test_lse_union_grad_hessian_match_autodiff(loss: str, wrt: str):
    axes, centers, gates, ones, side, beta = _problem(10)
    k, beta_u, lam = centers.shape[0], 8.0, 0.3
    logits = torch.log(gates / (1.0 - gates))

    if wrt == "all":
        params = torch.cat([centers.reshape(-1), logits]).clone().requires_grad_(True)

        def energy(p: torch.Tensor) -> torch.Tensor:
            c = p[: 2 * k].reshape(k, 2)
            g = torch.sigmoid(p[2 * k :])
            occ = shape.soft_box(axes, c, side, beta)
            return shape.coverage_energy(
                occ, g, ones, loss=loss, kappa=1.5, lam=lam, union="lse", beta_u=beta_u
            )
    else:
        params = centers.reshape(-1).clone().requires_grad_(True)

        def energy(p: torch.Tensor) -> torch.Tensor:
            c = p.reshape(k, 2)
            occ = shape.soft_box(axes, c, side, beta)
            return shape.coverage_energy(
                occ, gates, ones, loss=loss, kappa=1.5, lam=lam, union="lse", beta_u=beta_u
            )

    g_auto = torch.autograd.functional.jacobian(energy, params).reshape(-1)
    h_auto = torch.autograd.functional.hessian(energy, params)
    kw = dict(loss=loss, kappa=1.5, lam=lam, union="lse", beta_u=beta_u, wrt=wrt)
    g_cf = shape.coverage_energy_grad(axes, centers, side, beta, gates, ones, **kw)
    h_cf = shape.coverage_energy_hessian(axes, centers, side, beta, gates, ones, **kw)
    assert torch.allclose(g_cf, g_auto, atol=1e-9)
    assert torch.allclose(h_cf, h_auto, atol=1e-8)
    assert torch.allclose(h_cf, h_cf.T, atol=1e-10)


def test_lse_gauss_newton_is_psd():
    axes, centers, gates, ones, side, beta = _problem(11)
    h_gn = shape.coverage_energy_hessian(
        axes, centers, side, beta, gates, ones, loss="sq_hinge",
        union="lse", beta_u=8.0, gauss_newton=True, wrt="all",
    )
    eig = torch.linalg.eigvalsh(0.5 * (h_gn + h_gn.T))
    assert float(eig.min()) >= -1e-9


def test_invalid_union_raises():
    axes, centers, gates, ones, side, beta = _problem(0)
    with pytest.raises(ValueError, match="union"):
        shape.coverage_energy_hessian(axes, centers, side, beta, gates, ones, union="bogus")


def test_gauss_newton_hessian_is_psd_and_drops_residual_term():
    axes, centers, gates, ones, side, beta = _problem(2)
    h_full = shape.coverage_energy_hessian(axes, centers, side, beta, gates, ones, loss="sq_hinge")
    h_gn = shape.coverage_energy_hessian(axes, centers, side, beta, gates, ones,
                                         loss="sq_hinge", gauss_newton=True)
    eig = torch.linalg.eigvalsh(0.5 * (h_gn + h_gn.T))
    assert float(eig.min()) >= -1e-9
    assert not torch.allclose(h_full, h_gn)


def test_residual_objective_equals_sq_hinge_energy():
    axes, centers, gates, ones, side, beta = _problem(3)
    occ = shape.soft_box(axes, centers, side, beta)
    r = shape.coverage_residual(occ, gates, ones)
    e = shape.coverage_energy(occ, gates, ones, loss="sq_hinge")
    assert abs(float(0.5 * (r @ r)) - float(e)) < 1e-10


def test_lse_upper_bounds_max_membership():
    axes, centers, gates, _ones, side, beta = _problem(4)
    occ = shape.soft_box(axes, centers, side, beta)
    s = gates.reshape(-1, 1, 1) * occ
    lse = shape.lse_coverage(occ, gates, beta=20.0)
    assert torch.all(lse >= s.max(dim=0).values - 1e-6)
