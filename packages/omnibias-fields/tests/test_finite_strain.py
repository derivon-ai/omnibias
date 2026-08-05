# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite-strain solid mechanics (WS1): kinematics, hyperelasticity, balance laws.

Independent oracles:
1. analytic -- rigid rotation gives ``E = 0`` / ``W = 0``; incompressible
   deformation gives ``J = 1``; uniaxial neo-Hookean has a hand-derived stress;
2. internal-identity cross-checks -- ``pk*_stress`` (autodiff of the energy)
   equals the hand-derived closed-form StVK / neo-Hookean stress; the StVK
   tangent at ``F = I`` equals the isotropic elasticity tensor; the anisotropic
   Hooke law with an isotropic modulus equals ``linear_elastic_stress``; the
   finite-strain residual reduces to ``navier_cauchy_residual`` at small strain;
3. an independent numerical path -- the stress divergence ``Div(P)`` matches a
   central finite difference of the closed-form Piola stress at finite strain;
4. torch vs jax cross-backend parity to ``rtol=1e-9`` (float64).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from _analytic import Poly, make_field
from omnibias.fields.jax import ops as jo
from omnibias.fields.torch import ops as to

torch.set_default_dtype(torch.float64)


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _rand_F(batch=5, d=3, scale=0.15, seed=0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    return np.eye(d) + scale * rng.standard_normal((batch, d, d))


# ======================================================================
# 1. batched tensor algebra vs torch.linalg
# ======================================================================
def test_tensor_algebra_matches_linalg():  # type: ignore[no-untyped-def]
    f = torch.as_tensor(_rand_F(seed=1))
    assert torch.allclose(to.tensor_determinant(f), torch.linalg.det(f))
    assert torch.allclose(to.tensor_inverse(f), torch.linalg.inv(f))
    assert torch.allclose(to.tensor_trace(f),
                          torch.diagonal(f, dim1=-2, dim2=-1).sum(-1))
    assert torch.allclose(to.tensor_transpose(f), f.transpose(-1, -2))
    eye = torch.eye(3).expand(5, 3, 3)
    assert torch.allclose(to.tensor_matmul(f, to.tensor_inverse(f)), eye, atol=1e-10)
    # cofactor identity  A cof(A)^T = det(A) I
    cof = to.tensor_cofactor(f)
    lhs = torch.matmul(f, cof.transpose(-1, -2))
    assert torch.allclose(lhs, to.tensor_determinant(f)[:, None, None] * eye, atol=1e-9)


# ======================================================================
# 2. kinematics
# ======================================================================
def test_kinematics_definitions():  # type: ignore[no-untyped-def]
    f = torch.as_tensor(_rand_F(seed=2))
    c = to.right_cauchy_green(f)
    assert torch.allclose(c, f.transpose(-1, -2) @ f)
    e = to.green_lagrange_strain(f)
    assert torch.allclose(e, 0.5 * (c - torch.eye(3)))
    assert torch.allclose(to.jacobian_det(f), torch.linalg.det(f))


def test_rigid_rotation_has_zero_strain_and_energy():  # type: ignore[no-untyped-def]
    th = 0.7
    rot = torch.tensor([[np.cos(th), -np.sin(th), 0.0],
                        [np.sin(th), np.cos(th), 0.0],
                        [0.0, 0.0, 1.0]]).expand(3, 3, 3).contiguous()
    assert torch.allclose(to.green_lagrange_strain(rot),
                          torch.zeros(3, 3, 3), atol=1e-12)
    assert torch.allclose(to.st_venant_kirchhoff_energy(rot, lam=1.3, mu=0.8),
                          torch.zeros(3), atol=1e-12)
    assert torch.allclose(to.neo_hookean_energy(rot, lam=1.3, mu=0.8),
                          torch.zeros(3), atol=1e-12)


def test_incompressible_deformation_has_unit_jacobian():  # type: ignore[no-untyped-def]
    s = 1.0 / np.sqrt(2.0)
    f = torch.diag(torch.tensor([2.0, s, s])).unsqueeze(0)
    assert torch.allclose(to.jacobian_det(f), torch.ones(1), atol=1e-12)


# ======================================================================
# 3. stress: autodiff-of-energy == hand-derived closed form
# ======================================================================
def test_stvk_pk2_autodiff_matches_closed_form():  # type: ignore[no-untyped-def]
    f = torch.as_tensor(_rand_F(seed=3))
    s_auto = to.pk2_stress(f, lambda g: to.st_venant_kirchhoff_energy(g, lam=1.3, mu=0.8))
    s_cf = to.st_venant_kirchhoff_pk2(f, lam=1.3, mu=0.8)
    assert torch.allclose(s_auto, s_cf, atol=1e-10)
    assert torch.allclose(s_auto, s_auto.transpose(-1, -2), atol=1e-10)  # symmetric


def test_neo_hookean_pk2_autodiff_matches_closed_form():  # type: ignore[no-untyped-def]
    # This is the case that exposed the vmap-of-forward-mode-det defect.
    f = torch.as_tensor(_rand_F(seed=4))
    s_auto = to.pk2_stress(f, lambda g: to.neo_hookean_energy(g, lam=1.1, mu=0.9))
    s_cf = to.neo_hookean_pk2(f, lam=1.1, mu=0.9)
    assert torch.allclose(s_auto, s_cf, atol=1e-9)
    assert torch.allclose(s_auto, s_auto.transpose(-1, -2), atol=1e-9)


def test_neo_hookean_uniaxial_analytic():  # type: ignore[no-untyped-def]
    a, lam, mu = 1.2, 1.1, 0.9
    f = torch.diag(torch.tensor([a, 1.0, 1.0])).unsqueeze(0)
    p = to.pk1_stress(f, lambda g: to.neo_hookean_energy(g, lam=lam, mu=mu))
    # Neo-Hookean PK1 for F = diag(a, 1, 1):
    #   P11 = mu(a - 1/a) + lam ln(a)/a ;  P22 = P33 = mu(1 - 1) + lam ln(a).
    p11 = mu * (a - 1.0 / a) + lam * np.log(a) / a
    assert abs(float(p[0, 0, 0]) - p11) < 1e-10
    assert abs(float(p[0, 1, 1]) - lam * np.log(a)) < 1e-10


def test_pk1_pk2_cauchy_relations():  # type: ignore[no-untyped-def]
    f = torch.as_tensor(_rand_F(seed=5))
    energy = lambda g: to.neo_hookean_energy(g, lam=1.0, mu=1.0)  # noqa: E731
    p = to.pk1_stress(f, energy)
    s = to.pk2_stress(f, energy)
    sigma = to.cauchy_stress(f, energy)
    assert torch.allclose(p, f @ s, atol=1e-9)                       # P = F S
    j = torch.linalg.det(f)[:, None, None]
    assert torch.allclose(sigma, (p @ f.transpose(-1, -2)) / j, atol=1e-9)  # sigma = J^-1 P F^T
    assert torch.allclose(sigma, sigma.transpose(-1, -2), atol=1e-9)        # symmetric


def test_mooney_rivlin_stress_finite_difference():  # type: ignore[no-untyped-def]
    f = torch.as_tensor(_rand_F(batch=1, d=3, scale=0.1, seed=6))
    energy = lambda g: to.mooney_rivlin_energy(g, c1=0.6, c2=0.3, kappa=1.5)  # noqa: E731
    p = _np(to.pk1_stress(f, energy))[0]
    f0 = _np(f)[0]
    h = 1e-6
    p_fd = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            fp = f0.copy()
            fp[i, k] += h
            fm = f0.copy()
            fm[i, k] -= h
            wp = float(to.mooney_rivlin_energy(torch.as_tensor(fp[None]), c1=0.6, c2=0.3, kappa=1.5))
            wm = float(to.mooney_rivlin_energy(torch.as_tensor(fm[None]), c1=0.6, c2=0.3, kappa=1.5))
            p_fd[i, k] = (wp - wm) / (2 * h)
    assert np.allclose(p, p_fd, atol=1e-6)


def test_mooney_rivlin_requires_3d():  # type: ignore[no-untyped-def]
    f = torch.as_tensor(_rand_F(batch=2, d=2, seed=7))
    with pytest.raises(ValueError, match="3-D"):
        to.mooney_rivlin_energy(f)


# ======================================================================
# 4. anisotropic Hooke law
# ======================================================================
def test_hooke_general_isotropic_matches_linear_elastic():  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(8)
    strain = rng.standard_normal((6, 3, 3))
    strain = 0.5 * (strain + np.swapaxes(strain, -1, -2))  # symmetric
    lam, mu = 1.3, 0.8
    c = to.isotropic_stiffness(3, lam=lam, mu=mu)
    sig = _np(to.hooke_stress_general(torch.as_tensor(strain), c))
    eye = np.eye(3)
    tr = np.trace(strain, axis1=-2, axis2=-1)
    exp = 2 * mu * strain + lam * tr[:, None, None] * eye
    assert np.allclose(sig, exp, atol=1e-12)


def test_stvk_tangent_at_identity_is_isotropic_stiffness():  # type: ignore[no-untyped-def]
    from torch.func import jacrev, vmap
    lam, mu = 1.3, 0.8
    energy = lambda g: to.st_venant_kirchhoff_energy(g, lam=lam, mu=mu)  # noqa: E731
    f = torch.eye(3).unsqueeze(0)
    tangent = vmap(jacrev(jacrev(energy)))(f)[0]  # dP/dF at F=I
    c_iso = to.isotropic_stiffness(3, lam=lam, mu=mu)
    assert torch.allclose(tangent, c_iso, atol=1e-9)


# ======================================================================
# 5. field kinematics + balance laws
# ======================================================================
_DISP2 = {
    "u": (Poly([0.0, 0.30, 0.10]), Poly([1.0, 0.20, 0.05])),
    "v": (Poly([1.0, 0.15, 0.05]), Poly([0.0, 0.25, 0.10])),
    "fx": (Poly([0.1, 0.2, 0.0]), Poly([1.0, 0.1, 0.0])),
    "fy": (Poly([1.0, 0.1, 0.0]), Poly([0.2, 0.1, 0.0])),
}
_G2 = {"disp": ("u", "v")}


def _nodes2(seed=0, n=16):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.4, 0.4, size=(n, 2)).astype(np.float64)


def _state2(backend, nodes):  # type: ignore[no-untyped-def]
    f = make_field(backend, ("x", "y"), _DISP2, groups=_G2, time_axis=None)
    coords = torch.as_tensor(nodes) if backend == "torch" else jnp.asarray(nodes)
    return f(coords)


def test_deformation_gradient_finite():  # type: ignore[no-untyped-def]
    st = _state2("torch", _nodes2())
    f = to.deformation_gradient_finite(st, ("u", "v"))
    h = to.deformation_gradient(st, ("u", "v"))
    assert torch.allclose(f, h + torch.eye(2), atol=1e-12)
    assert (torch.linalg.det(f) > 0).all()  # physically valid over the node box


def test_finite_strain_divergence_matches_finite_difference():  # type: ignore[no-untyped-def]
    # Independent path: Div(P) from the tangent-modulus chain rule must equal a
    # central finite difference of the closed-form neo-Hookean Piola stress.
    lam, mu = 1.1, 0.9
    energy = lambda g: to.neo_hookean_energy(g, lam=lam, mu=mu)  # noqa: E731
    nodes = _nodes2(seed=3)
    st = _state2("torch", nodes)
    div_lib = _np(to.finite_strain_residual(st, ("u", "v"), energy))

    field = make_field("torch", ("x", "y"), _DISP2, groups=_G2, time_axis=None)

    def pk1_closed(coords):  # type: ignore[no-untyped-def]
        s = field(torch.as_tensor(coords))
        f = to.deformation_gradient_finite(s, ("u", "v"))
        finv_t = torch.linalg.inv(f).transpose(-1, -2)
        ln_j = torch.log(torch.linalg.det(f))
        return _np(mu * (f - finv_t) + lam * ln_j[:, None, None] * finv_t)

    hh = 1e-4
    d = 2
    div_fd = np.zeros((nodes.shape[0], d))
    for j in range(d):
        cp = nodes.copy()
        cp[:, j] += hh
        cm = nodes.copy()
        cm[:, j] -= hh
        dP = (pk1_closed(cp) - pk1_closed(cm)) / (2 * hh)  # (B, i, J)
        div_fd[:, :] += dP[:, :, j]
    assert np.allclose(div_lib, div_fd, atol=1e-6)


def test_finite_strain_small_strain_limit_navier_cauchy():  # type: ignore[no-untyped-def]
    # StVK finite-strain residual -> Navier-Cauchy as the displacement -> 0.
    eps = 1e-3
    small = {k: tuple(Poly([eps * c for c in p.coeffs]) for p in v) if k in ("u", "v") else v
             for k, v in _DISP2.items()}
    lam, mu = 1.2, 0.7
    nodes = _nodes2(seed=5)
    field = make_field("torch", ("x", "y"), small, groups=_G2, time_axis=None)
    st = field(torch.as_tensor(nodes))
    fs = _np(to.finite_strain_residual(
        st, ("u", "v"), lambda g: to.st_venant_kirchhoff_energy(g, lam=lam, mu=mu)))
    nc = _np(to.navier_cauchy_residual(st, displacement=("u", "v"), lam=lam, mu=mu))
    assert np.allclose(fs, nc, rtol=1e-2, atol=1e-7)


_DISP2T = {
    "u": (Poly([0.0, 0.2]), Poly([1.0, 0.15, 0.05]), Poly([1.0, 0.1])),
    "v": (Poly([1.0, 0.1]), Poly([0.0, 0.2, 0.05]), Poly([1.0, 0.2])),
}


def test_elastodynamic_residual_recomposition():  # type: ignore[no-untyped-def]
    field = make_field("torch", ("t", "x", "y"), _DISP2T, groups={"disp": ("u", "v")},
                       time_axis="t")
    rng = np.random.default_rng(9)
    nodes = rng.uniform(-0.3, 0.3, size=(12, 3)).astype(np.float64)
    st = field(torch.as_tensor(nodes))
    rho = 2.5
    energy = lambda g: to.neo_hookean_energy(g, lam=1.0, mu=1.0)  # noqa: E731
    res = _np(to.elastodynamic_residual(st, ("u", "v"), energy, density=rho))
    u_tt = np.stack([_np(to.derivative(st, n, axis="t", order=2)) for n in ("u", "v")], -1)
    div_p = _np(to.finite_strain_residual(st, ("u", "v"), energy))  # Div(P), no body force
    assert np.allclose(res, rho * u_tt - div_p, atol=1e-9)


# ======================================================================
# 6. cross-backend parity
# ======================================================================
def test_pointwise_stress_cross_backend():  # type: ignore[no-untyped-def]
    fnp = _rand_F(seed=10)
    ft, fj = torch.as_tensor(fnp), jnp.asarray(fnp)
    et = lambda g: to.neo_hookean_energy(g, lam=1.1, mu=0.9)  # noqa: E731
    ej = lambda g: jo.neo_hookean_energy(g, lam=1.1, mu=0.9)  # noqa: E731
    for fn in ("pk1_stress", "pk2_stress", "cauchy_stress"):
        t = _np(getattr(to, fn)(ft, et))
        j = _np(getattr(jo, fn)(fj, ej))
        assert np.allclose(t, j, rtol=1e-9, atol=1e-9), fn
    # closed-form + kinematics
    assert np.allclose(_np(to.green_lagrange_strain(ft)),
                       _np(jo.green_lagrange_strain(fj)), rtol=1e-9, atol=1e-9)


def test_residual_cross_backend():  # type: ignore[no-untyped-def]
    nodes = _nodes2(seed=12)
    ts = _state2("torch", nodes)
    js = _state2("jax", nodes)
    et = lambda g: to.neo_hookean_energy(g, lam=1.1, mu=0.9)  # noqa: E731
    ej = lambda g: jo.neo_hookean_energy(g, lam=1.1, mu=0.9)  # noqa: E731
    t = _np(to.finite_strain_residual(ts, ("u", "v"), et, body_force=("fx", "fy")))
    j = _np(jo.finite_strain_residual(js, ("u", "v"), ej, body_force=("fx", "fy")))
    assert np.allclose(t, j, rtol=1e-9, atol=1e-9)
