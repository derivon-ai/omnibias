# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the PDE-residual / certificate wiring adaptor.

Covers the physical-space (trained-MLP) route -- exact residuals on affine nets,
whole-box soundness against point-box evaluations, subdivision tightening, an
independent torch double-grad parity check -- the sealed a-posteriori error
certificate, the radii-polynomial bridge, and the spectral (Fourier) route.
"""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.fourier import ValidatedFourierSeries as VFS
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import radii_polynomial_certificate
from omnibias.core.verified.pde_certificate import (
    BoundaryFace,
    LinearPDE,
    adaptive_certified_interior_residual,
    advection_diffusion,
    aposteriori_error_certificate,
    certified_boundary_residual,
    certified_custom_residual,
    certified_interior_residual,
    certified_quadratic_reaction_residual,
    helmholtz,
    laplace,
    pinn_aposteriori_schema_errors,
    poisson,
    radii_polynomial_residual_certificate,
    replay_pinn_aposteriori_certificate,
    screened_poisson,
    spectral_residual_norm,
    structural_invariant,
    user_stability_estimate,
)

# Affine scalar net  u(x, y) = w0 x + w1 y + b.
_W0, _W1, _B = 2.0, -3.0, 1.0
_AFFINE = [([[_W0, _W1]], [_B], None)]
_DOMAIN = [(-1.0, 1.0), (-1.0, 1.0)]


def _u_affine(x: float, y: float) -> float:
    return _W0 * x + _W1 * y + _B


# --------------------------------------------------------------------------- #
# LinearPDE DSL basics.
# --------------------------------------------------------------------------- #
def test_required_order() -> None:
    assert laplace(2).required_order() == 2
    assert advection_diffusion(2, [1.0, 1.0], 0.5).required_order() == 2
    assert LinearPDE({}, source=3.0).required_order() == 0  # pure source


def test_advection_diffusion_velocity_length() -> None:
    with pytest.raises(ValueError):
        advection_diffusion(2, [1.0], 0.5)


# --------------------------------------------------------------------------- #
# Exact residuals on an affine network (closed-form ground truth).
# --------------------------------------------------------------------------- #
def test_affine_laplacian_and_poisson_are_zero() -> None:
    assert certified_interior_residual(_AFFINE, _DOMAIN, laplace(2), splits=2).mag < 1e-9
    assert (
        certified_interior_residual(_AFFINE, _DOMAIN, poisson(2, 0.0), splits=2).mag
        < 1e-9
    )


def test_affine_advection_diffusion_matches_closed_form() -> None:
    # b . grad u - nu lap u - f = (1,1).(w0,w1) - 0 - f  (constant).
    pde = advection_diffusion(2, [1.0, 1.0], 0.5, 0.25)
    r = certified_interior_residual(_AFFINE, _DOMAIN, pde)
    exact = 1.0 * _W0 + 1.0 * _W1 - 0.25
    assert r.lo <= exact <= r.hi
    assert r.mag < abs(exact) + 1e-9


def test_affine_helmholtz_encloses_pointwise_value() -> None:
    pde = helmholtz(2, 1.0, 0.0)  # lap u + u - 0 = u  (affine lap = 0)
    for x, y in [(0.3, -0.2), (-0.7, 0.5), (1.0, 1.0)]:
        r = certified_interior_residual(_AFFINE, [(x, x), (y, y)], pde)
        assert r.lo <= _u_affine(x, y) <= r.hi


def test_screened_poisson_sign() -> None:
    pde = screened_poisson(2, 2.0, 0.0)  # lap u - 4 u = -4 u  for affine
    x, y = 0.5, 0.5
    r = certified_interior_residual(_AFFINE, [(x, x), (y, y)], pde)
    assert r.lo <= -4.0 * _u_affine(x, y) <= r.hi


# --------------------------------------------------------------------------- #
# Soundness + tightening on a nonlinear (tanh) network.
# --------------------------------------------------------------------------- #
def _tanh_net() -> list:
    rng = random.Random(0)
    h = 4
    w = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(h)]
    bb = [rng.uniform(-1, 1) for _ in range(h)]
    v = [[rng.uniform(-1, 1) for _ in range(h)]]
    return [(w, bb, "tanh"), (v, [0.2], None)]


def test_whole_box_encloses_point_box_residuals() -> None:
    net = _tanh_net()
    pde = laplace(2)
    whole = certified_interior_residual(net, _DOMAIN, pde, splits=1)
    rng = random.Random(1)
    for _ in range(200):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        pt = certified_interior_residual(net, [(x, x), (y, y)], pde)
        # a point of the box -> its residual lies inside the whole-box enclosure
        assert whole.lo - 1e-12 <= pt.lo and pt.hi <= whole.hi + 1e-12


def test_subdivision_tightens() -> None:
    net = _tanh_net()
    pde = laplace(2)
    coarse = certified_interior_residual(net, _DOMAIN, pde, splits=1)
    fine = certified_interior_residual(net, _DOMAIN, pde, splits=6)
    assert fine.mag <= coarse.mag + 1e-12


def test_torch_double_grad_parity() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    h = 4
    w = torch.randn(h, 2, dtype=torch.float64)
    bb = torch.randn(h, dtype=torch.float64)
    v = torch.randn(1, h, dtype=torch.float64)
    bo = torch.randn(1, dtype=torch.float64)

    def u(p: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(p @ w.T + bb) @ v.T + bo).squeeze(-1)

    net = [
        (w.tolist(), bb.tolist(), "tanh"),
        (v.tolist(), bo.tolist(), None),
    ]
    pde = laplace(2)
    for px, py in [(0.3, -0.2), (-0.6, 0.4)]:
        p = torch.tensor([px, py], dtype=torch.float64, requires_grad=True)
        grad = torch.autograd.grad(u(p), p, create_graph=True)[0]
        lap = sum(
            torch.autograd.grad(grad[i], p, retain_graph=True)[0][i] for i in range(2)
        )
        r = certified_interior_residual(net, [(px, px), (py, py)], pde)
        assert r.lo - 1e-9 <= float(lap) <= r.hi + 1e-9


# --------------------------------------------------------------------------- #
# Boundary residual.
# --------------------------------------------------------------------------- #
def test_boundary_residual_encloses_face_values() -> None:
    # right edge x = 1, y in [-1, 1], target g = 0  ->  residual = u_NN on the face.
    face = BoundaryFace([(1.0, 1.0), (-1.0, 1.0)], 0.0)
    r = certified_boundary_residual(_AFFINE, [face], splits=4)
    for y in (-0.5, 0.0, 0.5):
        assert r.lo <= _u_affine(1.0, y) <= r.hi


def test_boundary_residual_empty_is_zero() -> None:
    assert certified_boundary_residual(_AFFINE, []).mag < 1e-300


# --------------------------------------------------------------------------- #
# A-posteriori error certificate.
# --------------------------------------------------------------------------- #
def test_aposteriori_exact_solution_has_tiny_error() -> None:
    # affine net is harmonic; no boundary term -> error bound ~ 0.
    cert = aposteriori_error_certificate(
        _AFFINE, _DOMAIN, laplace(2), stability_interior=0.5, splits=2
    )
    assert cert.error_bound < 1e-9
    assert cert.boundary_residual < 1e-300  # no boundary faces supplied
    assert verify_certificate_digest(cert.certificate)
    assert cert.certificate["payload"]["type"] == "pinn_aposteriori_error"
    assert pinn_aposteriori_schema_errors(cert.certificate) == []
    assert replay_pinn_aposteriori_certificate(cert.certificate)


def test_aposteriori_combines_residuals() -> None:
    net = _tanh_net()
    face = BoundaryFace([(1.0, 1.0), (-1.0, 1.0)], 0.0)
    ci, cb = 0.7, 1.3
    cert = aposteriori_error_certificate(
        net,
        _DOMAIN,
        laplace(2),
        boundary=[face],
        stability_interior=ci,
        stability_boundary=cb,
        splits=2,
        boundary_splits=3,
    )
    r_int = certified_interior_residual(net, _DOMAIN, laplace(2), splits=2).mag
    r_bnd = certified_boundary_residual(net, [face], splits=3).mag
    expected = (
        Interval.point(ci) * Interval.point(r_int)
        + Interval.point(cb) * Interval.point(r_bnd)
    ).hi
    assert cert.interior_residual == r_int
    assert cert.boundary_residual == r_bnd
    assert cert.error_bound == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_aposteriori_negative_stability_raises() -> None:
    with pytest.raises(ValueError):
        aposteriori_error_certificate(_AFFINE, _DOMAIN, laplace(2), stability_interior=-1.0)


def test_aposteriori_records_stability_invariants_and_formal_margin() -> None:
    stability = user_stability_estimate(
        0.5,
        1.0,
        source="maximum-principle fixture",
        pde_family="laplace",
        domain="unit-square",
        assumptions=("linear well-posed BVP",),
    )
    inv = structural_invariant(
        "streamfunction_2d",
        "u = d_y psi, v = -d_x psi => div u = 0",
        assumptions=("mixed partials commute",),
    )
    cert = aposteriori_error_certificate(
        _AFFINE,
        _DOMAIN,
        laplace(2),
        stability=stability,
        invariants=[inv],
        max_error=1e-6,
        splits=2,
    )
    payload = cert.certificate["payload"]
    assert payload["stability"]["source"] == "maximum-principle fixture"
    assert payload["invariants"][0]["kind"] == "streamfunction_2d"
    assert payload["finite_obligation"]["type"] == "error_bound_le_threshold"
    assert generate_obligation(cert.certificate) is not None


def test_adaptive_residual_diagnostics_reaches_target() -> None:
    diag = adaptive_certified_interior_residual(
        _AFFINE, _DOMAIN, laplace(2), target=1e-9, initial_splits=1, max_splits=4
    )
    assert diag.reached_target is True
    assert diag.residual_sup < 1e-9
    assert diag.boxes == 1


def test_custom_and_quadratic_residual_hooks() -> None:
    def residual(box, partials):
        return partials[(0,) * len(box)][0]

    r = certified_custom_residual(_AFFINE, [(0.5, 0.5), (0.25, 0.25)], residual, order=0)
    assert r.lo <= _u_affine(0.5, 0.25) <= r.hi

    q = certified_quadratic_reaction_residual(
        _AFFINE, [(0.0, 0.0), (0.0, 0.0)], laplace(2), coefficient=2.0
    )
    assert q.lo <= 2.0 * (_u_affine(0.0, 0.0) ** 2) <= q.hi


def test_pinn_schema_rejects_forged_structural_claim() -> None:
    cert = aposteriori_error_certificate(_AFFINE, _DOMAIN, laplace(2), splits=2).certificate
    cert["payload"]["invariants"] = [{"kind": "divergence_free", "certified": True}]
    assert any("expression" in err for err in pinn_aposteriori_schema_errors(cert))


# --------------------------------------------------------------------------- #
# Radii-polynomial bridge.
# --------------------------------------------------------------------------- #
def test_radii_bridge_matches_direct_call() -> None:
    res, a_norm, z0, z1, z2 = 1e-4, 2.0, 0.1, 0.0, 0.5
    bridged = radii_polynomial_residual_certificate(res, z0, z1, z2, a_norm=a_norm)
    direct = radii_polynomial_certificate(a_norm * res, z0, z1, z2)
    assert bridged is not None and direct is not None
    # the bridge applies ||A|| via sound interval arithmetic, so Y0 is >= the
    # plain float product and matches it to rounding.
    assert bridged.y0 >= a_norm * res
    assert bridged.y0 == pytest.approx(a_norm * res, rel=1e-12)
    assert bridged.radius == pytest.approx(direct.radius, rel=1e-9)
    assert verify_certificate_digest(bridged.certificate)


def test_radii_bridge_no_contraction_returns_none() -> None:
    # huge defect -> discriminant negative -> no certificate.
    assert radii_polynomial_residual_certificate(10.0, 0.1, 0.0, 0.5) is None


def test_radii_bridge_negative_inputs_raise() -> None:
    with pytest.raises(ValueError):
        radii_polynomial_residual_certificate(-1.0, 0.1, 0.0, 0.5)
    with pytest.raises(ValueError):
        radii_polynomial_residual_certificate(1e-4, 0.1, 0.0, 0.5, a_norm=-1.0)


# --------------------------------------------------------------------------- #
# Spectral (Fourier) route.
# --------------------------------------------------------------------------- #
def _zero_symbol(_k: tuple[int, ...]) -> ComplexInterval:
    return ComplexInterval.zero()


def test_spectral_linear_exact_residual_is_tiny() -> None:
    # L = 2 I, rhs = 2 a  ->  F(a) = 2a - 2a = 0.
    dim, n, nu = 1, 3, 1.0
    a = VFS.from_coeffs({(1,): 1.0, (-1,): 1.0}, dim, n, nu)

    def double(_k: tuple[int, ...]) -> ComplexInterval:
        return ComplexInterval.from_value(2.0)

    res = spectral_residual_norm(a, double, 2.0, rhs=a.scale(2.0))
    assert res.hi < 1e-12


def test_spectral_quadratic_matches_hand_computation() -> None:
    # a(x) = 2 cos x ; (a*a) has coeffs {-2:1, 0:2, 2:1}; L = 0, no rhs.
    dim, n, nu = 1, 4, 1.0
    a = VFS.from_coeffs({(1,): 1.0, (-1,): 1.0}, dim, n, nu)
    res = spectral_residual_norm(a, _zero_symbol, 0.0, quadratic=lambda s: s * s)
    assert res.hi == pytest.approx(4.0, abs=1e-12)  # 1 + 2 + 1, nu = 1


def test_spectral_feeds_radii_bridge() -> None:
    dim, n, nu = 1, 3, 1.0
    a = VFS.from_coeffs({(1,): 1e-3, (-1,): 1e-3}, dim, n, nu)
    res = spectral_residual_norm(a, _zero_symbol, 0.0, quadratic=lambda s: s * s)
    cert = radii_polynomial_residual_certificate(res.hi, 0.1, 0.0, 0.5)
    assert cert is not None
    assert math.isfinite(cert.radius)
