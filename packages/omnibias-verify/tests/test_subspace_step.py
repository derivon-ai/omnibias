# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for the certified subspace trust-region step (proof-carrying optimization).

Soundness is checked the way the repo requires: the rigorous model-vs-truth
enclosure is compared against a dense grid **and** a random sample of the true
restricted loss, and every certified descent margin is confirmed against the
actual float loss decrease. A cross-package test feeds the torch subspace
machinery (``taylor_subspace_model`` / ``solve_subspace_trust_region``) into the
pure-Python certifier -- the actual fusion of the two bets.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation, lean_check_available
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    MLPArchitecture,
    ParamSpaceLoss,
    SubspaceModel,
    SubspaceStepCertificate,
    cauchy_step,
    certify_subspace_step,
    certify_trajectory,
    enclose_subspace_model,
    krylov_basis,
)
from omnibias.verify._core.subspace_step import _poly_value_at

# --------------------------------------------------------------------------- #
# Float reference model (independent of the interval machinery under test).
# --------------------------------------------------------------------------- #
Data = list[tuple[list[float], list[float]]]


def _net_out(arch: MLPArchitecture, theta: Sequence[float], x: Sequence[float]) -> list[float]:
    off = 0
    layers: list[tuple[list[list[float]], list[float]]] = []
    for n_out, n_in in arch.layer_shapes:
        weight = [[theta[off + r * n_in + c] for c in range(n_in)] for r in range(n_out)]
        off += n_out * n_in
        bias = [theta[off + r] for r in range(n_out)]
        off += n_out
        layers.append((weight, bias))
    a = [float(xi) for xi in x]
    last = len(layers) - 1
    for depth, (weight, bias) in enumerate(layers):
        z = []
        for o in range(len(weight)):
            acc = bias[o]
            for j in range(len(weight[o])):
                acc += weight[o][j] * a[j]
            z.append(acc)
        a = z if depth == last else [math.tanh(v) for v in z]
    return a


def _loss_float(arch: MLPArchitecture, theta: Sequence[float], data: Data, l2: float = 0.0) -> float:
    n = len(data)
    s = 0.0
    for x, y in data:
        out = _net_out(arch, theta, x)
        for o, yo in enumerate(y):
            s += (out[o] - yo) ** 2
    s /= n
    if l2:
        s += l2 * sum(t * t for t in theta)
    return s


def _psi_float(
    arch: MLPArchitecture,
    data: Data,
    theta0: Sequence[float],
    basis: Sequence[Sequence[float]],
    a: Sequence[float],
    l2: float = 0.0,
) -> float:
    k = len(a)
    theta = [theta0[p] + sum(basis[p][j] * a[j] for j in range(k)) for p in range(len(theta0))]
    return _loss_float(arch, theta, data, l2=l2)


def _make_data(arch: MLPArchitecture, teacher: Sequence[float], xs: Sequence[float]) -> Data:
    return [([x], [_net_out(arch, teacher, [x])[0]]) for x in xs]


# Small net: 1-1-1 (P = 4).
ARCH = MLPArchitecture(dims=(1, 1, 1))
TEACHER = [-1.3, 0.25, 1.2, 0.1]
DATA: Data = _make_data(ARCH, TEACHER, [-1.0 + 0.25 * i for i in range(9)])
THETA0 = [0.5, -0.3, 0.8, 0.2]
RADIUS = 0.05

# Bigger net: 1-2-1 (P = 7), reaches past a single unit.
ARCH2 = MLPArchitecture(dims=(1, 2, 1))
TEACHER2 = [0.7, -1.1, 0.3, -0.4, 0.9, -0.6, 0.2]
DATA2: Data = _make_data(ARCH2, TEACHER2, [-1.0 + 0.2 * i for i in range(11)])
THETA0_2 = [0.4, -0.5, 0.2, 0.6, -0.3, 0.5, -0.2]
RADIUS2 = 0.05


def _basis(arch: MLPArchitecture, data: Data, theta0: Sequence[float], k: int) -> tuple[tuple[float, ...], ...]:
    return krylov_basis(ParamSpaceLoss(arch, data), theta0, k)


# --------------------------------------------------------------------------- #
# Reduced-coefficient exactness.
# --------------------------------------------------------------------------- #
def test_reduced_gradient_and_hessian_match_finite_differences() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    k = len(basis[0])
    model = enclose_subspace_model(ARCH, DATA, THETA0, basis, radius=RADIUS)
    h = 1e-6
    for i in range(k):
        ap = [0.0] * k
        am = [0.0] * k
        ap[i] = h
        am[i] = -h
        fd = (_psi_float(ARCH, DATA, THETA0, basis, ap) - _psi_float(ARCH, DATA, THETA0, basis, am)) / (2 * h)
        assert model.grad[i].mid == pytest.approx(fd, abs=1e-5)
    for i in range(k):
        for j in range(k):
            ei = [0.0] * k
            ej = [0.0] * k
            ei[i] = h
            ej[j] = h
            fpp = _psi_float(ARCH, DATA, THETA0, basis, [ei[t] + ej[t] for t in range(k)])
            fpm = _psi_float(ARCH, DATA, THETA0, basis, [ei[t] - ej[t] for t in range(k)])
            fmp = _psi_float(ARCH, DATA, THETA0, basis, [-ei[t] + ej[t] for t in range(k)])
            fmm = _psi_float(ARCH, DATA, THETA0, basis, [-ei[t] - ej[t] for t in range(k)])
            fd = (fpp - fpm - fmp + fmm) / (4 * h * h)
            assert model.hessian[i][j].mid == pytest.approx(fd, abs=1e-3)


def test_reduced_coeffs_match_param_loss_projection() -> None:
    """The reduced grad/Hessian equal Q^T g and Q^T H Q from the interval ParamSpaceLoss."""
    basis = _basis(ARCH, DATA, THETA0, 2)
    k = len(basis[0])
    p_dim = ARCH.n_params
    problem = ParamSpaceLoss(ARCH, DATA)
    point = [Interval.point(t) for t in THETA0]
    g = [iv.mid for iv in problem.grad(point)]
    hmat = [[cell.mid for cell in row] for row in problem.hessian(point)]
    model = enclose_subspace_model(ARCH, DATA, THETA0, basis, radius=RADIUS)
    for i in range(k):
        reduced_g = sum(basis[p][i] * g[p] for p in range(p_dim))
        assert model.grad[i].mid == pytest.approx(reduced_g, abs=1e-7)
    for i in range(k):
        for j in range(k):
            reduced_h = sum(
                basis[p][i] * hmat[p][q] * basis[q][j] for p in range(p_dim) for q in range(p_dim)
            )
            assert model.hessian[i][j].mid == pytest.approx(reduced_h, abs=1e-6)


def test_constant_encloses_true_loss() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    model = enclose_subspace_model(ARCH, DATA, THETA0, basis, radius=RADIUS)
    true = _loss_float(ARCH, THETA0, DATA)
    assert model.constant.lo <= true <= model.constant.hi


# --------------------------------------------------------------------------- #
# Remainder soundness (the repo rule: dense grid + random samples).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("arch", "data", "theta0", "k", "radius"),
    [
        (ARCH, DATA, THETA0, 2, RADIUS),
        (ARCH2, DATA2, THETA0_2, 3, RADIUS2),
    ],
)
def test_remainder_encloses_true_model_error(
    arch: MLPArchitecture, data: Data, theta0: list[float], k: int, radius: float
) -> None:
    basis = _basis(arch, data, theta0, k)
    kk = len(basis[0])
    model = enclose_subspace_model(arch, data, theta0, basis, radius=radius)

    def _check(a: list[float]) -> None:
        true = _psi_float(arch, data, theta0, basis, a)
        m = _poly_value_at(model, a)
        assert m.lo + model.remainder.lo - 1e-12 <= true <= m.hi + model.remainder.hi + 1e-12

    grid = [-radius, -radius / 2, 0.0, radius / 2, radius]
    for combo in range(len(grid) ** kk):
        a = []
        rem = combo
        for _ in range(kk):
            a.append(grid[rem % len(grid)])
            rem //= len(grid)
        if math.sqrt(sum(ai * ai for ai in a)) <= radius + 1e-15:
            _check(a)

    rng = random.Random(1234)
    for _ in range(3000):
        a = [rng.uniform(-radius, radius) for _ in range(kk)]
        if math.sqrt(sum(ai * ai for ai in a)) <= radius:
            _check(a)


# --------------------------------------------------------------------------- #
# Certified descent.
# --------------------------------------------------------------------------- #
def test_certified_descent_bounds_the_true_decrease() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    cert = certify_subspace_step(ARCH, DATA, THETA0, basis, radius=RADIUS)
    assert isinstance(cert, SubspaceStepCertificate)
    assert cert.certified
    assert cert.verified
    assert cert.decrease_enclosure.lo > 0.0
    # The rigorous lower bound is an honest under-estimate of the real loss drop.
    actual = _loss_float(ARCH, THETA0, DATA) - _psi_float(ARCH, DATA, THETA0, basis, cert.step)
    assert actual >= cert.decrease_enclosure.lo - 1e-12
    assert actual <= cert.decrease_enclosure.hi + 1e-12


def test_certified_descent_scales_past_a_single_unit() -> None:
    basis = _basis(ARCH2, DATA2, THETA0_2, 3)
    cert = certify_subspace_step(ARCH2, DATA2, THETA0_2, basis, radius=RADIUS2)
    assert cert.certified
    assert cert.verified
    actual = _loss_float(ARCH2, THETA0_2, DATA2) - _psi_float(ARCH2, DATA2, THETA0_2, basis, cert.step)
    assert actual >= cert.decrease_enclosure.lo - 1e-12


def test_large_radius_is_honestly_not_certified() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    cert = certify_subspace_step(ARCH, DATA, THETA0, basis, radius=1.0)
    assert not cert.certified
    assert cert.decrease_enclosure.lo <= 0.0
    assert cert.verified  # a refused certificate is still a well-formed, sealed object
    assert cert.lean is None
    assert not cert.theorem_prover_verified


def test_explicit_step_is_accepted_and_certified() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    model = enclose_subspace_model(ARCH, DATA, THETA0, basis, radius=RADIUS)
    step = cauchy_step(tuple(g.mid for g in model.grad), RADIUS)
    cert = certify_subspace_step(ARCH, DATA, THETA0, basis, radius=RADIUS, step=step)
    assert cert.certified
    assert cert.step == pytest.approx(step)


def test_step_outside_trust_box_is_rejected() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    with pytest.raises(ValueError, match="outside the trust box"):
        certify_subspace_step(ARCH, DATA, THETA0, basis, radius=RADIUS, step=[2.0 * RADIUS, 0.0])


def test_l2_objective_is_certified_and_reflected_in_constant() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    model = enclose_subspace_model(ARCH, DATA, THETA0, basis, radius=RADIUS, l2=0.1)
    true = _loss_float(ARCH, THETA0, DATA, l2=0.1)
    assert model.constant.lo <= true <= model.constant.hi
    cert = certify_subspace_step(ARCH, DATA, THETA0, basis, radius=RADIUS, l2=0.1)
    assert cert.certified
    assert cert.descent_certificate["meta"]["l2"] == 0.1


# --------------------------------------------------------------------------- #
# Basis / step helpers.
# --------------------------------------------------------------------------- #
def test_krylov_basis_is_orthonormal() -> None:
    basis = _basis(ARCH, DATA, THETA0, 3)
    p_dim = ARCH.n_params
    k = len(basis[0])
    for i in range(k):
        col_i = [basis[p][i] for p in range(p_dim)]
        assert math.sqrt(sum(v * v for v in col_i)) == pytest.approx(1.0, abs=1e-9)
        for j in range(i + 1, k):
            col_j = [basis[p][j] for p in range(p_dim)]
            dot = sum(col_i[p] * col_j[p] for p in range(p_dim))
            assert dot == pytest.approx(0.0, abs=1e-9)


def test_cauchy_step_hits_the_boundary() -> None:
    step = cauchy_step([3.0, -4.0], 0.5)
    assert math.sqrt(sum(s * s for s in step)) == pytest.approx(0.5)
    # antiparallel to the gradient (a descent direction)
    assert step[0] < 0.0 and step[1] > 0.0


def test_cauchy_step_zero_gradient() -> None:
    assert cauchy_step([0.0, 0.0], 0.5) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# Certificate integrity + Lean gate.
# --------------------------------------------------------------------------- #
def test_certificate_tamper_is_detected() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    cert = certify_subspace_step(ARCH, DATA, THETA0, basis, radius=RADIUS)
    assert cert.verified
    forged = copy.deepcopy(cert.descent_certificate)
    forged["payload"]["interval"]["lo"] = forged["payload"]["interval"]["hi"]
    assert not verify_certificate_digest(forged)


def test_descent_obligation_is_lean_well_formed() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    cert = certify_subspace_step(ARCH, DATA, THETA0, basis, radius=RADIUS)
    obligation = generate_obligation(cert.descent_certificate)
    assert obligation is not None
    assert "enclosed_quantity_pos" in obligation


def test_lean_gate_runs_or_degrades_gracefully() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    cert = certify_subspace_step(ARCH, DATA, THETA0, basis, radius=RADIUS, lean=True)
    assert cert.certified
    assert cert.lean is not None
    if lean_check_available():
        assert cert.lean.verified is True
        assert cert.theorem_prover_verified is True
    else:
        assert cert.lean.available is False
        assert cert.theorem_prover_verified is False


# --------------------------------------------------------------------------- #
# Proof-carrying trajectory.
# --------------------------------------------------------------------------- #
def test_trajectory_is_certified_and_monotone() -> None:
    certs = certify_trajectory(ARCH, DATA, THETA0, radius=RADIUS, k=2, steps=5)
    assert len(certs) == 5
    theta = list(THETA0)
    prev = _loss_float(ARCH, theta, DATA)
    for cert in certs:
        assert cert.certified
        assert cert.verified
        kk = len(cert.step)
        theta = [
            theta[p] + sum(cert.basis[p][j] * cert.step[j] for j in range(kk))
            for p in range(ARCH.n_params)
        ]
        cur = _loss_float(ARCH, theta, DATA)
        assert cur <= prev + 1e-15
        prev = cur


def test_trajectory_shrinks_radius_when_a_step_is_not_certified() -> None:
    # A huge starting radius cannot certify; the loop must shrink it and still deliver steps.
    certs = certify_trajectory(ARCH, DATA, THETA0, radius=4.0, k=2, steps=2)
    assert len(certs) == 2
    assert all(c.certified for c in certs)
    assert certs[0].radius < 4.0


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def test_rejects_bad_radius_and_l2() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    with pytest.raises(ValueError, match="radius must be"):
        certify_subspace_step(ARCH, DATA, THETA0, basis, radius=0.0)
    with pytest.raises(ValueError, match="l2"):
        certify_subspace_step(ARCH, DATA, THETA0, basis, radius=RADIUS, l2=-1.0)


def test_rejects_wrong_shapes() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    with pytest.raises(ValueError, match="theta0 has"):
        certify_subspace_step(ARCH, DATA, THETA0[:-1], basis, radius=RADIUS)
    with pytest.raises(ValueError, match="basis has"):
        certify_subspace_step(ARCH, DATA, THETA0, basis[:-1], radius=RADIUS)


def test_enclose_returns_subspace_model() -> None:
    basis = _basis(ARCH, DATA, THETA0, 2)
    model = enclose_subspace_model(ARCH, DATA, THETA0, basis, radius=RADIUS)
    assert isinstance(model, SubspaceModel)
    assert model.k == 2
    assert len(model.grad) == 2
    assert len(model.hessian) == 2 and len(model.hessian[0]) == 2
    assert len(model.third) == 2


# --------------------------------------------------------------------------- #
# Fusion: torch subspace machinery -> pure-Python certifier (both bets, one step).
# --------------------------------------------------------------------------- #
def test_fusion_torch_subspace_step_is_certified() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.torch.optim import solve_subspace_trust_region, taylor_subspace_model

    dtype = torch.float64
    xs = [torch.tensor(x, dtype=dtype) for x, _ in DATA]
    ys = [torch.tensor(y, dtype=dtype) for _, y in DATA]

    def loss_fn(theta: torch.Tensor) -> torch.Tensor:
        off = 0
        layers = []
        for n_out, n_in in ARCH.layer_shapes:
            weight = theta[off : off + n_out * n_in].reshape(n_out, n_in)
            off += n_out * n_in
            bias = theta[off : off + n_out]
            off += n_out
            layers.append((weight, bias))
        total = theta.new_zeros(())
        last = len(layers) - 1
        for xt, yt in zip(xs, ys, strict=True):
            a = xt
            for depth, (weight, bias) in enumerate(layers):
                u = weight @ a + bias
                a = u if depth == last else torch.tanh(u)
            total = total + ((a - yt) ** 2).sum()
        return total / len(DATA)

    basis = _basis(ARCH, DATA, THETA0, 2)
    k = len(basis[0])
    p_dim = ARCH.n_params
    params = torch.tensor(THETA0, dtype=dtype)
    q_mat = torch.tensor([[basis[p][j] for j in range(k)] for p in range(p_dim)], dtype=dtype)

    c, hess, tensor3 = taylor_subspace_model(loss_fn, params, q_mat, order=3)
    a_star = solve_subspace_trust_region(c, hess, tensor3, radius=RADIUS)

    # The differentiable and rigorous registers agree on the reduced model...
    model = enclose_subspace_model(ARCH, DATA, THETA0, basis, radius=RADIUS)
    for i in range(k):
        assert model.grad[i].mid == pytest.approx(float(c[i]), abs=1e-7)
        for j in range(k):
            assert model.hessian[i][j].mid == pytest.approx(float(hess[i][j]), abs=1e-6)

    # ...and the rigorous register certifies the torch-computed trust-region step.
    cert = certify_subspace_step(
        ARCH, DATA, THETA0, basis, radius=RADIUS, step=[float(v) for v in a_star.tolist()]
    )
    assert cert.certified
    assert cert.verified
    actual = _loss_float(ARCH, THETA0, DATA) - _psi_float(ARCH, DATA, THETA0, basis, cert.step)
    assert actual >= cert.decrease_enclosure.lo - 1e-12
