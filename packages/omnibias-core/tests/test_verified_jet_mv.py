# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the verified *multivariate* jet (``omnibias.core.verified.jet_mv``).

The certified jet must rigorously *enclose* every mixed partial ``D^alpha u(x0)``
for all ``x0`` in an input box.  Correctness is checked against an independent,
dependency-free float oracle for a single layer (where ``D^alpha u`` has the
closed form ``sigma^{(|alpha|)}(w·x+b) * prod w_i^{alpha_i}``), against the
production float kernel ``omnibias.torch.jet_mv`` when torch is available, and via
structural identities for the readouts and ``jet_multiply``.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.multi_index import multi_indices
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_mv import (
    certified_partials,
    certified_partials_subdivided,
    certified_residual_bound,
    compose_jet_mv,
    identity_jet,
    jet_gradient,
    jet_hessian,
    jet_laplacian,
    jet_multiply,
    jet_partials,
    layer_jet_mv,
    mlp_jet_mv,
)


# --------------------------------------------------------------------------- #
# Independent float oracle for the activation derivative towers.
# (Different code path from the interval Horner used inside the jet.)
# --------------------------------------------------------------------------- #
def _poly_eval(coeffs: list[float], t: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * t + c
    return acc


def _poly_deriv(coeffs: list[float]) -> list[float]:
    return [k * coeffs[k] for k in range(1, len(coeffs))] or [0.0]


def _poly_mul(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _tower_oracle(name: str, z: float, order: int) -> list[float]:
    """``(sigma(z), ..., sigma^(order)(z))`` via independent recurrences."""
    if name == "tanh":
        t = math.tanh(z)
        # P_0(t) = t ; P_{k+1} = P_k'(t) * (1 - t^2).
        poly = [0.0, 1.0]
        out = [_poly_eval(poly, t)]
        one_minus_sq = [1.0, 0.0, -1.0]
        for _ in range(order):
            poly = _poly_mul(_poly_deriv(poly), one_minus_sq)
            out.append(_poly_eval(poly, t))
        return out
    if name == "sigmoid":
        s = 1.0 / (1.0 + math.exp(-z))
        # P_0(s) = s ; P_{k+1} = P_k'(s) * (s - s^2).
        poly = [0.0, 1.0]
        out = [_poly_eval(poly, s)]
        s_minus_sq = [0.0, 1.0, -1.0]
        for _ in range(order):
            poly = _poly_mul(_poly_deriv(poly), s_minus_sq)
            out.append(_poly_eval(poly, s))
        return out
    if name == "gaussian":
        g = math.exp(-0.5 * z * z)
        # g^(n) = (-1)^n He_n(z) g ; He_{m+1} = z He_m - m He_{m-1}.
        out = [g]
        if order >= 1:
            out.append(-z * g)  # g' = -z g
        he_prev, he = 1.0, z  # He_0, He_1
        sign = -1.0  # (-1)^1
        for m in range(1, order):  # derivative order m + 1 = 2 .. order
            he_next = z * he - m * he_prev
            he_prev, he = he, he_next
            sign = -sign
            out.append(sign * he * g)
        return out
    raise ValueError(name)


def _single_layer(w: list[float], b: float, name: str):
    """A single scalar layer ``u(x) = sigma(w·x + b)`` as a verified ``layers`` list."""
    return [([list(w)], [b], name)]


def _analytic_partial(
    w: list[float], b: float, name: str, x0: list[float], alpha: tuple[int, ...]
) -> float:
    k = sum(alpha)
    a0 = sum(wi * xi for wi, xi in zip(w, x0, strict=True)) + b
    val = _tower_oracle(name, a0, k)[k]
    for wi, ai in zip(w, alpha, strict=True):
        val *= wi**ai
    return val


def _contains(enc: Interval, value: float, tol: float = 1e-9) -> bool:
    return enc.lo - tol <= value <= enc.hi + tol


# --------------------------------------------------------------------------- #
# Containment (the core soundness property).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["tanh", "sigmoid", "gaussian"])
def test_point_box_contains_and_is_tight(name: str) -> None:
    w, b = [0.7, -0.4], 0.2
    order = 4
    x0 = [0.3, -0.5]
    box = [(x0[0], x0[0]), (x0[1], x0[1])]
    part = certified_partials(box, _single_layer(w, b, name), order)
    for alpha in multi_indices(2, order):
        val = _analytic_partial(w, b, name, x0, alpha)
        enc = part[alpha][0]
        assert _contains(enc, val), (name, alpha, val, (enc.lo, enc.hi))
        assert enc.width < 1e-8  # degenerate box -> essentially a point enclosure


@pytest.mark.parametrize("name", ["tanh", "sigmoid", "gaussian"])
def test_wide_box_contains_interior_samples(name: str) -> None:
    w, b = [0.9, -0.6], 0.1
    order = 3
    box = [(0.1, 0.5), (-0.7, -0.2)]
    part = certified_partials(box, _single_layer(w, b, name), order)
    for sx in (0.1, 0.25, 0.4, 0.5):
        for sy in (-0.7, -0.5, -0.3, -0.2):
            for alpha in multi_indices(2, order):
                val = _analytic_partial(w, b, name, [sx, sy], alpha)
                assert _contains(part[alpha][0], val), (name, alpha, sx, sy)


def test_subdivision_shrinks_and_stays_sound() -> None:
    w, b, name = [1.1, -0.8], 0.0, "tanh"
    order = 3
    box = [(-0.5, 0.6), (-0.4, 0.5)]
    coarse = certified_partials_subdivided(box, _single_layer(w, b, name), order, 1)
    fine = certified_partials_subdivided(box, _single_layer(w, b, name), order, 4)
    for alpha in multi_indices(2, order):
        assert fine[alpha][0].width <= coarse[alpha][0].width + 1e-12
    # still sound after subdivision
    for sx in (-0.5, 0.0, 0.6):
        for sy in (-0.4, 0.1, 0.5):
            for alpha in multi_indices(2, order):
                val = _analytic_partial(w, b, name, [sx, sy], alpha)
                assert _contains(fine[alpha][0], val)


def test_subdivision_is_strictly_tighter_somewhere() -> None:
    w, b, name = [1.5, -1.2], 0.3, "tanh"
    order = 2
    box = [(-0.8, 0.8), (-0.8, 0.8)]
    coarse = certified_partials_subdivided(box, _single_layer(w, b, name), order, 1)
    fine = certified_partials_subdivided(box, _single_layer(w, b, name), order, 6)
    # at least one derivative enclosure should tighten noticeably
    assert any(
        fine[a][0].width < 0.9 * coarse[a][0].width for a in multi_indices(2, order)
    )


# --------------------------------------------------------------------------- #
# Readout consistency.
# --------------------------------------------------------------------------- #
def test_gradient_hessian_laplacian_readouts() -> None:
    w, b, name = [0.6, -0.5, 0.3], 0.1, "tanh"
    order = 2
    box = [(0.1, 0.2), (-0.3, -0.2), (0.0, 0.1)]
    jet = mlp_jet_mv(box, _single_layer(w, b, name), order)
    part = jet_partials(jet, 3, order)
    grad = jet_gradient(jet, 3, order)
    hess = jet_hessian(jet, 3, order)
    lap = jet_laplacian(jet, 3, order)
    for i in range(3):
        e = tuple(1 if j == i else 0 for j in range(3))
        assert grad[i][0].lo == part[e][0].lo and grad[i][0].hi == part[e][0].hi
    lap_from_part = Interval.point(0.0)
    for i in range(3):
        a = tuple(2 if j == i else 0 for j in range(3))
        assert (
            hess[i][i][0].lo == part[a][0].lo and hess[i][i][0].hi == part[a][0].hi
        )
        lap_from_part = lap_from_part + part[a][0]
    assert lap[0].lo == lap_from_part.lo and lap[0].hi == lap_from_part.hi
    # Hessian symmetry of the enclosures.
    for i in range(3):
        for j in range(3):
            assert hess[i][j][0].lo == hess[j][i][0].lo


def test_identity_jet_structure() -> None:
    box = [Interval(0.0, 1.0), Interval(-1.0, 2.0)]
    jet = identity_jet(box, 2)
    idx = multi_indices(2, 2)
    pos = {a: i for i, a in enumerate(idx)}
    assert jet[pos[(0, 0)]][0].lo == 0.0 and jet[pos[(0, 0)]][0].hi == 1.0
    assert jet[pos[(0, 0)]][1].lo == -1.0 and jet[pos[(0, 0)]][1].hi == 2.0
    assert jet[pos[(1, 0)]][0].lo == 1.0 and jet[pos[(1, 0)]][0].hi == 1.0
    assert jet[pos[(1, 0)]][1].lo == 0.0  # cross entry zero
    assert jet[pos[(0, 1)]][1].lo == 1.0 and jet[pos[(0, 1)]][1].hi == 1.0
    assert jet[pos[(2, 0)]][0].lo == 0.0 and jet[pos[(2, 0)]][0].hi == 0.0


# --------------------------------------------------------------------------- #
# compose_jet_mv vs layer_jet_mv.
# --------------------------------------------------------------------------- #
def test_compose_jet_mv_matches_layer_path() -> None:
    from omnibias.core.verified.jet import affine_jet
    from omnibias.core.verified.sigma import sigma_tower_interval

    w, b, name = [0.7, -0.3], 0.2, "sigmoid"
    order = 3
    box = [(0.0, 0.4), (-0.5, 0.1)]
    jet0 = identity_jet(box, order)
    u_jet = affine_jet(jet0, [list(w)], [b])
    tower = [
        [sigma_tower_interval(name, u_jet[0][0], order)[k]] for k in range(order + 1)
    ]
    composed = compose_jet_mv(u_jet, tower, 2, order)
    layer = layer_jet_mv(jet0, [list(w)], [b], name, 2, order)
    for g in range(len(composed)):
        assert composed[g][0].lo == layer[g][0].lo
        assert composed[g][0].hi == layer[g][0].hi


# --------------------------------------------------------------------------- #
# jet_multiply (the verified Leibniz rule).
# --------------------------------------------------------------------------- #
def test_jet_multiply_scales_by_constant() -> None:
    w, b, name = [0.7, -0.3], 0.2, "tanh"
    order = 3
    box = [(0.0, 0.4), (-0.5, 0.1)]
    f = mlp_jet_mv(box, _single_layer(w, b, name), order)
    m = len(f)
    const = [[Interval.point(3.0 if i == 0 else 0.0)] for i in range(m)]
    scaled = jet_multiply(const, f, 2, order)
    for g in range(m):
        assert abs(scaled[g][0].lo - 3.0 * f[g][0].lo) < 1e-12
        assert abs(scaled[g][0].hi - 3.0 * f[g][0].hi) < 1e-12


def test_jet_multiply_leibniz_contains_product_derivatives() -> None:
    # f = tanh layer, g = sigmoid layer; (fg) partials via Leibniz must be enclosed.
    wf, bf = [0.6, -0.4], 0.1
    wg, bg = [0.5, 0.3], -0.2
    order = 2
    x0 = [0.2, -0.3]
    box = [(x0[0], x0[0]), (x0[1], x0[1])]
    fj = mlp_jet_mv(box, _single_layer(wf, bf, "tanh"), order)
    gj = mlp_jet_mv(box, _single_layer(wg, bg, "sigmoid"), order)
    prod = jet_multiply(fj, gj, 2, order)
    part = jet_partials(prod, 2, order)
    # Leibniz: D^alpha(fg) = sum_{beta<=alpha} C(alpha,beta) D^beta f D^{alpha-beta} g.
    from math import comb

    def dpart(w, b, name, alpha):
        return _analytic_partial(w, b, name, x0, alpha)

    for alpha in multi_indices(2, order):
        total = 0.0
        a0, a1 = alpha
        for b0 in range(a0 + 1):
            for b1 in range(a1 + 1):
                beta = (b0, b1)
                rest = (a0 - b0, a1 - b1)
                c = comb(a0, b0) * comb(a1, b1)
                total += c * dpart(wf, bf, "tanh", beta) * dpart(wg, bg, "sigmoid", rest)
        assert _contains(part[alpha][0], total), (alpha, total, part[alpha][0])


def test_jet_multiply_column_broadcast() -> None:
    order = 1
    box = [(0.1, 0.1), (0.2, 0.2)]
    # vector field with 2 outputs (two tanh neurons), mask is scalar (M,1)
    layers = [([[0.5, 0.2], [-0.3, 0.6]], [0.0, 0.1], "tanh")]
    field = mlp_jet_mv(box, layers, order)
    m = len(field)
    mask = [[Interval.point(2.0 if i == 0 else 0.0)] for i in range(m)]
    out = jet_multiply(mask, field, 2, order)
    assert len(out[0]) == 2
    for g in range(m):
        for c in range(2):
            assert abs(out[g][c].lo - 2.0 * field[g][c].lo) < 1e-12


# --------------------------------------------------------------------------- #
# Certified PDE residual.
# --------------------------------------------------------------------------- #
def test_certified_residual_bound_brackets_laplacian() -> None:
    w, b, name = [0.8, -0.5], 0.0, "gaussian"
    order = 2
    box = [(-0.4, 0.4), (-0.3, 0.3)]

    def laplacian(part):
        return part[(2, 0)][0] + part[(0, 2)][0]

    coarse = certified_residual_bound(box, _single_layer(w, b, name), order, laplacian, 1)
    fine = certified_residual_bound(box, _single_layer(w, b, name), order, laplacian, 5)
    assert fine.width <= coarse.width + 1e-12
    # contains analytic Laplacian samples = g''(a)*|w|^2
    for sx in (-0.4, 0.0, 0.4):
        for sy in (-0.3, 0.0, 0.3):
            lap = _analytic_partial(w, b, name, [sx, sy], (2, 0)) + _analytic_partial(
                w, b, name, [sx, sy], (0, 2)
            )
            assert fine.lo - 1e-9 <= lap <= fine.hi + 1e-9


# --------------------------------------------------------------------------- #
# Error handling.
# --------------------------------------------------------------------------- #
def test_error_handling() -> None:
    box = [(0.0, 1.0), (0.0, 1.0)]
    with pytest.raises(ValueError):
        identity_jet(box, -1)
    jet0 = mlp_jet_mv(box, _single_layer([0.5, 0.5], 0.0, "tanh"), 0)
    with pytest.raises(ValueError):
        jet_gradient(jet0, 2, 0)
    jet1 = mlp_jet_mv(box, _single_layer([0.5, 0.5], 0.0, "tanh"), 1)
    with pytest.raises(ValueError):
        jet_hessian(jet1, 2, 1)
    with pytest.raises(ValueError):
        jet_laplacian(jet1, 2, 1)
    with pytest.raises(ValueError):
        mlp_jet_mv(box, [([[0.5, 0.5]], [0.0], "relu")], 2)
    with pytest.raises(ValueError):
        jet_multiply(jet1, jet0, 2, 1)  # row mismatch
    with pytest.raises(ValueError):
        certified_partials_subdivided(box, _single_layer([0.5, 0.5], 0.0, "tanh"), 1, 0)
    with pytest.raises(ValueError):
        certified_partials_subdivided(
            box, _single_layer([0.5, 0.5], 0.0, "tanh"), 1, [2]
        )


# --------------------------------------------------------------------------- #
# Parity with the production float kernel (skipped without torch).
# --------------------------------------------------------------------------- #
def test_torch_parity_containment() -> None:
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    from omnibias.torch.jet_mv import jet_partials as f_partials
    from omnibias.torch.jet_mv import mlp_jet_mv as f_mlp

    torch.manual_seed(0)
    dim, order = 2, 3
    W1 = torch.randn(4, 2) * 0.6
    b1 = torch.randn(4) * 0.3
    W2 = torch.randn(3, 4) * 0.5
    b2 = torch.randn(3) * 0.3
    W3 = torch.randn(1, 3) * 0.7
    b3 = torch.randn(1) * 0.2
    flayers = [(W1, b1, "tanh"), (W2, b2, "sigmoid"), (W3, b3, None)]

    def to_v(W, b, name):
        return (
            [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])],
            None if b is None else [float(x) for x in b],
            name,
        )

    vlayers = [to_v(W1, b1, "tanh"), to_v(W2, b2, "sigmoid"), to_v(W3, b3, None)]

    box = [(0.1, 0.5), (-0.6, -0.2)]
    vpart = certified_partials(box, vlayers, order)
    for cx in (0.1, 0.3, 0.5):
        for cy in (-0.6, -0.4, -0.2):
            fp = f_partials(f_mlp(torch.tensor([cx, cy]), flayers, order), dim, order)
            for alpha in multi_indices(dim, order):
                fv = float(fp[alpha][0])
                enc = vpart[alpha][0]
                assert enc.lo - 1e-9 <= fv <= enc.hi + 1e-9, (alpha, cx, cy)
