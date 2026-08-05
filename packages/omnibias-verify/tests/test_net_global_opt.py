# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified global minimization of an omnibias network's scalar output.

The bridge feeds the closed-form *verified* jet (value = jet row 0, gradient =
:func:`jet_gradient`, Hessian = :func:`jet_hessian`) into the interval
branch-and-bound.  Every test is either

* **soundness** -- the returned ``f_lower`` under-estimates a dense grid + random
  sample of the true float network output, and the enclosure brackets the grid min;
* **derivative soundness** -- the closed-form interval gradient encloses a
  finite-difference gradient at random points;
* **behaviour** -- the exact gradient accelerates the search, strict-local-min
  certification is sound, and the duck-typed ``_layer_specs`` path matches raw layers.
"""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_mv import jet_gradient, mlp_jet_mv
from omnibias.verify import (
    certified_network_critical_points,
    certified_network_flatness,
    certified_network_minimize,
    certify_network_strict_local_min,
)

# --------------------------- float reference forward ------------------------

Layer = tuple[list[list[float]], list[float] | None, str | None]


def eval_net(layers: list[Layer], x: list[float]) -> list[float]:
    v = list(x)
    for weight, bias, name in layers:
        pre = []
        for i, row in enumerate(weight):
            s = sum(row[j] * v[j] for j in range(len(v)))
            if bias is not None:
                s += bias[i]
            pre.append(s)
        if name == "tanh":
            v = [math.tanh(z) for z in pre]
        elif name == "sigmoid":
            v = [1.0 / (1.0 + math.exp(-z)) for z in pre]
        elif name == "gaussian":
            v = [math.exp(-z * z / 2.0) for z in pre]
        elif name == "silu":
            v = [z / (1.0 + math.exp(-z)) for z in pre]
        elif name == "softplus":
            v = [max(z, 0.0) + math.log1p(math.exp(-abs(z))) for z in pre]
        elif name == "gelu":
            v = [z * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) for z in pre]
        elif name is None:
            v = pre
        else:  # pragma: no cover - defensive
            raise ValueError(name)
    return v


def grid_min(layers: list[Layer], box: list[tuple[float, float]], component: int, n: int = 61) -> float:
    (xlo, xhi), (ylo, yhi) = box
    best = math.inf
    for i in range(n):
        for j in range(n):
            x = xlo + (xhi - xlo) * i / (n - 1)
            y = ylo + (yhi - ylo) * j / (n - 1)
            best = min(best, eval_net(layers, [x, y])[component])
    return best


def assert_sound(layers: list[Layer], box: list[tuple[float, float]], f_lower: float, component: int = 0, n_rand: int = 4000, seed: int = 0) -> float:
    rng = random.Random(seed)
    worst = grid_min(layers, box, component)
    (xlo, xhi), (ylo, yhi) = box
    for _ in range(n_rand):
        x = xlo + (xhi - xlo) * rng.random()
        y = ylo + (yhi - ylo) * rng.random()
        worst = min(worst, eval_net(layers, [x, y])[component])
    assert worst >= f_lower - 1e-9, f"unsound: sample {worst} < f_lower {f_lower}"
    return worst


# --------------------------- fixtures ---------------------------------------

# A tanh MLP with a genuinely loose interval-bound-propagation value enclosure,
# so branch-and-bound must actually refine (unlike an exact-at-min bump).
TANH_MLP: list[Layer] = [
    ([[1.5, -0.7], [0.8, 1.2], [-1.1, 0.5], [0.3, -1.4]], [0.1, -0.2, 0.05, 0.2], "tanh"),
    ([[0.9, -1.3, 0.6, 0.7]], [0.15], None),
]
TANH_BOX = [(-2.0, 2.0), (-2.0, 2.0)]

# u(x, y) = -exp(-x^2/2) - exp(-y^2/2): min -2 at the origin, Hessian ~ I near it.
BUMP_NET: list[Layer] = [
    ([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], "gaussian"),
    ([[-1.0, -1.0]], [0.0], None),
]

# softplus(x) + softplus(-x) + softplus(y) + softplus(-y): a smooth convex well,
# global min 4*ln 2 at the origin (exercises the new verified softplus tower).
SOFTPLUS_WELL: list[Layer] = [
    ([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], [0.0, 0.0, 0.0, 0.0], "softplus"),
    ([[1.0, 1.0, 1.0, 1.0]], [0.0], None),
]

# gelu(x) + gelu(-x) + gelu(y) + gelu(-y): each pair is x*(2 Phi(x) - 1) >= 0, so
# the global min is 0 at the origin (exercises the new exact verified gelu tower).
GELU_WELL: list[Layer] = [
    ([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], [0.0, 0.0, 0.0, 0.0], "gelu"),
    ([[1.0, 1.0, 1.0, 1.0]], [0.0], None),
]


# --------------------------- tests ------------------------------------------


def test_bump_net_exact_global_min() -> None:
    r = certified_network_minimize(BUMP_NET, [(-2.0, 2.0), (-2.0, 2.0)], tol=1e-3)
    assert r.converged
    assert r.f_lower <= -2.0 <= r.f_upper  # enclosure contains the true min
    assert abs(r.x[0]) < 1e-2 and abs(r.x[1]) < 1e-2


def test_softplus_well_certified_global_min() -> None:
    box = [(-2.0, 2.0), (-2.0, 2.0)]
    r = certified_network_minimize(SOFTPLUS_WELL, box, tol=1e-3, second_order=True)
    assert r.converged
    true_min = 4.0 * math.log(2.0)
    assert r.f_lower <= true_min <= r.f_upper  # enclosure contains the analytic min
    assert abs(r.x[0]) < 1e-2 and abs(r.x[1]) < 1e-2
    assert_sound(SOFTPLUS_WELL, box, r.f_lower)


def test_gelu_well_certified_global_min() -> None:
    box = [(-2.0, 2.0), (-2.0, 2.0)]
    r = certified_network_minimize(GELU_WELL, box, tol=1e-3, max_boxes=300_000)
    assert r.converged
    assert r.f_lower <= 0.0 <= r.f_upper  # exact gelu well: min 0 at the origin
    assert abs(r.x[0]) < 1e-2 and abs(r.x[1]) < 1e-2
    assert_sound(GELU_WELL, box, r.f_lower)


def test_tanh_mlp_lower_bound_is_sound() -> None:
    r = certified_network_minimize(TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=300_000)
    assert r.converged
    gmin = grid_min(TANH_MLP, TANH_BOX, 0)
    assert r.f_lower <= gmin  # certified lower bound under-estimates the grid min
    assert r.f_upper <= gmin + 1e-2  # incumbent is at least as good as the grid
    assert_sound(TANH_MLP, TANH_BOX, r.f_lower)


def test_tanh_mlp_branch_and_bound_actually_refines() -> None:
    # the loose interval-bound-propagation value forces real work (not a trivial
    # 1-box certificate); the natural-extension-only path exposes it plainly.
    r = certified_network_minimize(TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=300_000, use_gradient=False)
    assert r.converged
    assert r.boxes_explored > 100


def test_gradient_enclosure_encloses_finite_difference() -> None:
    # the closed-form interval gradient over a small box must contain the true
    # (finite-difference) gradient at every interior point.
    rng = random.Random(1)
    h = 1e-5
    half = 1e-3
    for _ in range(40):
        px = rng.uniform(-1.8, 1.8)
        py = rng.uniform(-1.8, 1.8)
        box = [Interval(px - half, px + half), Interval(py - half, py + half)]
        jet = mlp_jet_mv(box, TANH_MLP, 1)
        g = jet_gradient(jet, 2, 1)  # shape (2, 1)
        gx, gy = g[0][0], g[1][0]
        fd_x = (eval_net(TANH_MLP, [px + h, py])[0] - eval_net(TANH_MLP, [px - h, py])[0]) / (2 * h)
        fd_y = (eval_net(TANH_MLP, [px, py + h])[0] - eval_net(TANH_MLP, [px, py - h])[0]) / (2 * h)
        assert gx.lo - 1e-6 <= fd_x <= gx.hi + 1e-6
        assert gy.lo - 1e-6 <= fd_y <= gy.hi + 1e-6


def test_gradient_accelerates_search() -> None:
    no_grad = certified_network_minimize(TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=400_000, use_gradient=False)
    with_grad = certified_network_minimize(TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=400_000, use_gradient=True)
    assert with_grad.converged
    # both sound, and the exact gradient (mean-value + monotonicity) cuts the work
    assert with_grad.boxes_explored < no_grad.boxes_explored


def test_strict_local_min_pd_near_origin_and_not_in_flat_region() -> None:
    # near the origin the bump net is a convex well (Hessian ~ I) -> certified PD
    assert certify_network_strict_local_min(BUMP_NET, [(-0.3, 0.3), (-0.3, 0.3)])
    # out on the shoulder u_xx = (1 - x^2) e^{-x^2/2} < 0 -> not positive definite
    assert not certify_network_strict_local_min(BUMP_NET, [(1.5, 2.0), (1.5, 2.0)])


def test_accepts_jetmlp_like_object() -> None:
    class FakeJetMLP:
        def _layer_specs(self):  # type: ignore[no-untyped-def]
            return list(TANH_MLP)

    from_net = certified_network_minimize(FakeJetMLP(), TANH_BOX, tol=1e-3, max_boxes=300_000)
    from_layers = certified_network_minimize(TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=300_000)
    assert from_net.f_lower == from_layers.f_lower
    assert from_net.f_upper == from_layers.f_upper


def test_multi_output_component_selection() -> None:
    # two-output affine readout; component 1 = -(-2 bump) path differs from 0
    net: list[Layer] = [
        ([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], "gaussian"),
        ([[-1.0, -1.0], [1.0, 1.0]], [0.0, 0.0], None),  # out0 = -g-g, out1 = g+g
    ]
    r0 = certified_network_minimize(net, [(-2.0, 2.0), (-2.0, 2.0)], tol=1e-3, component=0)
    r1 = certified_network_minimize(net, [(-2.0, 2.0), (-2.0, 2.0)], tol=1e-3, component=1)
    assert r0.f_lower <= -2.0 <= r0.f_upper  # min of -g-g is -2 at origin
    # min of g+g over the box is at the corners (g small); ~2*exp(-2) = 0.27
    assert 0.0 < r1.f_upper < 0.4


def test_second_order_network_min_sound_and_converges() -> None:
    # second_order lifts the jet to order 2 (Hessian) -> second-order bound + Newton
    r = certified_network_minimize(TANH_MLP, TANH_BOX, tol=1e-3, max_boxes=300_000, second_order=True)
    assert r.converged
    gmin = grid_min(TANH_MLP, TANH_BOX, 0)
    assert r.f_lower <= gmin  # still a sound lower bound
    assert_sound(TANH_MLP, TANH_BOX, r.f_lower)


def test_network_critical_points_finds_bump_origin() -> None:
    # u = -exp(-x^2/2) - exp(-y^2/2): the only stationary point is the origin (a min)
    cps = certified_network_critical_points(BUMP_NET, [(-1.5, 1.5), (-1.5, 1.5)], tol=1e-8)
    minima = [c for c in cps if c.kind == "min"]
    assert minima, cps
    assert any(abs(c.point[0]) < 1e-4 and abs(c.point[1]) < 1e-4 for c in minima)
    assert all(c.eig_min > 0.0 for c in minima)  # certified strict min


def test_network_flatness_bump_pd_near_origin() -> None:
    fr = certified_network_flatness(BUMP_NET, [(-0.3, 0.3), (-0.3, 0.3)])
    assert fr.certified_positive_definite  # curvature ~ diag(1, 1) near the origin
    assert 0.0 < fr.sharpness < 1.5


def test_validation() -> None:
    with pytest.raises(ValueError):
        certified_network_minimize(TANH_MLP, TANH_BOX, order=0)
    with pytest.raises(ValueError):
        certified_network_minimize(TANH_MLP, TANH_BOX, component=5)
    with pytest.raises(ValueError):
        certify_network_strict_local_min(BUMP_NET, [(-1.0, 1.0), (-1.0, 1.0)], order=1)
    with pytest.raises(ValueError):
        certified_network_critical_points(BUMP_NET, [(-1.0, 1.0), (-1.0, 1.0)], order=1)


def test_torch_jetmlp_certified_min_end_to_end() -> None:
    # a *trained* omnibias network, certified directly (duck-typed _layer_specs):
    # the certificate must be sound against the real torch forward pass.
    torch = pytest.importorskip("torch")
    from omnibias.torch.architectures.pinn import JetMLP

    torch.manual_seed(0)
    net = JetMLP(in_dim=2, hidden=4, out_dim=1, depth=2, base="tanh").double()
    box = [(-1.5, 1.5), (-1.5, 1.5)]
    r = certified_network_minimize(net, box, tol=1e-2, max_boxes=200_000)
    assert r.converged

    @torch.no_grad()
    def fwd(px: float, py: float) -> float:
        z = torch.tensor([px, py], dtype=torch.float64)
        for w, b, spec in net._layer_specs():
            z = z @ w.T
            if b is not None:
                z = z + b
            if spec is not None:
                z = torch.tanh(z)
        return float(z.squeeze(-1))

    axis = torch.linspace(-1.5, 1.5, 41).tolist()
    gmin = min(fwd(px, py) for px in axis for py in axis)
    assert r.f_lower <= gmin  # certified lower bound is sound vs the torch network
    assert r.f_upper <= gmin + 5e-2  # incumbent as good as the grid
    assert r.f_lower <= r.f_upper
