# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form (jet-based) analytic fractional derivative validation.

Unlike ``test_fractional.py`` (grid / spectral approximations) these check the
*closed-form* operator on the analytic-function class:

    D^alpha f = sum_k a_k Gamma(k+1)/Gamma(k+1-alpha) (x-a)^{k-alpha}.

float64 throughout (jax x64 enabled in ``conftest``). The core tests build
polynomial jets by hand, so they exercise the op with no backend package; the
MLP-convenience tests import ``omnibias-torch`` / ``omnibias-jax``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fractional.jax.ops import analytic as ja
from omnibias.fractional.torch import ops as tfr
from omnibias.fractional.torch.ops import analytic as ta

F = torch.float64


def _poly(x: torch.Tensor, coeffs: list[float]) -> torch.Tensor:
    """Evaluate sum_k coeffs[k] * x**k (coeffs are the Taylor jet about a=0)."""
    return sum(c * x**k for k, c in enumerate(coeffs))


# ----- alpha = 0 recovers the function -----


def test_d0_identity() -> None:
    coeffs = [2.0, 3.0, 4.0, -1.0]
    jet = torch.tensor(coeffs, dtype=F)
    x = torch.linspace(0.1, 2.0, 25, dtype=F)
    got = ta.fractional_derivative(jet, x, alpha=0.0)
    assert torch.allclose(got, _poly(x, coeffs), rtol=1e-12, atol=1e-12)


# ----- single-term jet reproduces the analytic power law -----


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_monomial_power_law(k: int) -> None:
    alpha = 0.6
    jet = torch.zeros(6, dtype=F)
    jet[k] = 1.0
    x = torch.linspace(0.2, 1.7, 20, dtype=F)
    got = ta.fractional_derivative(jet, x, alpha=alpha)
    coef = math.gamma(k + 1) / math.gamma(k + 1 - alpha)
    assert torch.allclose(got, coef * x ** (k - alpha), rtol=1e-12, atol=1e-12)


# ----- integer order recovers the ordinary derivative (t > 0) -----


def test_integer_order_recovers_ordinary_derivative() -> None:
    coeffs = [2.0, 3.0, 4.0, -1.0, 0.5]  # f(t) = 2 + 3t + 4t^2 - t^3 + t^4/2
    jet = torch.tensor(coeffs, dtype=F)
    x = torch.linspace(0.1, 2.0, 30, dtype=F)
    d1 = ta.fractional_derivative(jet, x, alpha=1.0)
    d2 = ta.fractional_derivative(jet, x, alpha=2.0)
    d3 = ta.fractional_derivative(jet, x, alpha=3.0)
    f1 = 3 + 8 * x - 3 * x**2 + 2 * x**3
    f2 = 8 - 6 * x + 6 * x**2
    f3 = -6 + 12 * x
    assert torch.allclose(d1, f1, rtol=1e-11, atol=1e-11)
    assert torch.allclose(d2, f2, rtol=1e-11, atol=1e-11)
    assert torch.allclose(d3, f3, rtol=1e-11, atol=1e-11)


# ----- Caputo vs Riemann-Liouville at the terminal -----


def test_caputo_vs_rl_constant() -> None:
    alpha = 0.5
    c = 5.0
    jet = torch.tensor([c, 0.0, 0.0], dtype=F)
    x = torch.linspace(0.2, 1.5, 15, dtype=F)
    cap = ta.fractional_derivative(jet, x, alpha=alpha, kind="caputo")
    rl = ta.fractional_derivative(jet, x, alpha=alpha, kind="riemann_liouville")
    # Caputo of a constant is exactly 0; RL of a constant is c t^-a / Gamma(1-a).
    assert torch.allclose(cap, torch.zeros_like(x), atol=1e-14)
    assert torch.allclose(rl, c * x ** (-alpha) / math.gamma(1 - alpha), rtol=1e-12)


def test_caputo_drops_terms_below_ceil_alpha() -> None:
    # For 1 < alpha < 2, Caputo drops k = 0, 1 (keeps k >= 2).
    alpha = 1.5
    coeffs = [7.0, -2.0, 3.0, 1.0]
    jet = torch.tensor(coeffs, dtype=F)
    x = torch.linspace(0.3, 1.4, 12, dtype=F)
    cap = ta.fractional_derivative(jet, x, alpha=alpha, kind="caputo")
    # Expected: only k>=2 survive.
    expected = torch.zeros_like(x)
    for k in (2, 3):
        coef = math.gamma(k + 1) / math.gamma(k + 1 - alpha)
        expected = expected + coeffs[k] * coef * x ** (k - alpha)
    assert torch.allclose(cap, expected, rtol=1e-12, atol=1e-12)


# ----- validate the closed form against the established grid path -----


def test_closed_form_matches_grunwald_letnikov() -> None:
    # Polynomial about the grid's left edge; GL is the RL discretisation with
    # lower terminal = left edge, so the closed-form RL should agree (loosely,
    # GL is O(h)).
    alpha = 0.5
    n = 4000
    x0 = 0.3
    xs = np.linspace(x0, x0 + 2.0, n)
    h = float(xs[1] - xs[0])
    t = xs - x0  # offset from the terminal
    jet_coeffs = [1.0, 2.0, 0.5]  # f = 1 + 2 t + 0.5 t^2 about the terminal
    f = torch.as_tensor(_poly(torch.as_tensor(t, dtype=F), jet_coeffs), dtype=F)
    grid = tfr.grunwald_letnikov(f, alpha=alpha, h=h).detach().numpy()

    jet = torch.tensor(jet_coeffs, dtype=F)
    closed = ta.fractional_derivative(
        jet, torch.as_tensor(t, dtype=F), alpha=alpha, a=0.0
    ).numpy()

    sl = slice(n // 4, 3 * n // 4)  # interior window (GL edges are least accurate)
    rel = np.abs(grid[sl] - closed[sl]) / (np.abs(closed[sl]) + 1e-9)
    assert np.max(rel) < 2e-2


# ----- differentiability -----


def test_order_gradient_matches_finite_difference() -> None:
    jet = torch.tensor([1.0, 0.0, 2.0, -0.5], dtype=F)
    x = torch.tensor([0.7, 1.3], dtype=F)
    a0 = 0.65
    alpha = torch.tensor(a0, dtype=F, requires_grad=True)
    ta.fractional_derivative(jet, x, alpha=alpha).sum().backward()
    eps = 1e-6
    fp = float(ta.fractional_derivative(jet, x, alpha=a0 + eps).sum())
    fm = float(ta.fractional_derivative(jet, x, alpha=a0 - eps).sum())
    assert abs(float(alpha.grad) - (fp - fm) / (2 * eps)) < 1e-6


def test_jet_coefficient_gradient() -> None:
    coeffs = [1.0, 0.0, 2.0, -0.5]
    x = torch.tensor([0.7, 1.3], dtype=F)
    alpha = 0.6
    jet = torch.tensor(coeffs, dtype=F, requires_grad=True)
    ta.fractional_derivative(jet, x, alpha=alpha).sum().backward()
    # d/d a_k [sum_x D^alpha f] = sum_x ratio_k * t^(k-alpha).
    k = torch.arange(len(coeffs), dtype=F)
    ratio = ta._gamma_ratio(k, torch.tensor(alpha, dtype=F))
    expected = torch.stack(
        [(ratio[j] * x ** (j - alpha)).sum() for j in range(len(coeffs))]
    )
    assert torch.allclose(jet.grad, expected, rtol=1e-11, atol=1e-11)


# ----- cross-backend parity -----


def test_torch_jax_value_and_order_gradient_parity() -> None:
    coeffs = [2.0, 3.0, 4.0, -1.0]
    x_np = np.linspace(0.2, 1.8, 17)
    a0 = 0.7

    jt = torch.tensor(coeffs, dtype=F)
    xt = torch.as_tensor(x_np, dtype=F)
    jj = jnp.asarray(coeffs, dtype=jnp.float64)
    xj = jnp.asarray(x_np)

    vt = ta.fractional_derivative(jt, xt, alpha=a0).detach().numpy()
    vj = np.asarray(ja.fractional_derivative(jj, xj, alpha=a0))
    assert np.allclose(vt, vj, rtol=1e-9, atol=1e-11)

    at = torch.tensor(a0, dtype=F, requires_grad=True)
    ta.fractional_derivative(jt, xt, alpha=at).sum().backward()
    gj = jax.grad(lambda al: ja.fractional_derivative(jj, xj, alpha=al).sum())(a0)
    assert np.allclose(float(at.grad), float(gj), rtol=1e-6, atol=1e-8)


# ----- MLP-level convenience (needs the backend jet kernels) -----


def _mlp_weights(seed: int = 0, dims: tuple[int, ...] = (1, 4, 1)):
    rng = np.random.default_rng(seed)
    ws, bs = [], []
    for i in range(len(dims) - 1):
        ws.append(rng.normal(scale=0.7, size=(dims[i + 1], dims[i])))
        bs.append(rng.normal(scale=0.5, size=(dims[i + 1],)))
    return ws, bs


def test_mlp_fractional_derivative_torch_jax_parity_and_grad() -> None:
    pytest.importorskip("omnibias.torch.jet")
    pytest.importorskip("omnibias.jax.jet")

    ws, bs = _mlp_weights(seed=3, dims=(1, 5, 1))
    x0_np, v_np = np.array([0.3]), np.array([1.0])
    t_np = np.array([0.4, 0.9, 1.3])
    alpha, order = 0.5, 5

    layers_t = [
        (torch.as_tensor(ws[0], dtype=F), torch.as_tensor(bs[0], dtype=F), "tanh"),
        (torch.as_tensor(ws[1], dtype=F), torch.as_tensor(bs[1], dtype=F), None),
    ]
    out_t = ta.mlp_fractional_derivative(
        torch.as_tensor(x0_np, dtype=F),
        torch.as_tensor(v_np, dtype=F),
        layers_t,
        torch.as_tensor(t_np, dtype=F),
        alpha=alpha,
        order=order,
    )

    layers_j = [
        (jnp.asarray(ws[0]), jnp.asarray(bs[0]), "tanh"),
        (jnp.asarray(ws[1]), jnp.asarray(bs[1]), None),
    ]
    out_j = ja.mlp_fractional_derivative(
        jnp.asarray(x0_np),
        jnp.asarray(v_np),
        layers_j,
        jnp.asarray(t_np),
        alpha=alpha,
        order=order,
    )
    assert np.allclose(out_t.detach().numpy(), np.asarray(out_j), rtol=1e-9, atol=1e-11)

    # the order is learnable end-to-end through the MLP jet
    a_learn = torch.tensor(alpha, dtype=F, requires_grad=True)
    loss = ta.mlp_fractional_derivative(
        torch.as_tensor(x0_np, dtype=F),
        torch.as_tensor(v_np, dtype=F),
        layers_t,
        torch.as_tensor(t_np, dtype=F),
        alpha=a_learn,
        order=order,
    ).pow(2).sum()
    loss.backward()
    assert a_learn.grad is not None and torch.isfinite(a_learn.grad)


# ----- integrates with the Phase A LearnableOrder reparametrisation -----


def test_learnable_order_integration() -> None:
    from omnibias.fractional.torch.order import LearnableOrder

    order = LearnableOrder(init=0.5, lo=0.0, hi=2.0)
    jet = torch.tensor([1.0, 0.0, 2.0, -0.5], dtype=F)
    x = torch.tensor([0.6, 1.1], dtype=F)
    out = ta.fractional_derivative(jet, x, alpha=order())
    out.pow(2).sum().backward()
    assert order.raw.grad is not None and torch.isfinite(order.raw.grad)


# ----- error paths -----


def test_invalid_kind_raises() -> None:
    jet = torch.tensor([1.0, 2.0], dtype=F)
    x = torch.tensor([0.5], dtype=F)
    with pytest.raises(ValueError, match="kind must be"):
        ta.fractional_derivative(jet, x, alpha=0.5, kind="grunwald")


def test_scalar_jet_raises() -> None:
    with pytest.raises(ValueError, match="leading order axis"):
        ta.fractional_derivative(torch.tensor(1.0, dtype=F), torch.tensor([0.5], dtype=F), alpha=0.5)
