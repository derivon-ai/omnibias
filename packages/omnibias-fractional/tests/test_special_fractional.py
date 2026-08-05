# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Special functions of fractional calculus + activation-specific closed forms.

Two layers are exercised:

* the differentiable truncated-series special functions
  (:func:`mittag_leffler` / :func:`polylog` / :func:`lerch` /
  :func:`lower_incomplete_gamma`) against their classical closed-form special
  values, and

* the activation-fractional registry (``exp`` via Mittag-Leffler, ``sigmoid`` via
  the logistic ``e^{-x}`` expansion) against independent closed forms -- the
  jet-based analytic operator for ``exp`` and the integer-order limits
  (``alpha in {0, 1}``) for ``sigmoid``.

float64 throughout (jax x64 enabled in ``conftest``); torch<->jax parity and
autograd in the order ``alpha`` are checked directly.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fractional.jax.ops import activation as ja
from omnibias.fractional.jax.ops import special as js
from omnibias.fractional.torch.ops import activation as ta
from omnibias.fractional.torch.ops import special as ts

F = torch.float64


# ============================ special functions =============================


def test_mittag_leffler_reduces_to_exp() -> None:
    z = torch.linspace(-1.5, 1.5, 21, dtype=F)
    got = ts.mittag_leffler(z, 1.0, 1.0, terms=64)
    assert torch.allclose(got, torch.exp(z), rtol=1e-12, atol=1e-12)


def test_mittag_leffler_reduces_to_cosh() -> None:
    # E_{2,1}(w^2) = cosh(w).
    w = torch.linspace(0.0, 2.0, 21, dtype=F)
    got = ts.mittag_leffler(w**2, 2.0, 1.0, terms=64)
    assert torch.allclose(got, torch.cosh(w), rtol=1e-12, atol=1e-12)


def test_mittag_leffler_e_1_2_closed_form() -> None:
    # E_{1,2}(z) = (e^z - 1) / z.
    z = torch.linspace(0.1, 2.0, 20, dtype=F)
    got = ts.mittag_leffler(z, 1.0, 2.0, terms=64)
    assert torch.allclose(got, (torch.exp(z) - 1.0) / z, rtol=1e-12, atol=1e-12)


def test_polylog_order_one_is_neg_log() -> None:
    z = torch.linspace(-0.7, 0.7, 21, dtype=F)
    got = ts.polylog(1.0, z, terms=256)
    assert torch.allclose(got, -torch.log1p(-z), rtol=1e-10, atol=1e-11)


def test_polylog_dilog_at_half() -> None:
    # Li_2(1/2) = pi^2/12 - (ln 2)^2 / 2.
    got = float(ts.polylog(2.0, torch.tensor(0.5, dtype=F), terms=200))
    exact = math.pi**2 / 12.0 - math.log(2.0) ** 2 / 2.0
    assert abs(got - exact) < 1e-12


def test_lerch_reduces_to_neg_log_over_z() -> None:
    # Phi(z, 1, 1) = -log(1 - z) / z.
    z = torch.linspace(0.1, 0.7, 13, dtype=F)
    got = ts.lerch(z, 1.0, 1.0, terms=256)
    assert torch.allclose(got, -torch.log1p(-z) / z, rtol=1e-10, atol=1e-11)


def test_lerch_matches_polylog_scaling() -> None:
    # Phi(z, s, 1) = Li_s(z) / z.
    z = torch.tensor(0.5, dtype=F)
    lhs = ts.lerch(z, 2.0, 1.0, terms=200) * z
    rhs = ts.polylog(2.0, z, terms=200)
    assert torch.allclose(lhs, rhs, rtol=1e-12, atol=1e-13)


def test_lower_incomplete_gamma_s1_and_s2() -> None:
    x = torch.linspace(0.1, 2.0, 20, dtype=F)
    g1 = ts.lower_incomplete_gamma(1.0, x, terms=80)
    g2 = ts.lower_incomplete_gamma(2.0, x, terms=80)
    assert torch.allclose(g1, 1.0 - torch.exp(-x), rtol=1e-10, atol=1e-11)
    assert torch.allclose(g2, 1.0 - (1.0 + x) * torch.exp(-x), rtol=1e-10, atol=1e-11)


def test_recip_gamma_matches_math_gamma_including_negative() -> None:
    ys = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.0, 2.0, 3.5], dtype=F)
    got = ts._recip_gamma(ys)
    exp = torch.tensor([1.0 / math.gamma(float(y)) for y in ys], dtype=F)
    assert torch.allclose(got, exp, rtol=1e-11, atol=1e-12)


def test_recip_gamma_zero_at_nonpositive_integers() -> None:
    ys = torch.tensor([0.0, -1.0, -2.0, -3.0], dtype=F)
    got = ts._recip_gamma(ys)
    assert torch.allclose(got, torch.zeros_like(ys), atol=1e-300)


# ----- special-function differentiability + parity -----


def test_mittag_leffler_grad_in_z_matches_finite_difference() -> None:
    z0 = torch.tensor([0.3, 0.7, 1.1], dtype=F)
    z = z0.clone().requires_grad_(True)
    ts.mittag_leffler(z, 0.7, 1.0, terms=96).sum().backward()
    eps = 1e-6
    fp = ts.mittag_leffler(z0 + eps, 0.7, 1.0, terms=96)
    fm = ts.mittag_leffler(z0 - eps, 0.7, 1.0, terms=96)
    assert torch.allclose(z.grad, (fp - fm) / (2 * eps), rtol=1e-6, atol=1e-7)


def test_mittag_leffler_grad_in_z_is_exp_at_alpha_one() -> None:
    # E_{1,1}(z) = e^z, so d/dz E_{1,1}(z) = e^z.
    z = torch.tensor([0.3, 0.7, 1.1], dtype=F, requires_grad=True)
    ts.mittag_leffler(z, 1.0, 1.0, terms=96).sum().backward()
    assert torch.allclose(z.grad, torch.exp(z.detach()), rtol=1e-9, atol=1e-10)


def test_special_functions_torch_jax_parity() -> None:
    z = np.linspace(-0.6, 0.6, 11)
    x = np.linspace(0.2, 1.8, 11)
    ml_t = ts.mittag_leffler(torch.as_tensor(z, dtype=F), 0.8, 1.2, terms=96).numpy()
    ml_j = np.asarray(js.mittag_leffler(jnp.asarray(z), 0.8, 1.2, terms=96))
    pl_t = ts.polylog(2.0, torch.as_tensor(z, dtype=F), terms=128).numpy()
    pl_j = np.asarray(js.polylog(2.0, jnp.asarray(z), terms=128))
    lz_t = ts.lerch(torch.as_tensor(z, dtype=F), 1.5, 1.0, terms=128).numpy()
    lz_j = np.asarray(js.lerch(jnp.asarray(z), 1.5, 1.0, terms=128))
    ig_t = ts.lower_incomplete_gamma(1.5, torch.as_tensor(x, dtype=F), terms=96).numpy()
    ig_j = np.asarray(js.lower_incomplete_gamma(1.5, jnp.asarray(x), terms=96))
    assert np.allclose(ml_t, ml_j, rtol=1e-10, atol=1e-12)
    assert np.allclose(pl_t, pl_j, rtol=1e-10, atol=1e-12)
    assert np.allclose(lz_t, lz_j, rtol=1e-10, atol=1e-12)
    assert np.allclose(ig_t, ig_j, rtol=1e-10, atol=1e-12)


def test_special_function_bad_terms_raises() -> None:
    for fn in (ts.mittag_leffler, js.mittag_leffler):
        with pytest.raises(ValueError, match="terms must be"):
            fn(0.5, 1.0, 1.0, terms=0)


# ======================== activation fractional forms =======================


def _exp_jet(lam: float, order: int) -> torch.Tensor:
    return torch.tensor([lam**k / math.factorial(k) for k in range(order + 1)], dtype=F)


def test_exp_fractional_matches_jet_operator_rl() -> None:
    # The Mittag-Leffler closed form must equal the jet-based analytic operator on
    # the exponential's Taylor jet (two independent closed forms).
    from omnibias.fractional.torch.ops import analytic as tan

    alpha, lam = 0.6, 1.0
    x = torch.linspace(0.2, 1.5, 25, dtype=F)
    via_ml = ta.exp_fractional(x, alpha=alpha, lam=lam, terms=96)
    via_jet = tan.fractional_derivative(_exp_jet(lam, 30), x, alpha=alpha, a=0.0)
    assert torch.allclose(via_ml, via_jet, rtol=1e-8, atol=1e-9)


def test_exp_fractional_caputo_matches_jet_operator() -> None:
    from omnibias.fractional.torch.ops import analytic as tan

    alpha, lam = 0.7, 1.3
    x = torch.linspace(0.2, 1.4, 20, dtype=F)
    via_ml = ta.exp_fractional(x, alpha=alpha, lam=lam, kind="caputo", terms=96)
    via_jet = tan.fractional_derivative(
        _exp_jet(lam, 30), x, alpha=alpha, kind="caputo", a=0.0
    )
    assert torch.allclose(via_ml, via_jet, rtol=1e-8, atol=1e-9)


def test_exp_fractional_integer_orders() -> None:
    x = torch.linspace(0.3, 1.6, 20, dtype=F)
    # RL and Caputo of order 1 both reduce to the ordinary derivative e^x.
    assert torch.allclose(ta.exp_fractional(x, alpha=1.0), torch.exp(x), rtol=1e-9, atol=1e-9)
    assert torch.allclose(
        ta.exp_fractional(x, alpha=2.0, kind="caputo"), torch.exp(x), rtol=1e-9, atol=1e-9
    )


def test_exp_fractional_caputo_is_rl_minus_head() -> None:
    alpha, lam = 0.4, 1.0
    x = torch.linspace(0.3, 1.5, 15, dtype=F)
    rl = ta.exp_fractional(x, alpha=alpha, lam=lam)
    cap = ta.exp_fractional(x, alpha=alpha, lam=lam, kind="caputo")
    head = x ** (-alpha) / math.gamma(1.0 - alpha)  # k=0 term only for 0<alpha<1
    assert torch.allclose(cap, rl - head, rtol=1e-10, atol=1e-11)


def test_cosh_sinh_fractional_alpha_zero_recovers_function() -> None:
    x = torch.linspace(0.3, 1.6, 20, dtype=F)
    c0 = ta.cosh_fractional(x, alpha=0.0)
    s0 = ta.sinh_fractional(x, alpha=0.0)
    assert torch.allclose(c0, torch.cosh(x), rtol=1e-10, atol=1e-11)
    assert torch.allclose(s0, torch.sinh(x), rtol=1e-10, atol=1e-11)


def test_cosh_sinh_fractional_integer_order_derivatives() -> None:
    x = torch.linspace(0.3, 1.6, 20, dtype=F)
    # d/dx cosh = sinh; d/dx sinh = cosh (RL order 1 = ordinary derivative for t>0).
    assert torch.allclose(ta.cosh_fractional(x, alpha=1.0), torch.sinh(x), rtol=1e-9, atol=1e-9)
    assert torch.allclose(ta.sinh_fractional(x, alpha=1.0), torch.cosh(x), rtol=1e-9, atol=1e-9)
    # Second derivative: cosh'' = cosh, sinh'' = sinh.
    assert torch.allclose(
        ta.cosh_fractional(x, alpha=2.0, kind="caputo"), torch.cosh(x), rtol=1e-8, atol=1e-8
    )


def test_cosh_fractional_half_order_matches_jet_operator() -> None:
    from omnibias.fractional.torch.ops import analytic as tan

    alpha = 0.5
    x = torch.linspace(0.3, 1.4, 18, dtype=F)
    cosh_jet = torch.tensor(
        [1.0 / math.factorial(k) if k % 2 == 0 else 0.0 for k in range(31)], dtype=F
    )
    via_special = ta.cosh_fractional(x, alpha=alpha, terms=96)
    via_jet = tan.fractional_derivative(cosh_jet, x, alpha=alpha, a=0.0)
    assert torch.allclose(via_special, via_jet, rtol=1e-8, atol=1e-9)


def test_activation_dispatch_and_registry() -> None:
    x = torch.linspace(0.3, 1.5, 10, dtype=F)
    direct = ta.exp_fractional(x, alpha=0.5)
    routed = ta.activation_fractional_derivative("exp", x, alpha=0.5)
    assert torch.allclose(direct, routed, rtol=1e-12, atol=1e-13)
    assert set(ta.ACTIVATION_FRACTIONAL) == {"exp", "cosh", "sinh"}
    croute = ta.activation_fractional_derivative("cosh", x, alpha=0.5)
    assert torch.allclose(croute, ta.cosh_fractional(x, alpha=0.5), rtol=1e-12, atol=1e-13)


def test_activation_dispatch_unknown_raises() -> None:
    with pytest.raises(KeyError, match="no closed-form fractional derivative"):
        ta.activation_fractional_derivative("relu", torch.tensor([0.5], dtype=F), alpha=0.5)


def test_exp_fractional_invalid_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        ta.exp_fractional(torch.tensor([0.5], dtype=F), alpha=0.5, kind="grunwald")


# ----- activation-fractional differentiability + parity -----


def test_exp_fractional_order_gradient_matches_finite_difference() -> None:
    x = torch.tensor([0.6, 1.1, 1.4], dtype=F)
    a0 = 0.55
    alpha = torch.tensor(a0, dtype=F, requires_grad=True)
    ta.exp_fractional(x, alpha=alpha).sum().backward()
    eps = 1e-6
    fp = float(ta.exp_fractional(x, alpha=a0 + eps).sum())
    fm = float(ta.exp_fractional(x, alpha=a0 - eps).sum())
    assert abs(float(alpha.grad) - (fp - fm) / (2 * eps)) < 1e-6


def test_exp_fractional_torch_jax_value_and_grad_parity() -> None:
    x_np = np.linspace(0.2, 1.6, 15)
    a0 = 0.65
    vt = ta.exp_fractional(torch.as_tensor(x_np, dtype=F), alpha=a0).numpy()
    vj = np.asarray(ja.exp_fractional(jnp.asarray(x_np), alpha=a0))
    assert np.allclose(vt, vj, rtol=1e-9, atol=1e-11)

    at = torch.tensor(a0, dtype=F, requires_grad=True)
    ta.exp_fractional(torch.as_tensor(x_np, dtype=F), alpha=at).sum().backward()
    gj = jax.grad(lambda al: ja.exp_fractional(jnp.asarray(x_np), alpha=al).sum())(a0)
    assert np.allclose(float(at.grad), float(gj), rtol=1e-6, atol=1e-8)


def test_cosh_sinh_fractional_torch_jax_parity() -> None:
    x_np = np.linspace(0.3, 1.6, 15)
    ct = ta.cosh_fractional(torch.as_tensor(x_np, dtype=F), alpha=0.5).numpy()
    cj = np.asarray(ja.cosh_fractional(jnp.asarray(x_np), alpha=0.5))
    st = ta.sinh_fractional(torch.as_tensor(x_np, dtype=F), alpha=0.7).numpy()
    sj = np.asarray(ja.sinh_fractional(jnp.asarray(x_np), alpha=0.7))
    assert np.allclose(ct, cj, rtol=1e-9, atol=1e-11)
    assert np.allclose(st, sj, rtol=1e-9, atol=1e-11)
