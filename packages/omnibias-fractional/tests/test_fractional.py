# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fractional-derivative validation.

These are grid-based numerical approximations, so tolerances are looser than the
closed-form ops (documented per assertion). Checks:
1. analytic ``D^alpha x^p = Gamma(p+1)/Gamma(p+1-alpha) x^{p-alpha}`` (Caputo),
2. spectral integer derivatives (alpha=1, 2) recover ordinary derivatives,
3. spectral composition ``D^{alpha}(D^{alpha} f) = D^{2 alpha} f``,
4. torch vs jax cross-backend agreement.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import torch
from omnibias.fractional.jax import ops as jfr
from omnibias.fractional.torch import ops as tfr


def _np(x):  # type: ignore[no-untyped-def]
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def test_caputo_power_law() -> None:
    # Caputo D^alpha x^p = Gamma(p+1)/Gamma(p+1-alpha) x^{p-alpha} for x>0.
    p, alpha = 3.0, 0.5
    n = 4000
    x = np.linspace(0.0, 2.0, n)
    h = float(x[1] - x[0])
    f = torch.as_tensor(x**p, dtype=torch.float64)
    got = _np(tfr.caputo(f, alpha=alpha, h=h))
    coef = math.gamma(p + 1) / math.gamma(p + 1 - alpha)
    exact = coef * x ** (p - alpha)
    # compare on an interior window (GL is O(h); edges are least accurate)
    sl = slice(n // 4, 3 * n // 4)
    rel = np.abs(got[sl] - exact[sl]) / (np.abs(exact[sl]) + 1e-9)
    assert np.max(rel) < 2e-2


def test_spectral_integer_derivatives() -> None:
    n = 256
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, n, endpoint=False)
    f = torch.as_tensor(np.sin(x), dtype=torch.float64)
    d1 = _np(tfr.spectral_fractional(f, alpha=1.0, length=L)).real
    d2 = _np(tfr.spectral_fractional(f, alpha=2.0, length=L)).real
    assert np.allclose(d1, np.cos(x), atol=1e-9)
    assert np.allclose(d2, -np.sin(x), atol=1e-9)


def test_spectral_half_derivative_composes() -> None:
    n = 256
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, n, endpoint=False)
    f = torch.as_tensor(np.sin(3 * x) + 0.5 * np.cos(x), dtype=torch.float64)
    half = tfr.spectral_fractional(f, alpha=0.5, length=L)
    twice = tfr.spectral_fractional(half, alpha=0.5, length=L)
    full = tfr.spectral_fractional(f, alpha=1.0, length=L)
    assert np.allclose(_np(twice), _np(full), atol=1e-9)


def test_gl_weights_recurrence() -> None:
    from omnibias.fractional._core.kernels import gl_weights

    w = gl_weights(0.5, 5)
    # w_0 = 1, w_1 = -alpha, ...
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], -0.5)


def test_fractional_cross_backend() -> None:
    n = 512
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, n, endpoint=False)
    fx = np.sin(x) + 0.3 * np.cos(2 * x)
    tf = torch.as_tensor(fx, dtype=torch.float64)
    jf = jnp.asarray(fx, dtype=jnp.float64)
    # spectral
    assert np.allclose(
        _np(tfr.spectral_fractional(tf, alpha=0.7, length=L)),
        _np(jfr.spectral_fractional(jf, alpha=0.7, length=L)),
        rtol=1e-9, atol=1e-9,
    )
    # GL on a power grid
    xg = np.linspace(0.0, 2.0, 400)
    h = float(xg[1] - xg[0])
    tg = torch.as_tensor(xg**2, dtype=torch.float64)
    jg = jnp.asarray(xg**2, dtype=jnp.float64)
    assert np.allclose(
        _np(tfr.grunwald_letnikov(tg, alpha=0.5, h=h)),
        _np(jfr.grunwald_letnikov(jg, alpha=0.5, h=h)),
        rtol=1e-9, atol=1e-9,
    )


# --------------------------------------------------------------------------- #
# Phase A: differentiable / learnable order (tensor-valued alpha).
# --------------------------------------------------------------------------- #
def test_gl_tensor_alpha_matches_float() -> None:
    # Dispatch equivalence: a tensor alpha reproduces the numpy float path (GL).
    n = 200
    xg = np.linspace(0.0, 2.0, n)
    h = float(xg[1] - xg[0])
    f = torch.as_tensor(xg**2, dtype=torch.float64)
    got_f = _np(tfr.grunwald_letnikov(f, alpha=0.5, h=h))
    got_t = _np(tfr.grunwald_letnikov(f, alpha=torch.tensor(0.5, dtype=torch.float64), h=h))
    assert np.allclose(got_f, got_t, rtol=1e-9, atol=1e-12)


def test_spectral_tensor_alpha_matches_float() -> None:
    n = 256
    length = 2.0 * np.pi
    x = np.linspace(0.0, length, n, endpoint=False)
    f = torch.as_tensor(np.sin(x), dtype=torch.float64)
    got_f = _np(tfr.spectral_fractional(f, alpha=0.7, length=length))
    got_t = _np(
        tfr.spectral_fractional(f, alpha=torch.tensor(0.7, dtype=torch.float64), length=length)
    )
    assert np.allclose(got_f, got_t, rtol=1e-9, atol=1e-12)


def test_gl_alpha_zero_is_identity() -> None:
    # alpha = 0 -> weights [1, 0, ...] -> the operator returns f unchanged.
    n = 64
    xg = np.linspace(0.0, 2.0, n)
    h = float(xg[1] - xg[0])
    f = torch.as_tensor(np.sin(xg) + 0.5, dtype=torch.float64)
    out = tfr.grunwald_letnikov(f, alpha=torch.tensor(0.0, dtype=torch.float64), h=h)
    assert np.allclose(_np(out), _np(f), rtol=1e-12, atol=1e-12)


def test_gl_order_gradient_matches_fd() -> None:
    n = 48
    xg = np.linspace(0.1, 2.0, n)
    h = float(xg[1] - xg[0])
    f = torch.as_tensor(xg**2, dtype=torch.float64)

    a = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    tfr.grunwald_letnikov(f, alpha=a, h=h).sum().backward()
    grad = float(a.grad)  # type: ignore[arg-type]

    def value(av: float) -> float:
        out = tfr.grunwald_letnikov(f, alpha=torch.tensor(av, dtype=torch.float64), h=h)
        return float(out.sum())

    eps = 1e-6
    fd = (value(0.5 + eps) - value(0.5 - eps)) / (2 * eps)
    assert abs(grad - fd) < 1e-4


def test_spectral_order_gradient_matches_fd() -> None:
    n = 64
    length = 2.0 * np.pi
    x = np.linspace(0.0, length, n, endpoint=False)
    f = torch.as_tensor(np.sin(x) + 0.3 * np.cos(2 * x), dtype=torch.float64)

    a = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    out = tfr.spectral_fractional(f, alpha=a, length=length)
    (out.real**2 + out.imag**2).sum().backward()
    grad = float(a.grad)  # type: ignore[arg-type]

    def value(av: float) -> float:
        o = tfr.spectral_fractional(f, alpha=torch.tensor(av, dtype=torch.float64), length=length)
        return float((o.real**2 + o.imag**2).sum())

    eps = 1e-6
    fd = (value(0.7 + eps) - value(0.7 - eps)) / (2 * eps)
    assert abs(grad - fd) < 1e-4


def test_gl_tensor_alpha_cross_backend() -> None:
    n = 64
    xg = np.linspace(0.0, 2.0, n)
    h = float(xg[1] - xg[0])
    tg = torch.as_tensor(xg**2, dtype=torch.float64)
    jg = jnp.asarray(xg**2, dtype=jnp.float64)

    tv = _np(tfr.grunwald_letnikov(tg, alpha=torch.tensor(0.5, dtype=torch.float64), h=h))
    jv = _np(jfr.grunwald_letnikov(jg, alpha=jnp.asarray(0.5), h=h))
    assert np.allclose(tv, jv, rtol=1e-9, atol=1e-9)

    ta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    tfr.grunwald_letnikov(tg, alpha=ta, h=h).sum().backward()

    def jloss(a):  # type: ignore[no-untyped-def]
        return jfr.grunwald_letnikov(jg, alpha=a, h=h).sum()

    jgrad = float(jax.grad(jloss)(jnp.asarray(0.5)))
    assert abs(float(ta.grad) - jgrad) < 1e-6  # type: ignore[arg-type]


def test_spectral_tensor_alpha_cross_backend() -> None:
    n = 64
    length = 2.0 * np.pi
    x = np.linspace(0.0, length, n, endpoint=False)
    fx = np.sin(x) + 0.3 * np.cos(2 * x)
    tf = torch.as_tensor(fx, dtype=torch.float64)
    jf = jnp.asarray(fx, dtype=jnp.float64)

    tv = _np(tfr.spectral_fractional(tf, alpha=torch.tensor(0.6, dtype=torch.float64), length=length))
    jv = _np(jfr.spectral_fractional(jf, alpha=jnp.asarray(0.6), length=length))
    assert np.allclose(tv, jv, rtol=1e-9, atol=1e-9)

    ta = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    out = tfr.spectral_fractional(tf, alpha=ta, length=length)
    (out.real**2 + out.imag**2).sum().backward()

    def jloss(a):  # type: ignore[no-untyped-def]
        o = jfr.spectral_fractional(jf, alpha=a, length=length)
        return (jnp.real(o) ** 2 + jnp.imag(o) ** 2).sum()

    jgrad = float(jax.grad(jloss)(jnp.asarray(0.6)))
    assert abs(float(ta.grad) - jgrad) < 1e-6  # type: ignore[arg-type]


def test_learnable_order_torch_roundtrip_and_grad() -> None:
    from omnibias.fractional.torch.order import LearnableOrder

    mod = LearnableOrder(init=0.5, lo=0.0, hi=2.0)
    a = mod()
    av = float(a.detach())
    assert 0.0 < av < 2.0
    assert abs(av - 0.5) < 1e-6
    a.backward()
    assert mod.raw.grad is not None


def test_learnable_order_jax_matches_torch() -> None:
    from omnibias.fractional.jax.order import constrain_order, init_order
    from omnibias.fractional.torch.order import LearnableOrder

    lo, hi, init = 0.0, 2.0, 0.75
    raw = init_order(init, lo=lo, hi=hi)
    ja = float(constrain_order(raw, lo=lo, hi=hi))
    assert abs(ja - init) < 1e-6  # round-trip through logit / sigmoid
    tmod = LearnableOrder(init=init, lo=lo, hi=hi)
    assert abs(float(tmod().detach()) - ja) < 1e-6  # torch and jax mappings agree
