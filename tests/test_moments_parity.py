# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity + correctness of analytic moment propagation.

The torch and jax ``gaussian_moment_propagation`` share the multivariate jet and
the ring-generic delta-method core, so they must agree to float64 precision; the
linear case is exact and the nonlinear case is validated against Monte-Carlo.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.moments import gaussian_moment_propagation as jax_prop  # noqa: E402
from omnibias.torch.moments import gaussian_moment_propagation as torch_prop  # noqa: E402

torch.set_default_dtype(torch.float64)


def _rng_net(seed: int):
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((4, 3)) * 0.6
    b1 = rng.standard_normal(4) * 0.3
    w2 = rng.standard_normal((2, 4)) * 0.6
    b2 = rng.standard_normal(2) * 0.3
    mean = rng.standard_normal(3) * 0.4
    a = rng.standard_normal((3, 3)) * 0.3
    cov = a @ a.T + 0.05 * np.eye(3)  # SPD
    return w1, b1, w2, b2, mean, cov


def test_linear_propagation_is_exact() -> None:
    rng = np.random.default_rng(7)
    w = rng.standard_normal((2, 3))
    b = rng.standard_normal(2)
    mean = rng.standard_normal(3)
    a = rng.standard_normal((3, 3))
    cov = a @ a.T + 0.1 * np.eye(3)

    layers_t = [(torch.as_tensor(w), torch.as_tensor(b), None)]
    out_mean, out_cov = torch_prop(layers_t, torch.as_tensor(mean), torch.as_tensor(cov))
    exp_mean = w @ mean + b
    exp_cov = w @ cov @ w.T
    assert np.allclose(out_mean.numpy(), exp_mean, rtol=1e-12, atol=1e-12)
    assert np.allclose(out_cov.numpy(), exp_cov, rtol=1e-12, atol=1e-12)


def test_cross_backend_parity_tanh_mlp() -> None:
    w1, b1, w2, b2, mean, cov = _rng_net(0)
    layers_t = [
        (torch.as_tensor(w1), torch.as_tensor(b1), "tanh"),
        (torch.as_tensor(w2), torch.as_tensor(b2), None),
    ]
    layers_j = [
        (jnp.asarray(w1), jnp.asarray(b1), "tanh"),
        (jnp.asarray(w2), jnp.asarray(b2), None),
    ]
    tm, tc = torch_prop(layers_t, torch.as_tensor(mean), torch.as_tensor(cov))
    jm, jc = jax_prop(layers_j, jnp.asarray(mean), jnp.asarray(cov))
    assert np.allclose(tm.numpy(), np.asarray(jm), rtol=1e-9, atol=1e-12)
    assert np.allclose(tc.numpy(), np.asarray(jc), rtol=1e-9, atol=1e-12)


def test_nonlinear_propagation_matches_monte_carlo() -> None:
    # The second-order delta method is exact up to O(sigma^4); compare to MC in
    # the small-variance regime where that truncation error is negligible.
    w1, b1, w2, b2, mean, cov_big = _rng_net(3)
    cov = 0.02 * cov_big
    layers_t = [
        (torch.as_tensor(w1), torch.as_tensor(b1), "tanh"),
        (torch.as_tensor(w2), torch.as_tensor(b2), None),
    ]
    out_mean, out_cov = torch_prop(layers_t, torch.as_tensor(mean), torch.as_tensor(cov), order=2)

    rng = np.random.default_rng(123)
    samples = rng.multivariate_normal(mean, cov, size=1_000_000)
    h = np.tanh(samples @ w1.T + b1)
    y = h @ w2.T + b2
    mc_mean = y.mean(axis=0)
    mc_cov = np.cov(y, rowvar=False)

    assert np.allclose(out_mean.numpy(), mc_mean, atol=2e-3)
    assert np.allclose(out_cov.numpy(), mc_cov, atol=2e-3)


def test_second_order_correction_improves_on_mean_value() -> None:
    # The Hessian correction must move the mean toward the MC truth vs f(mean).
    w1, b1, w2, b2, mean, cov_big = _rng_net(3)
    cov = 0.3 * cov_big
    layers_t = [
        (torch.as_tensor(w1), torch.as_tensor(b1), "tanh"),
        (torch.as_tensor(w2), torch.as_tensor(b2), None),
    ]
    m2, _ = torch_prop(layers_t, torch.as_tensor(mean), torch.as_tensor(cov), order=2)
    m1, _ = torch_prop(layers_t, torch.as_tensor(mean), torch.as_tensor(cov), order=1)

    rng = np.random.default_rng(9)
    samples = rng.multivariate_normal(mean, cov, size=1_000_000)
    y = np.tanh(samples @ w1.T + b1) @ w2.T + b2
    mc_mean = y.mean(axis=0)

    err1 = np.abs(m1.numpy() - mc_mean).sum()
    err2 = np.abs(m2.numpy() - mc_mean).sum()
    assert err2 < err1


def test_delta_method_gaussian_elementwise_matches_mc() -> None:
    from omnibias.torch.moments import delta_method_gaussian

    # f(x) = x^3 (exact cubic): derivative tower at mu, Gaussian input.
    mu = torch.tensor([0.5, -1.0, 2.0])
    var = 0.09
    derivs = [mu**3, 3 * mu**2, 6 * mu, torch.full_like(mu, 6.0), torch.zeros_like(mu)]
    out = delta_method_gaussian(derivs, var, order=4)

    rng = np.random.default_rng(5)
    x = rng.normal(mu.numpy()[:, None], np.sqrt(var), size=(3, 400_000))
    y = x**3
    assert np.allclose(out["mean"].numpy(), y.mean(axis=1), atol=5e-3)
    assert np.allclose(out["variance"].numpy(), y.var(axis=1), atol=2e-2)
