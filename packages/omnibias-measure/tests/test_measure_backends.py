# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity and autograd tests for the measure primitives."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

from omnibias.measure._core import integrate as core_int  # noqa: E402
from omnibias.measure._core import measure as mc  # noqa: E402
from omnibias.measure.jax import layers as jlayers  # noqa: E402
from omnibias.measure.jax import ops as jops  # noqa: E402
from omnibias.measure.torch import layers as tlayers  # noqa: E402
from omnibias.measure.torch import ops as tops  # noqa: E402


def _np_f(p: np.ndarray) -> np.ndarray:
    return (p**2).sum(axis=1)


def _t_f(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=1)


def _j_f(x: jax.Array) -> jax.Array:
    return (x**2).sum(axis=1)


def test_lebesgue_integral_parity() -> None:
    mu = mc.lebesgue([(0.0, 1.0), (-1.0, 1.0)], 10)
    core = float(core_int.lebesgue_integral(_np_f, mu))
    t = float(tops.lebesgue_integral(_t_f, mu))
    j = float(jops.lebesgue_integral(_j_f, mu))
    assert t == pytest.approx(core, rel=1e-11)
    assert j == pytest.approx(core, rel=1e-11)


def test_layer_cake_parity() -> None:
    mu = mc.lebesgue([(-1.0, 1.0)], 48)
    kw = dict(beta=80.0, num_t=400, signed=True, t_max=2.0)
    core = float(core_int.layer_cake_integral(lambda p: p[:, 0] ** 3, mu, **kw))
    t = float(tops.layer_cake_integral(lambda x: x[:, 0] ** 3, mu, **kw))
    j = float(jops.layer_cake_integral(lambda x: x[:, 0] ** 3, mu, **kw))
    assert t == pytest.approx(core, rel=1e-9, abs=1e-9)
    assert j == pytest.approx(core, rel=1e-9, abs=1e-9)


def test_simple_function_parity() -> None:
    mu = mc.lebesgue([(0.0, 1.0)], 40)
    levels_np = np.linspace(0.0, 1.0, 64)
    core = float(core_int.simple_function_approx(lambda p: p[:, 0], mu, levels=levels_np, beta=120.0).integral)
    t = float(
        tops.simple_function_approx(
            lambda x: x[:, 0], mu, levels=torch.as_tensor(levels_np), beta=120.0
        ).integral
    )
    j = float(
        jops.simple_function_approx(
            lambda x: x[:, 0], mu, levels=jnp.asarray(levels_np), beta=120.0
        ).integral
    )
    assert t == pytest.approx(core, rel=1e-9)
    assert j == pytest.approx(core, rel=1e-9)


def test_importance_expectation_parity() -> None:
    q = mc.gaussian(48)

    def np_lr(p: np.ndarray) -> np.ndarray:
        return -0.5 * (p[:, 0] - 0.5) ** 2 + 0.5 * p[:, 0] ** 2

    core = float(core_int.importance_expectation(lambda p: p[:, 0], q, np_lr))
    t = float(
        tops.importance_expectation(
            lambda x: x[:, 0], q, lambda x: -0.5 * (x[:, 0] - 0.5) ** 2 + 0.5 * x[:, 0] ** 2
        )
    )
    j = float(
        jops.importance_expectation(
            lambda x: x[:, 0], q, lambda x: -0.5 * (x[:, 0] - 0.5) ** 2 + 0.5 * x[:, 0] ** 2
        )
    )
    assert t == pytest.approx(core, rel=1e-10)
    assert j == pytest.approx(core, rel=1e-10)


def test_gradient_through_integrand_torch() -> None:
    # d/dtheta int theta * x^2 dmu = int x^2 dmu
    mu = mc.lebesgue([(0.0, 1.0)], 16)
    theta = torch.tensor(2.0, requires_grad=True)
    out = tops.lebesgue_integral(lambda x: theta * (x[:, 0] ** 2), mu)
    out.backward()
    expected = float(core_int.lebesgue_integral(lambda p: p[:, 0] ** 2, mu))
    assert float(theta.grad) == pytest.approx(expected, rel=1e-10)


def test_gradient_through_measure_weights_torch() -> None:
    # d/dw_i sum_j w_j f_j = f_i
    mu = mc.lebesgue([(0.0, 1.0)], 8)
    w = torch.as_tensor(mu.weights).clone().requires_grad_(True)
    nodes = torch.as_tensor(mu.nodes)
    out = tops.lebesgue_integral(lambda x: x[:, 0] ** 2, nodes=nodes, weights=w)
    out.backward()
    assert torch.allclose(w.grad, nodes[:, 0] ** 2)


def test_gradient_through_measure_weights_jax() -> None:
    mu = mc.lebesgue([(0.0, 1.0)], 8)
    nodes = jnp.asarray(mu.nodes)
    w = jnp.asarray(mu.weights)

    def loss(weights: jax.Array) -> jax.Array:
        return jops.lebesgue_integral(lambda x: x[:, 0] ** 2, nodes=nodes, weights=weights)

    g = jax.grad(loss)(w)
    assert np.allclose(np.asarray(g), np.asarray(nodes[:, 0] ** 2))


def test_layer_cake_matches_direct_on_smooth() -> None:
    # differentiable layer-cake agrees with direct quadrature on a smooth positive f
    mu = mc.lebesgue([(0.0, 1.0)], 64)
    direct = float(tops.lebesgue_integral(lambda x: x[:, 0] ** 2 + 0.5, mu))
    cake = float(
        tops.layer_cake_integral(
            lambda x: x[:, 0] ** 2 + 0.5, mu, beta=400.0, num_t=1500, signed=False
        )
    )
    assert cake == pytest.approx(direct, rel=2e-2)


def test_torch_layers_forward_and_train_step() -> None:
    mu = mc.lebesgue([(-1.0, 1.0)], 40)
    leb = tlayers.LebesgueIntegral(mu)
    val = leb(_t_f)
    assert float(val) == pytest.approx(float(tops.lebesgue_integral(_t_f, mu)), rel=1e-12)

    cake = tlayers.LayerCakeIntegral(mu, beta=50.0, num_t=200, learnable_beta=True)
    opt = torch.optim.SGD(cake.parameters(), lr=1e-2)
    target = torch.tensor(0.4)
    beta0 = float(cake.beta.detach())
    for _ in range(5):
        opt.zero_grad()
        out = cake(lambda x: x[:, 0] ** 2, t_max=1.5)
        loss = (out - target) ** 2
        loss.backward()
        assert cake.log_beta.grad is not None
        opt.step()
    assert float(cake.beta.detach()) != beta0  # beta was updated by training


def test_jax_layer_is_grad_friendly_pytree() -> None:
    mu = mc.lebesgue([(-1.0, 1.0)], 40)
    layer = jlayers.LayerCakeIntegral.from_measure(mu, beta=50.0, num_t=200)

    def loss(lyr: jlayers.LayerCakeIntegral) -> jax.Array:
        return (lyr(lambda x: x[:, 0] ** 2, t_max=1.5) - 0.4) ** 2

    grads = jax.grad(loss)(layer)
    # gradient w.r.t the learnable softness leaf exists and is finite
    assert np.isfinite(float(grads.log_beta))
    assert np.any(np.asarray(grads.weight) != 0.0)


def test_lebesgue_integral_exact_on_simple_function() -> None:
    # The Lebesgue integral of a simple function s = sum_k c_k * 1[A_k] is exact
    # (it is the weight contraction) -- bit-identical across all three backends.
    mu = mc.counting([[0.0], [0.0], [1.0], [1.0], [1.0]])  # 2 atoms at 0, 3 atoms at 1
    exact = float(core_int.lebesgue_integral(lambda p: p[:, 0], mu))  # 0*2 + 1*3
    assert exact == pytest.approx(3.0)
    assert float(tops.lebesgue_integral(lambda x: x[:, 0], mu)) == pytest.approx(3.0)
    assert float(jops.lebesgue_integral(lambda x: x[:, 0], mu)) == pytest.approx(3.0)


def test_simple_function_approx_converges_backend() -> None:
    # Soft from-below simple-function approx converges to int f dmu on a smooth,
    # non-negative f with fine levels avoiding exact ties.
    mu = mc.lebesgue([(0.0, 1.0)], 64)
    levels = torch.linspace(0.0, 1.0, 401)[:-1] + 0.5 / 400  # midpoints, no ties at f=x
    res = tops.simple_function_approx(lambda x: x[:, 0], mu, levels=levels, beta=800.0)
    assert float(res.integral) == pytest.approx(0.5, abs=1e-2)
