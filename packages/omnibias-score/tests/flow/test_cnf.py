# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CNF operators: exact trace-of-Jacobian and augmented integration.

Linear velocity fields have closed-form divergence and log-density change.
Cross-backend parity is checked. All float64.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch")
from omnibias.score.flow.torch import ops as tops

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from omnibias.score.flow.jax import ops as jops
except ModuleNotFoundError:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    jops = None  # type: ignore[assignment]

DTYPE = np.float64


def _np(v):  # type: ignore[no-untyped-def]
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _linear_velocity_factory(a_matrix):  # type: ignore[no-untyped-def]
    def velocity(t, x):  # type: ignore[no-untyped-def]
        if isinstance(x, torch.Tensor):
            return x @ a_matrix.T
        return x @ a_matrix.T

    return velocity


def _scaling_velocity(c, d):  # type: ignore[no-untyped-def]
    def velocity(t, x):  # type: ignore[no-untyped-def]
        return c * x

    return velocity


@pytest.fixture
def linear_field():  # type: ignore[no-untyped-def]
    a = np.array([[0.3, -0.1], [0.2, -0.4]], dtype=DTYPE)
    return a, _linear_velocity_factory(torch.as_tensor(a, dtype=torch.float64))


def test_linear_trace_exact(linear_field) -> None:
    a, vel = linear_field
    x = torch.tensor([[1.0, -0.5], [0.2, 0.7]], dtype=torch.float64)
    tr = _np(tops.exact_trace_jacobian(vel, 0.0, x))
    expected = np.trace(a)
    assert np.allclose(tr, expected, atol=1e-10)


def test_scaling_closed_form_likelihood() -> None:
    c = -0.7
    d = 3
    t_end = 1.25
    vel = _scaling_velocity(c, d)
    x0 = torch.randn(5, d, dtype=torch.float64)
    x1, delta = tops.integrate_cnf(vel, x0, 0.0, t_end, steps=80)
    expected_x1 = x0 * np.exp(c * t_end)
    expected_delta = -d * c * t_end
    assert np.allclose(_np(x1), _np(expected_x1), rtol=1e-5, atol=1e-7)
    assert np.allclose(_np(delta), expected_delta, rtol=1e-5, atol=1e-6)


def test_linear_log_density_change(linear_field) -> None:
    a, vel = linear_field
    t_end = 1.0
    x0 = torch.randn(4, 2, dtype=torch.float64)
    _, delta = tops.integrate_cnf(vel, x0, 0.0, t_end, steps=100)
    expected = -np.trace(a) * t_end
    assert np.allclose(_np(delta), expected, rtol=1e-5, atol=1e-7)


def test_brute_force_jacobian_trace_torch(linear_field) -> None:
    a, vel = linear_field
    x = torch.tensor([[0.5, -1.0], [-0.3, 0.8]], dtype=torch.float64)
    traces = []
    for i in range(x.shape[0]):
        xi = x[i : i + 1].detach().requires_grad_(True)
        ji = torch.autograd.functional.jacobian(lambda z: vel(0.0, z), xi)
        # ji shape (1, d, 1, d): trace of dv/dx is ji[0, :, 0, :]
        traces.append(torch.trace(ji[0, :, 0, :]))
    brute = torch.stack(traces).numpy()
    exact = _np(tops.exact_trace_jacobian(vel, 0.0, x))
    assert np.allclose(exact, brute, rtol=1e-7, atol=1e-9)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_brute_force_jacobian_trace_jax(linear_field) -> None:
    a, _ = linear_field
    vel = _linear_velocity_factory(jnp.asarray(a, dtype=jnp.float64))

    def f(x_row):  # type: ignore[no-untyped-def]
        return vel(0.0, x_row[None, :])[0]

    x = jnp.array([[0.5, -1.0], [-0.3, 0.8]], dtype=jnp.float64)
    jac = jax.vmap(jax.jacfwd(f))(x)
    brute = jnp.trace(jac, axis1=1, axis2=2)
    exact = jops.exact_trace_jacobian(vel, 0.0, x)
    assert np.allclose(np.asarray(exact), np.asarray(brute), rtol=1e-7, atol=1e-9)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_cross_backend_trace_and_integrate(linear_field) -> None:
    a, tvel = linear_field
    jvel = _linear_velocity_factory(jnp.asarray(a, dtype=jnp.float64))
    x_np = np.array([[1.0, 0.5], [-0.2, 0.3]], dtype=DTYPE)
    xt = torch.as_tensor(x_np, dtype=torch.float64)
    xj = jnp.asarray(x_np, dtype=jnp.float64)

    tt = _np(tops.exact_trace_jacobian(tvel, 0.0, xt))
    tj = _np(jops.exact_trace_jacobian(jvel, 0.0, xj))
    assert np.allclose(tt, tj, rtol=1e-7, atol=1e-9)

    t_end = 0.75
    x1t, deltat = tops.integrate_cnf(tvel, xt, 0.0, t_end, steps=60)
    x1j, deltaj = jops.integrate_cnf(jvel, xj, 0.0, t_end, steps=60)
    assert np.allclose(_np(x1t), _np(x1j), rtol=1e-7, atol=1e-9)
    assert np.allclose(_np(deltat), _np(deltaj), rtol=1e-7, atol=1e-9)


def test_integrate_with_log_p0(linear_field) -> None:
    _, vel = linear_field
    x0 = torch.randn(3, 2, dtype=torch.float64)
    lp0 = torch.zeros(3, dtype=torch.float64)
    x1, delta, lp1 = tops.integrate_cnf(vel, x0, 0.0, 1.0, steps=40, log_p0=lp0)
    assert np.allclose(_np(lp1), _np(lp0 + delta))


def test_log_prob_roundtrip_scaling() -> None:
    c = 0.4
    d = 2
    vel = _scaling_velocity(c, d)

    def base_lp(x):  # type: ignore[no-untyped-def]
        return -0.5 * torch.sum(x * x, dim=-1)

    x0 = torch.randn(4, d, dtype=torch.float64)
    x1, delta_fwd = tops.integrate_cnf(vel, x0, 0.0, 1.0, steps=80)
    lp1_direct = base_lp(x0) + delta_fwd
    lp1_back = tops.log_prob(vel, x1, 1.0, 0.0, base_lp, steps=80)
    assert np.allclose(_np(lp1_direct), _np(lp1_back), rtol=1e-5, atol=1e-6)


# --- Hutchinson trace estimator (stochastic FFJORD baseline) -----------------


def _nonlinear_velocity(w, b):  # type: ignore[no-untyped-def]
    def velocity(t, x):  # type: ignore[no-untyped-def]
        return torch.tanh(x @ w.T + b)

    return velocity


def _rademacher_basis_d2():  # type: ignore[no-untyped-def]
    return torch.tensor(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=torch.float64
    )


def test_hutchinson_exact_in_expectation_d2() -> None:
    # Averaging eps^T J eps over ALL 2^d Rademacher sign patterns equals tr(J) *exactly*
    # (the empirical mean over the full Rademacher law is its expectation), so the
    # single-probe estimator is provably unbiased -- machine-precision, not statistical.
    w = torch.tensor([[0.5, -0.3], [0.2, 0.7]], dtype=torch.float64)
    b = torch.tensor([0.1, -0.2], dtype=torch.float64)
    vel = _nonlinear_velocity(w, b)
    x = torch.tensor([[0.3, -0.4], [1.0, 0.5], [-0.7, 0.2]], dtype=torch.float64)
    exact = tops.exact_trace_jacobian(vel, 0.0, x)
    ests = [tops.hutchinson_trace_jacobian(vel, 0.0, x, e.expand_as(x)) for e in _rademacher_basis_d2()]
    mean_est = torch.stack(ests).mean(dim=0)
    assert np.allclose(_np(mean_est), _np(exact), atol=1e-10)


def test_hutchinson_single_probe_has_variance() -> None:
    # motivates the study: the single-probe estimator is noisy (variance sum_{i!=j} J_ij^2),
    # whereas exact_trace_jacobian is exact (zero variance).
    w = torch.tensor([[0.4, 0.9], [-0.8, 0.3]], dtype=torch.float64)  # large off-diagonals
    b = torch.zeros(2, dtype=torch.float64)
    vel = _nonlinear_velocity(w, b)
    x = torch.tensor([[0.2, 0.1]], dtype=torch.float64)
    ests = [
        float(tops.hutchinson_trace_jacobian(vel, 0.0, x, e.expand_as(x)).detach()[0])
        for e in _rademacher_basis_d2()
    ]
    assert np.std(ests) > 1e-3  # genuinely noisy
    exact = float(tops.exact_trace_jacobian(vel, 0.0, x).detach()[0])
    assert np.allclose(np.mean(ests), exact, atol=1e-10)


def test_hutchinson_differentiable_to_params() -> None:
    w = torch.tensor([[0.5, -0.3], [0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(2, dtype=torch.float64)
    vel = _nonlinear_velocity(w, b)
    x = torch.randn(5, 2, dtype=torch.float64)
    noise = (torch.randint(0, 2, (5, 2)).double() * 2.0 - 1.0)
    tops.hutchinson_trace_jacobian(vel, 0.0, x, noise).sum().backward()
    assert w.grad is not None and bool(torch.isfinite(w.grad).all())


def test_exact_trace_differentiable_to_params() -> None:
    # confirms the exact estimator trains: the divergence is differentiable in the
    # velocity parameters (create_graph follows the ambient grad mode).
    w = torch.tensor([[0.5, -0.3], [0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(2, dtype=torch.float64)
    vel = _nonlinear_velocity(w, b)
    x = torch.randn(5, 2, dtype=torch.float64)
    tops.exact_trace_jacobian(vel, 0.0, x).sum().backward()
    assert w.grad is not None and bool(torch.isfinite(w.grad).all())


def test_integrate_hutchinson_mean_matches_exact_d2() -> None:
    # the x-trajectory is independent of the trace estimator, and the RK4 log-density
    # increment is linear in the divergence, so the mean over the 4 sign patterns of the
    # *integrated* delta_log_p equals the exact integrated delta_log_p to machine precision.
    w = torch.tensor([[0.3, -0.6], [0.5, 0.2]], dtype=torch.float64)
    b = torch.tensor([0.05, -0.1], dtype=torch.float64)
    vel = _nonlinear_velocity(w, b)
    x0 = torch.randn(6, 2, dtype=torch.float64)
    exact_x1, exact_delta = tops.integrate_cnf(vel, x0, 0.0, 1.0, steps=30)
    deltas = []
    for e in _rademacher_basis_d2():
        noise = e.expand_as(x0)

        def tf(vf, t, x, n=noise):  # type: ignore[no-untyped-def]
            return tops.hutchinson_trace_jacobian(vf, t, x, n)

        x1_h, delta_h = tops.integrate_cnf(vel, x0, 0.0, 1.0, steps=30, trace_fn=tf)
        assert np.allclose(_np(x1_h), _np(exact_x1), atol=1e-12)  # x-path is estimator-independent
        deltas.append(delta_h)
    mean_delta = torch.stack(deltas).mean(dim=0)
    assert np.allclose(_np(mean_delta), _np(exact_delta), atol=1e-9)


# --- JAX Hutchinson twin: cross-backend parity + unbiasedness ----------------


def _nonlinear_velocity_jax(w, b):  # type: ignore[no-untyped-def]
    def velocity(t, x):  # type: ignore[no-untyped-def]
        return jnp.tanh(x @ w.T + b)

    return velocity


_RADEMACHER_D2 = np.array(
    [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=DTYPE
)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_hutchinson_cross_backend_parity() -> None:
    # torch uses a reverse-mode VJP and jax a forward-mode JVP, but both evaluate the
    # same quadratic form eps^T J eps, so the estimates match bit-closely.
    w_np = np.array([[0.5, -0.3], [0.2, 0.7]], dtype=DTYPE)
    b_np = np.array([0.1, -0.2], dtype=DTYPE)
    x_np = np.array([[0.3, -0.4], [1.0, 0.5], [-0.7, 0.2]], dtype=DTYPE)
    noise_np = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=DTYPE)

    tvel = _nonlinear_velocity(
        torch.as_tensor(w_np, dtype=torch.float64), torch.as_tensor(b_np, dtype=torch.float64)
    )
    jvel = _nonlinear_velocity_jax(jnp.asarray(w_np), jnp.asarray(b_np))
    t_est = _np(
        tops.hutchinson_trace_jacobian(
            tvel, 0.0, torch.as_tensor(x_np, dtype=torch.float64), torch.as_tensor(noise_np, dtype=torch.float64)
        )
    )
    j_est = _np(jops.hutchinson_trace_jacobian(jvel, 0.0, jnp.asarray(x_np), jnp.asarray(noise_np)))
    assert np.allclose(t_est, j_est, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_hutchinson_exact_in_expectation_d2_jax() -> None:
    # The mean over ALL 2^d Rademacher patterns is tr(J) to machine precision (jax twin).
    w = jnp.asarray(np.array([[0.5, -0.3], [0.2, 0.7]], dtype=DTYPE))
    b = jnp.asarray(np.array([0.1, -0.2], dtype=DTYPE))
    vel = _nonlinear_velocity_jax(w, b)
    x = jnp.asarray(np.array([[0.3, -0.4], [1.0, 0.5], [-0.7, 0.2]], dtype=DTYPE))
    exact = np.asarray(jops.exact_trace_jacobian(vel, 0.0, x))
    ests = [
        np.asarray(jops.hutchinson_trace_jacobian(vel, 0.0, x, jnp.broadcast_to(jnp.asarray(e), x.shape)))
        for e in _RADEMACHER_D2
    ]
    assert np.allclose(np.mean(ests, axis=0), exact, atol=1e-10)


def test_callable_velocity_mixed_partial_raises_clear_message_torch() -> None:
    """Bare ``NotImplementedError`` is forbidden; the message must name the gap."""
    from omnibias.fields.torch import _ops_dispatch
    from omnibias.score.flow.torch.ops import cnf as cnf_mod

    field = cnf_mod._CallableVelocityField(
        lambda t, x: x, 0.0, 2, _ops_dispatch
    )
    state = field.evaluate(torch.zeros(3, 2, dtype=torch.float64))
    with pytest.raises(NotImplementedError, match="mixed_partial is not supported"):
        field.mixed_partial(state, "v0", (0, 1), (1, 1))


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_callable_velocity_mixed_partial_raises_clear_message_jax() -> None:
    from omnibias.fields.jax import _ops_dispatch
    from omnibias.score.flow.jax.ops import cnf as cnf_mod

    field = cnf_mod._CallableVelocityField(
        lambda t, x: x, 0.0, 2, _ops_dispatch
    )
    state = field.evaluate(jnp.zeros((3, 2), dtype=jnp.float64))
    with pytest.raises(NotImplementedError, match="mixed_partial is not supported"):
        field.mixed_partial(state, "v0", (0, 1), (1, 1))


@pytest.mark.skipif(jax is None, reason="jax not installed")
def test_integrate_hutchinson_jax_mean_matches_exact_d2() -> None:
    # The x-trajectory is estimator-independent and the RK4 log-density increment is
    # linear in the divergence, so the mean integrated delta_log_p over the 4 sign
    # patterns equals the exact integrated delta_log_p (jax trace_fn plumbing).
    w = jnp.asarray(np.array([[0.3, -0.6], [0.5, 0.2]], dtype=DTYPE))
    b = jnp.asarray(np.array([0.05, -0.1], dtype=DTYPE))
    vel = _nonlinear_velocity_jax(w, b)
    x0 = jnp.asarray(np.random.default_rng(0).standard_normal((6, 2)))
    exact_x1, exact_delta = jops.integrate_cnf(vel, x0, 0.0, 1.0, steps=30)
    deltas = []
    for e in _RADEMACHER_D2:
        noise = jnp.broadcast_to(jnp.asarray(e), x0.shape)

        def tf(vf, t, x, n=noise):  # type: ignore[no-untyped-def]
            return jops.hutchinson_trace_jacobian(vf, t, x, n)

        x1_h, delta_h = jops.integrate_cnf(vel, x0, 0.0, 1.0, steps=30, trace_fn=tf)
        assert np.allclose(np.asarray(x1_h), np.asarray(exact_x1), atol=1e-12)
        deltas.append(np.asarray(delta_h))
    assert np.allclose(np.mean(deltas, axis=0), np.asarray(exact_delta), atol=1e-9)
