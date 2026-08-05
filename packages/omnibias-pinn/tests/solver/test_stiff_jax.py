# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stiff integrators: phi functions, order of accuracy, and stability (jax).

Twin of ``test_stiff.py``, plus the two things only JAX can be asked: that a
step survives ``jit`` (with the squaring count pinned, since it is a Python
int) and that ``grad`` flows through a whole rollout.
"""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import omnibias.pinn.solver.jax as pj  # noqa: E402
from omnibias.pinn.solver.jax import stiff  # noqa: E402


def _linear_problem():
    """``u' = A u`` with a decade-1000 stiffness ratio."""
    a = jnp.asarray([[-1000.0, 1.0], [0.0, -1.0]])
    u0 = jnp.asarray([1.0, 1.0])
    return a, u0, (lambda u: a @ u)


def _expm(a):
    return stiff.phi_matrix(a, 0)[0]


# ---------------------------------------------------------------- phi ----


def test_phi_matches_the_closed_form_where_the_closed_form_is_safe() -> None:
    z = jnp.asarray([0.3, 1.0, 5.0, 40.0, -1.0, -20.0, -300.0])
    p = stiff.phi_diagonal(z, 3)
    assert jnp.allclose(p[0], jnp.exp(z), rtol=1e-12, atol=0.0)
    assert jnp.allclose(p[1], (jnp.exp(z) - 1) / z, rtol=1e-12, atol=0.0)
    assert jnp.allclose(p[2], (jnp.exp(z) - 1 - z) / z**2, rtol=1e-12, atol=0.0)
    assert jnp.allclose(
        p[3], (jnp.exp(z) - 1 - z - z**2 / 2) / z**3, rtol=1e-12, atol=0.0
    )


def test_phi_stays_accurate_where_the_closed_form_cancels_away() -> None:
    z = jnp.asarray([0.0, 1e-14, 1e-8])
    p = stiff.phi_diagonal(z, 3)
    assert float(p[1][0]) == pytest.approx(1.0)
    assert float(p[2][0]) == pytest.approx(0.5)
    assert float(p[3][0]) == pytest.approx(1.0 / 6.0)
    assert float(p[1][2]) > 1.0
    assert float(p[1][2]) == pytest.approx(1.0 + 0.5e-8)


def test_phi_handles_a_complex_symbol() -> None:
    z = jnp.asarray([3j, -2.0 + 5j, -30.0 + 1j])
    p = stiff.phi_diagonal(z, 2)
    assert jnp.allclose(p[0], jnp.exp(z), rtol=1e-12, atol=1e-14)
    assert jnp.allclose(p[1], (jnp.exp(z) - 1) / z, rtol=1e-12, atol=1e-14)


def test_phi_matrix_satisfies_its_defining_identities() -> None:
    a = jax.random.normal(jax.random.PRNGKey(0), (6, 6)) * 3.0
    eye = jnp.eye(6)
    p = stiff.phi_matrix(a, 2)
    expm = _expm(a)
    assert jnp.allclose(a @ p[1], expm - eye, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(a @ a @ p[2], expm - eye - a, rtol=1e-11, atol=1e-11)


def test_explicit_squarings_reproduce_the_automatic_choice() -> None:
    z = jnp.asarray([-40.0, 0.5, 12.0])
    assert jnp.allclose(
        stiff.phi_diagonal(z, 2), stiff.phi_diagonal(z, 2, squarings=7)
    )


# ------------------------------------------------------------- ROS2 ------


def test_rosenbrock_is_second_order() -> None:
    a, u0, f = _linear_problem()

    def err(h: float) -> float:
        u = u0
        for _ in range(int(round(1.0 / h))):
            u = stiff.rosenbrock_step(f, u, h, jacobian=a)
        return float(jnp.abs(u - _expm(a) @ u0).max())

    e1, e2 = err(0.05), err(0.025)
    assert e2 < e1
    assert 3.0 < e1 / e2 < 5.0


def test_rosenbrock_is_l_stable_where_rk4_explodes() -> None:
    a, u0, f = _linear_problem()
    landed = stiff.rosenbrock_step(f, u0, 1e6, jacobian=a)
    assert float(jnp.abs(landed).max()) < 1e-5

    blown = u0
    for _ in range(20):
        blown = pj.rk4_step(f, blown, 0.1)
    assert float(jnp.abs(blown).max()) > 1e10


def test_rosenbrock_works_without_being_handed_a_jacobian() -> None:
    a, u0, f = _linear_problem()
    assert jnp.allclose(
        stiff.rosenbrock_step(f, u0, 0.01),
        stiff.rosenbrock_step(f, u0, 0.01, jacobian=a),
        rtol=1e-12,
        atol=1e-14,
    )


def test_rosenbrock_survives_a_stiff_nonlinear_problem() -> None:
    """Van der Pol at mu = 500: explicit stepping is hopeless, ROS2 is not."""
    mu = 500.0

    def f(u):
        return jnp.stack([u[1], mu * (1.0 - u[0] ** 2) * u[1] - u[0]])

    u = jnp.asarray([2.0, 0.0])
    for _ in range(200):
        u = stiff.rosenbrock_step(f, u, 1e-3)
    assert bool(jnp.isfinite(u).all())
    assert float(jnp.abs(u).max()) < 10.0


# ----------------------------------------------- exponential Rosenbrock --


def test_exponential_euler_is_exact_for_an_affine_right_hand_side() -> None:
    a, u0, _ = _linear_problem()
    b = jnp.asarray([2.0, -1.0])

    def g(u):
        return a @ u + b

    for h in (0.5, 5.0, 50.0):
        p = stiff.phi_matrix(h * a, 1)
        ref = _expm(h * a) @ u0 + h * (p[1] @ b)
        got = stiff.exponential_rosenbrock_step(g, u0, h, jacobian=a)
        assert jnp.allclose(got, ref, rtol=1e-10, atol=1e-12)


# --------------------------------------------------------- Jacobians -----


def test_closed_form_jacobian_matches_autodiff_exactly() -> None:
    from omnibias.jax.activations import get_activation

    spec = get_activation("tanh")
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    w1 = jax.random.normal(keys[0], (5, 3))
    b1 = jax.random.normal(keys[1], (5,))
    w2 = jax.random.normal(keys[2], (3, 5))
    b2 = jax.random.normal(keys[3], (3,))
    layers = [(w1, b1, spec), (w2, b2, None)]

    def rhs(u):
        return w2 @ spec.forward(w1 @ u + b1) + b2

    u = jnp.asarray([0.3, -0.2, 0.7])
    assert jnp.allclose(
        stiff.closed_form_jacobian(layers, u),
        stiff.dense_jacobian(rhs, u),
        rtol=1e-13,
        atol=1e-14,
    )


def test_flat_state_is_required() -> None:
    with pytest.raises(ValueError, match="flat state"):
        stiff.rosenbrock_step(lambda u: u, jnp.zeros((2, 2)), 0.1)


# ------------------------------------------------------------ spectral ---


def _heat_setup(n: int = 64):
    grid = pj.SpectralGrid1D(n, 2.0 * math.pi)
    x = grid.points()
    u0 = jnp.sin(x) + 0.5 * jnp.cos(2.0 * x)
    symbol = -1.0 * grid.k**2

    def exact(t: float):
        uh = jnp.fft.fft(u0) * jnp.exp(symbol.astype(jnp.complex128) * t)
        return jnp.real(jnp.fft.ifft(uh))

    return grid, symbol, u0, exact


def test_etdrk4_integrates_a_purely_linear_problem_exactly() -> None:
    _, symbol, u0, exact = _heat_setup()
    for h in (0.05, 0.5, 5.0):
        u = u0
        for _ in range(2):
            u = stiff.etdrk4_step(symbol, jnp.zeros_like, u, h)
        assert jnp.allclose(u, exact(2 * h), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    ("step", "expected_ratio"), [("imex_euler", 2.0), ("imex_cnab2", 4.0)]
)
def test_imex_schemes_have_their_advertised_order(
    step: str, expected_ratio: float
) -> None:
    _, symbol, u0, exact = _heat_setup()

    def err(h: float) -> float:
        u, prev = u0, None
        for _ in range(int(round(0.5 / h))):
            if step == "imex_euler":
                u = stiff.imex_euler_step(symbol, jnp.zeros_like, u, h)
            else:
                u, prev = stiff.imex_cnab2_step(
                    symbol, jnp.zeros_like, u, h, previous_nonlinear=prev
                )
        return float(jnp.abs(u - exact(0.5)).max())

    ratio = err(0.05) / err(0.025)
    assert 0.7 * expected_ratio < ratio < 1.4 * expected_ratio


def _ks_setup(n: int = 128):
    grid = pj.SpectralGrid1D(n, 32.0 * math.pi)
    x = grid.points()
    u0 = jnp.cos(x / 16.0) * (1.0 + jnp.sin(x / 16.0))
    return grid, pj.kuramoto_sivashinsky_semidiscrete(grid), u0


def test_etdrk4_holds_kuramoto_sivashinsky_where_rk4_blows_up() -> None:
    _, semi, u0 = _ks_setup()
    assert float(jnp.abs(semi.symbol).max()) > 200.0

    coarse, fine = 0.25, 0.03125
    u_coarse = u0
    for _ in range(int(round(5.0 / coarse))):
        u_coarse = stiff.etdrk4_step(semi.symbol, semi.nonlinear, u_coarse, coarse)
    u_fine = u0
    for _ in range(int(round(5.0 / fine))):
        u_fine = stiff.etdrk4_step(semi.symbol, semi.nonlinear, u_fine, fine)
    assert bool(jnp.isfinite(u_coarse).all())
    assert float(jnp.abs(u_coarse - u_fine).max()) < 1e-4

    blown = u0
    for _ in range(int(round(5.0 / coarse))):
        blown = pj.rk4_step(semi.rhs, blown, coarse)
    assert not bool(jnp.isfinite(blown).all())


def test_method_of_lines_exposes_the_stiff_schemes() -> None:
    _, semi, u0 = _ks_setup()
    times = [0.0, 1.0, 2.0]
    for scheme in ("etdrk4", "imex_euler", "imex_cnab2"):
        snaps, _ = pj.method_of_lines(semi, u0, times, integrator=scheme)
        assert snaps.shape == (3, u0.shape[0])
        assert bool(jnp.isfinite(snaps).all())


def test_an_implicit_linear_step_refuses_a_split_problem() -> None:
    grid = pj.SpectralGrid1D(32, 2.0 * math.pi)
    semi = pj.burgers_semidiscrete(grid, 0.05)
    assert semi.symbol is not None and not semi.is_linear
    with pytest.raises(ValueError, match="drop"):
        pj.method_of_lines(
            semi, jnp.sin(grid.points()), [0.0, 0.1], integrator="implicit_euler"
        )


# ---------------------------------------------------------- jit / grad ---


def test_a_step_survives_jit_with_the_squaring_count_pinned() -> None:
    _, symbol, u0, _ = _heat_setup()

    def body(u):
        return stiff.etdrk4_step(symbol, jnp.zeros_like, u, 0.1, squarings=8)

    jitted = jax.jit(body)
    assert jnp.allclose(jitted(u0), body(u0), rtol=1e-12, atol=1e-14)


def test_grad_flows_through_a_whole_rollout() -> None:
    """The layer claim: a loss on the last state reaches the RHS parameters."""

    def loss(theta):
        def rhs(u):
            return theta * u

        traj = stiff.stiff_rollout(
            lambda u, h: stiff.rosenbrock_step(rhs, u, h),
            jnp.ones(1),
            dt=0.01,
            n_steps=8,
        )
        return jnp.sum(traj[-1])

    g = jax.grad(loss)(jnp.asarray([-30.0]))
    assert bool(jnp.isfinite(g).all())
    assert float(jnp.abs(g).max()) > 0.0
