# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stiff integrators: phi functions, order of accuracy, and stability (torch).

The claims under test are the ones that justify the module existing at all:

* the phi functions are accurate *both* where the closed form cancels (tiny
  ``z``) and where it overflows the series (``|z| ~ 300``);
* ROS2 is second order and **L-stable** -- an infinitely stiff mode is
  annihilated, not merely bounded;
* the exponential Rosenbrock-Euler step is *exact* for an affine right-hand
  side at any step size;
* ETDRK4 integrates the linear part exactly and stays stable on
  Kuramoto-Sivashinsky at a step where RK4 produces NaN;
* every step is differentiable end to end, which is what makes it a layer.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver.torch as pt  # noqa: E402
from omnibias.pinn.solver.torch import stiff  # noqa: E402

DT = torch.float64


def _linear_problem():
    """``u' = A u`` with a decade-1000 stiffness ratio."""
    a = torch.tensor([[-1000.0, 1.0], [0.0, -1.0]], dtype=DT)
    u0 = torch.tensor([1.0, 1.0], dtype=DT)
    return a, u0, (lambda u: a @ u)


# ---------------------------------------------------------------- phi ----


def test_phi_matches_the_closed_form_where_the_closed_form_is_safe() -> None:
    z = torch.tensor([0.3, 1.0, 5.0, 40.0, -1.0, -20.0, -300.0], dtype=DT)
    p = stiff.phi_diagonal(z, 3)
    torch.testing.assert_close(p[0], torch.exp(z), rtol=1e-12, atol=0.0)
    torch.testing.assert_close(p[1], (torch.exp(z) - 1) / z, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(
        p[2], (torch.exp(z) - 1 - z) / z**2, rtol=1e-12, atol=0.0
    )
    torch.testing.assert_close(
        p[3], (torch.exp(z) - 1 - z - z**2 / 2) / z**3, rtol=1e-12, atol=0.0
    )


def test_phi_stays_accurate_where_the_closed_form_cancels_away() -> None:
    """``(e^z - 1)/z`` at ``z = 1e-14`` has no significant digits left; the series does."""
    z = torch.tensor([0.0, 1e-14, 1e-8], dtype=DT)
    p = stiff.phi_diagonal(z, 3)
    torch.testing.assert_close(p[1][0], torch.tensor(1.0, dtype=DT))
    torch.testing.assert_close(p[2][0], torch.tensor(0.5, dtype=DT))
    torch.testing.assert_close(p[3][0], torch.tensor(1.0 / 6.0, dtype=DT))
    # phi_1(z) = 1 + z/2 + O(z^2) -- resolved, not rounded to 1.
    assert float(p[1][2]) > 1.0
    torch.testing.assert_close(p[1][2], torch.tensor(1.0 + 0.5e-8, dtype=DT))


def test_phi_handles_a_complex_symbol() -> None:
    z = torch.tensor([1j * 3.0, -2.0 + 5j, -30.0 + 1j], dtype=torch.complex128)
    p = stiff.phi_diagonal(z, 2)
    torch.testing.assert_close(p[0], torch.exp(z), rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(p[1], (torch.exp(z) - 1) / z, rtol=1e-12, atol=1e-14)


def test_phi_matrix_satisfies_its_defining_identities() -> None:
    torch.manual_seed(0)
    a = torch.randn(6, 6, dtype=DT) * 3.0
    eye = torch.eye(6, dtype=DT)
    p = stiff.phi_matrix(a, 2)
    expm = torch.matrix_exp(a)
    torch.testing.assert_close(p[0], expm, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(a @ p[1], expm - eye, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(a @ a @ p[2], expm - eye - a, rtol=1e-11, atol=1e-11)


def test_phi_matrix_rejects_a_non_square_argument() -> None:
    with pytest.raises(ValueError, match="square"):
        stiff.phi_matrix(torch.zeros(2, 3, dtype=DT), 1)


def test_explicit_squarings_reproduce_the_automatic_choice() -> None:
    z = torch.tensor([-40.0, 0.5, 12.0], dtype=DT)
    torch.testing.assert_close(
        stiff.phi_diagonal(z, 2), stiff.phi_diagonal(z, 2, squarings=7)
    )


# ------------------------------------------------------------- ROS2 ------


def test_rosenbrock_is_second_order() -> None:
    a, u0, f = _linear_problem()

    def err(h: float) -> float:
        u = u0
        for _ in range(int(round(1.0 / h))):
            u = stiff.rosenbrock_step(f, u, h, jacobian=a)
        return float((u - torch.matrix_exp(a) @ u0).abs().max())

    e1, e2 = err(0.05), err(0.025)
    assert e2 < e1
    assert 3.0 < e1 / e2 < 5.0  # halving the step quarters the error


def test_rosenbrock_is_l_stable_where_rk4_explodes() -> None:
    a, u0, f = _linear_problem()
    landed = stiff.rosenbrock_step(f, u0, 1e6, jacobian=a)
    assert float(landed.abs().max()) < 1e-5  # R(inf) = 0: annihilated in one step

    blown = u0
    for _ in range(20):
        blown = pt.rk4_step(f, blown, 0.1)
    assert float(blown.abs().max()) > 1e10


def test_rosenbrock_works_without_being_handed_a_jacobian() -> None:
    a, u0, f = _linear_problem()
    with_j = stiff.rosenbrock_step(f, u0, 0.01, jacobian=a)
    auto = stiff.rosenbrock_step(f, u0, 0.01)
    torch.testing.assert_close(auto, with_j, rtol=1e-12, atol=1e-14)


def test_rosenbrock_accepts_a_jacobian_callable() -> None:
    a, u0, f = _linear_problem()
    called: list[int] = []

    def jac(u: torch.Tensor) -> torch.Tensor:
        called.append(1)
        return a

    got = stiff.rosenbrock_step(f, u0, 0.01, jacobian=jac)
    torch.testing.assert_close(got, stiff.rosenbrock_step(f, u0, 0.01, jacobian=a))
    assert len(called) == 1  # one factorisation per step, not one per stage


def test_rosenbrock_survives_a_stiff_nonlinear_problem() -> None:
    """Van der Pol at mu = 500: explicit stepping is hopeless, ROS2 is not."""
    mu = 500.0

    def f(u: torch.Tensor) -> torch.Tensor:
        return torch.stack([u[1], mu * (1.0 - u[0] ** 2) * u[1] - u[0]])

    u = torch.tensor([2.0, 0.0], dtype=DT)
    for _ in range(200):
        u = stiff.rosenbrock_step(f, u, 1e-3)
    assert torch.isfinite(u).all()
    assert float(u.abs().max()) < 10.0  # on the limit cycle, not diverging


# ----------------------------------------------- exponential Rosenbrock --


def test_exponential_euler_is_exact_for_an_affine_right_hand_side() -> None:
    a, u0, _ = _linear_problem()
    b = torch.tensor([2.0, -1.0], dtype=DT)

    def g(u: torch.Tensor) -> torch.Tensor:
        return a @ u + b

    for h in (0.5, 5.0, 50.0):
        got = stiff.exponential_rosenbrock_step(g, u0, h, jacobian=a)
        p = stiff.phi_matrix(h * a, 1)
        ref = torch.matrix_exp(h * a) @ u0 + h * (p[1] @ b)
        torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-12)


def test_exponential_euler_is_second_order_on_a_nonlinear_problem() -> None:
    def f(u: torch.Tensor) -> torch.Tensor:
        return torch.stack([-50.0 * u[0] + u[1] ** 2, -u[1]])

    u0 = torch.tensor([1.0, 1.0], dtype=DT)

    def march(h: float) -> torch.Tensor:
        u = u0
        for _ in range(int(round(0.2 / h))):
            u = stiff.exponential_rosenbrock_step(f, u, h)
        return u

    ref = march(0.2 / 2048)
    e1 = float((march(0.02) - ref).abs().max())
    e2 = float((march(0.01) - ref).abs().max())
    assert 3.0 < e1 / e2 < 5.0


# --------------------------------------------------------- Jacobians -----


def test_closed_form_jacobian_matches_autodiff_exactly() -> None:
    from omnibias.torch.activations.registry import get_activation

    spec = get_activation("tanh")
    torch.manual_seed(0)
    w1 = torch.randn(5, 3, dtype=DT)
    b1 = torch.randn(5, dtype=DT)
    w2 = torch.randn(3, 5, dtype=DT)
    b2 = torch.randn(3, dtype=DT)
    layers = [(w1, b1, spec), (w2, b2, None)]

    def rhs(u: torch.Tensor) -> torch.Tensor:
        return w2 @ spec.forward(w1 @ u + b1) + b2

    u = torch.randn(3, dtype=DT)
    closed = stiff.closed_form_jacobian(layers, u)
    torch.testing.assert_close(
        closed, stiff.dense_jacobian(rhs, u), rtol=1e-13, atol=1e-14
    )


def test_a_closed_form_jacobian_drives_a_stiff_step() -> None:
    """The point of the closed form: a neural RHS stepped with no autodiff."""
    from omnibias.torch.activations.registry import get_activation

    spec = get_activation("tanh")
    torch.manual_seed(1)
    w1 = torch.randn(6, 2, dtype=DT)
    b1 = torch.zeros(6, dtype=DT)
    w2 = torch.randn(2, 6, dtype=DT) * 10.0
    b2 = torch.zeros(2, dtype=DT)
    layers = [(w1, b1, spec), (w2, b2, None)]

    def rhs(u: torch.Tensor) -> torch.Tensor:
        return w2 @ spec.forward(w1 @ u + b1) + b2

    u0 = torch.tensor([0.3, -0.2], dtype=DT)
    exact = stiff.rosenbrock_step(rhs, u0, 0.01)
    jetted = stiff.rosenbrock_step(
        rhs, u0, 0.01, jacobian=lambda u: stiff.closed_form_jacobian(layers, u)
    )
    torch.testing.assert_close(jetted, exact, rtol=1e-12, atol=1e-14)


def test_flat_state_is_required() -> None:
    with pytest.raises(ValueError, match="flat state"):
        stiff.rosenbrock_step(lambda u: u, torch.zeros(2, 2, dtype=DT), 0.1)


# ------------------------------------------------------------ spectral ---


def _heat_setup(n: int = 64):
    grid = pt.SpectralGrid1D(n, 2.0 * math.pi)
    x = grid.points()
    u0 = torch.sin(x) + 0.5 * torch.cos(2.0 * x)
    symbol = -1.0 * grid.k**2

    def exact(t: float) -> torch.Tensor:
        uh = torch.fft.fft(u0) * torch.exp(symbol.to(torch.complex128) * t)
        return torch.fft.ifft(uh).real

    return grid, symbol, u0, exact


def test_etdrk4_integrates_a_purely_linear_problem_exactly() -> None:
    _, symbol, u0, exact = _heat_setup()
    zero = torch.zeros_like
    for h in (0.05, 0.5, 5.0):  # accuracy does not degrade with the step
        u = u0
        for _ in range(2):
            u = stiff.etdrk4_step(symbol, zero, u, h)
        torch.testing.assert_close(u, exact(2 * h), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    ("step", "expected_ratio"),
    [("imex_euler", 2.0), ("imex_cnab2", 4.0)],
)
def test_imex_schemes_have_their_advertised_order(
    step: str, expected_ratio: float
) -> None:
    _, symbol, u0, exact = _heat_setup()
    zero = torch.zeros_like

    def err(h: float) -> float:
        u, prev = u0, None
        for _ in range(int(round(0.5 / h))):
            if step == "imex_euler":
                u = stiff.imex_euler_step(symbol, zero, u, h)
            else:
                u, prev = stiff.imex_cnab2_step(
                    symbol, zero, u, h, previous_nonlinear=prev
                )
        return float((u - exact(0.5)).abs().max())

    ratio = err(0.05) / err(0.025)
    assert 0.7 * expected_ratio < ratio < 1.4 * expected_ratio


def test_cnab2_reports_the_nonlinear_term_for_the_next_step() -> None:
    _, symbol, u0, _ = _heat_setup()

    def nonlinear(u: torch.Tensor) -> torch.Tensor:
        return -0.5 * u

    _, reported = stiff.imex_cnab2_step(symbol, nonlinear, u0, 0.01)
    torch.testing.assert_close(reported, nonlinear(u0))


def _ks_setup(n: int = 128):
    grid = pt.SpectralGrid1D(n, 32.0 * math.pi)
    x = grid.points()
    u0 = torch.cos(x / 16.0) * (1.0 + torch.sin(x / 16.0))
    return grid, pt.kuramoto_sivashinsky_semidiscrete(grid), u0


def test_etdrk4_holds_kuramoto_sivashinsky_where_rk4_blows_up() -> None:
    grid, semi, u0 = _ks_setup()
    assert float(semi.symbol.abs().max()) > 200.0  # genuinely stiff

    coarse, fine = 0.25, 0.03125
    u_coarse = u0
    for _ in range(int(round(5.0 / coarse))):
        u_coarse = stiff.etdrk4_step(semi.symbol, semi.nonlinear, u_coarse, coarse)
    u_fine = u0
    for _ in range(int(round(5.0 / fine))):
        u_fine = stiff.etdrk4_step(semi.symbol, semi.nonlinear, u_fine, fine)
    assert torch.isfinite(u_coarse).all()
    assert float((u_coarse - u_fine).abs().max()) < 1e-4

    blown = u0
    for _ in range(int(round(5.0 / coarse))):
        blown = pt.rk4_step(semi.rhs, blown, coarse)
    assert not torch.isfinite(blown).all()


def test_method_of_lines_exposes_the_stiff_schemes() -> None:
    _, semi, u0 = _ks_setup()
    times = [0.0, 1.0, 2.0]
    for scheme in ("etdrk4", "imex_euler", "imex_cnab2"):
        snaps, ts = pt.method_of_lines(semi, u0, times, integrator=scheme)
        assert snaps.shape == (3, u0.shape[0])
        assert torch.isfinite(snaps).all()
        assert ts.tolist() == times


def test_an_implicit_linear_step_refuses_a_split_problem() -> None:
    """Burgers now carries a symbol; it must not be integrated as if linear."""
    grid = pt.SpectralGrid1D(32, 2.0 * math.pi)
    semi = pt.burgers_semidiscrete(grid, 0.05)
    assert semi.symbol is not None and not semi.is_linear
    with pytest.raises(ValueError, match="drop"):
        pt.method_of_lines(semi, torch.sin(grid.points()), [0.0, 0.1], integrator="implicit_euler")


def test_a_stiff_scheme_refuses_a_problem_with_no_symbol() -> None:
    grid = pt.SpectralGrid1D(32, 2.0 * math.pi)
    semi = pt.reaction_diffusion_semidiscrete(
        grid, (0.1, 0.1), lambda u, v: (-u * v, u * v)
    )
    u0 = torch.stack([torch.sin(grid.points()), torch.cos(grid.points())])
    with pytest.raises(ValueError, match="has none"):
        pt.method_of_lines(semi, u0, [0.0, 0.1], integrator="etdrk4")


def test_burgers_agrees_across_the_stiff_and_jet_taylor_routes() -> None:
    """Two unrelated integrators, one answer -- the cross-check that matters."""
    grid = pt.SpectralGrid1D(64, 2.0 * math.pi)
    semi = pt.burgers_semidiscrete(grid, 0.1)
    u0 = torch.sin(grid.points())
    times = [i * 0.02 for i in range(11)]
    etd, _ = pt.method_of_lines(semi, u0, times, integrator="etdrk4")
    jet, _ = pt.method_of_lines(semi, u0, times, integrator="jet_taylor", order=8)
    assert float((etd[-1] - jet[-1]).abs().max()) < 1e-6


# -------------------------------------------------------------- layer ----


def test_a_rollout_is_differentiable_end_to_end() -> None:
    """The layer claim: a loss on the last state reaches the RHS parameters."""
    theta = torch.tensor([-30.0], dtype=DT, requires_grad=True)

    def rhs(u: torch.Tensor) -> torch.Tensor:
        return theta * u

    u0 = torch.ones(1, dtype=DT)
    traj = stiff.stiff_rollout(
        lambda u, h: stiff.rosenbrock_step(rhs, u, h), u0, dt=0.01, n_steps=8
    )
    assert traj.shape == (9, 1)
    traj[-1].sum().backward()
    assert theta.grad is not None
    assert float(theta.grad.abs()) > 0.0


def test_rollout_rejects_a_negative_step_count() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        stiff.stiff_rollout(
            lambda u, h: u, torch.ones(2, dtype=DT), dt=0.1, n_steps=-1
        )
