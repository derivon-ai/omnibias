# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fredholm & Volterra integral equations on the measure integral.

The four solvers make different promises, so each is checked against the thing it
actually claims rather than against one another:

* :func:`nystrom_solve` -- against **analytic solutions** of separable problems
  whose exact answer closes in a line of algebra, and against the measure's own
  rule: on a smooth kernel a Gauss-Legendre measure must be spectrally accurate
  where a trapezoid rule of the same node count is second order.
* :func:`degenerate_kernel_solve` -- the **oracle**, exact in the kernel, so it is
  what the Nystrom solve is compared to on a finite-rank problem rather than the
  reverse.
* :func:`volterra_solve` -- against ``u = e^x``, the solution of
  ``u = 1 + int_0^x u``, with the second-order convergence its cumulative rule
  implies verified as a *rate*, not a single tolerance.
* :func:`neumann_series` -- both inside and, importantly, **outside** its radius.
  The divergent case is the one failure mode in this module that produces a
  finite, plausible, wrong array, so the test asserts the report catches it and
  that ``nystrom_solve`` still gets the right answer at the same ``lam``.

Then: torch/jax/numpy parity, autograd through the kernel / source / ``lam``, and
the Fredholm-alternative error path where ``1/lam`` is an eigenvalue.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

from omnibias.measure._core import integraleq as C  # noqa: E402
from omnibias.measure._core.measure import Measure, lebesgue  # noqa: E402
from omnibias.measure.jax import integraleq as J  # noqa: E402
from omnibias.measure.torch import integraleq as T  # noqa: E402

#: ``u = 1 + lam int_0^1 x t u(t) dt`` has ``u = 1 + c x`` with
#: ``c = lam (1/2) / (1 - lam/3)``: substitute and match the ``x`` coefficient.
SEPARABLE_LAM = 1.0


def separable_exact(x: np.ndarray, lam: float = SEPARABLE_LAM) -> np.ndarray:
    c = lam * 0.5 / (1.0 - lam / 3.0)
    return 1.0 + c * x


def trapezoid_measure(a: float, b: float, n: int) -> Measure:
    x = np.linspace(a, b, n)
    w = np.full(n, (b - a) / (n - 1))
    w[0] *= 0.5
    w[-1] *= 0.5
    return Measure(nodes=x.reshape(-1, 1), weights=w, name="trapezoid")


# Kernel / source triples: numpy, torch, jax spellings of the same objects.
def k_np(x, t):
    return x[:, None, 0] * t[None, :, 0]


def k_torch(x, t):
    return x[:, 0:1] * t[:, 0].unsqueeze(0)


def k_jax(x, t):
    return x[:, 0:1] * t[None, :, 0]


def f_np(x):
    return np.ones(x.shape[0])


def f_torch(x):
    return torch.ones(x.shape[0], dtype=x.dtype)


def f_jax(x):
    return jnp.ones(x.shape[0])


def a_np(x):
    return x[:, 0]


# --------------------------------------------------------------------------- #
# Fredholm: against analytic solutions
# --------------------------------------------------------------------------- #
def test_nystrom_matches_the_analytic_separable_solution() -> None:
    mu = lebesgue([(0.0, 1.0)], 32)
    got = C.nystrom_solve(k_np, f_np, mu)
    assert np.abs(got - separable_exact(mu.nodes[:, 0])).max() < 1e-13


@pytest.mark.parametrize("lam", [-2.0, -0.5, 0.5, 2.0])
def test_nystrom_is_right_across_lambda_including_negative(lam: float) -> None:
    """Including ``lam`` past the Neumann radius, which a direct solve does not care about."""
    mu = lebesgue([(0.0, 1.0)], 24)
    got = C.nystrom_solve(k_np, f_np, mu, lam=lam)
    assert np.abs(got - separable_exact(mu.nodes[:, 0], lam)).max() < 1e-13


def test_the_degenerate_oracle_and_nystrom_agree_on_a_rank_two_kernel() -> None:
    """``K = x t + x^2 t^2``: the oracle never discretises ``K``, Nystrom does."""
    mu = lebesgue([(0.0, 1.0)], 40)

    def kernel(x, t):
        return x[:, None, 0] * t[None, :, 0] + (x[:, None, 0] ** 2) * (t[None, :, 0] ** 2)

    factors = [
        (lambda p: p[:, 0], lambda p: p[:, 0]),
        (lambda p: p[:, 0] ** 2, lambda p: p[:, 0] ** 2),
    ]
    oracle = C.degenerate_kernel_solve(factors, f_np, mu, lam=0.7)
    numeric = C.nystrom_solve(kernel, f_np, mu, lam=0.7)
    assert np.abs(numeric - oracle).max() < 1e-12
    # ...and the oracle really does satisfy the equation, checked independently
    assert np.abs(C.fredholm_residual(oracle, kernel, f_np, mu, lam=0.7)).max() < 1e-12


def test_a_better_measure_buys_a_better_answer() -> None:
    """The Nystrom error *is* the measure's quadrature error, so the rule matters.

    On a smooth non-separable kernel, 16 Gauss-Legendre nodes beat 16 trapezoid
    nodes by orders of magnitude -- which is the whole reason the solver takes a
    Measure instead of deriving weights from the grid itself.
    """

    def kernel(x, t):
        return np.exp(-((x[:, None, 0] - t[None, :, 0]) ** 2))

    def source(x):
        return np.cos(2.0 * x[:, 0])

    reference = C.nystrom_solve(kernel, source, lebesgue([(0.0, 1.0)], 200), lam=0.6)

    def error_at(mu: Measure) -> float:
        # Compare through the residual, since the two rules live on different nodes.
        return float(np.abs(C.fredholm_residual(
            C.nystrom_solve(kernel, source, mu, lam=0.6), kernel, source, mu, lam=0.6
        )).max())

    assert error_at(lebesgue([(0.0, 1.0)], 16)) < 1e-13
    assert reference.shape == (200,)

    # The honest comparison is against a common evaluation: solve on each rule and
    # measure how well each reproduces the reference solution's defining moment
    # int_0^1 u, which both rules can report.
    def moment(mu: Measure) -> float:
        u = C.nystrom_solve(kernel, source, mu, lam=0.6)
        return float(mu.weights @ u)

    truth = moment(lebesgue([(0.0, 1.0)], 200))
    gauss_err = abs(moment(lebesgue([(0.0, 1.0)], 16)) - truth)
    trap_err = abs(moment(trapezoid_measure(0.0, 1.0, 16)) - truth)
    assert gauss_err < 1e-12
    assert trap_err > 100.0 * max(gauss_err, 1e-15), (gauss_err, trap_err)


# --------------------------------------------------------------------------- #
# Volterra: causality, and the convergence order it really has
# --------------------------------------------------------------------------- #
def test_volterra_recovers_the_exponential() -> None:
    """``u(x) = 1 + int_0^x u`` is solved by ``e^x``."""
    mu = trapezoid_measure(0.0, 2.0, 801)
    x = mu.nodes[:, 0]
    got = C.volterra_solve(
        lambda p, q: np.ones((p.shape[0], q.shape[0])), f_np, mu
    )
    assert np.abs(got - np.exp(x)).max() < 1e-5


def test_volterra_converges_at_second_order() -> None:
    """A measured rate, not a tolerance: halving ``h`` must quarter the error."""
    errors = []
    for n in (201, 401, 801):
        mu = trapezoid_measure(0.0, 2.0, n)
        got = C.volterra_solve(
            lambda p, q: np.ones((p.shape[0], q.shape[0])), f_np, mu
        )
        errors.append(float(np.abs(got - np.exp(mu.nodes[:, 0])).max()))
    rates = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(1.8 < r < 2.2 for r in rates), (errors, rates)


def test_volterra_with_a_kernel_matches_its_analytic_solution() -> None:
    """``u(x) = 1 - int_0^x (x - t) u(t) dt`` is solved by ``cos x``.

    Differentiating twice turns it into ``u'' = -u`` with ``u(0)=1``, ``u'(0)=0``,
    which is the standard way an integral equation encodes an initial-value
    problem -- initial conditions and all, with no separate boundary term.
    """
    mu = trapezoid_measure(0.0, 3.0, 1201)
    x = mu.nodes[:, 0]
    got = C.volterra_solve(
        lambda p, q: p[:, None, 0] - q[None, :, 0], f_np, mu, lam=-1.0
    )
    assert np.abs(got - np.cos(x)).max() < 1e-5


def test_volterra_needs_a_sorted_one_dimensional_measure() -> None:
    flat = trapezoid_measure(0.0, 1.0, 8)
    with pytest.raises(ValueError, match="needs a 1-D measure"):
        C.volterra_solve(k_np, f_np, flat.product(flat))
    reversed_nodes = Measure(
        nodes=flat.nodes[::-1].copy(), weights=flat.weights[::-1].copy()
    )
    with pytest.raises(ValueError, match="strictly increasing nodes"):
        C.volterra_solve(k_np, f_np, reversed_nodes)


# --------------------------------------------------------------------------- #
# Neumann: the honest divergence report
# --------------------------------------------------------------------------- #
def test_neumann_converges_inside_its_radius() -> None:
    mu = lebesgue([(0.0, 1.0)], 32)
    result = C.neumann_series(k_np, f_np, mu, lam=0.5)
    assert result.converged
    assert result.spectral_radius < 1.0
    assert np.abs(result.solution - separable_exact(mu.nodes[:, 0], 0.5)).max() < 1e-10
    assert np.array_equal(result.raise_if_diverged(), result.solution)


def test_neumann_reports_divergence_instead_of_returning_garbage() -> None:
    """The one failure here that is otherwise silent: finite, plausible, wrong.

    The series radius for this kernel is ``|lam| / 3 < 1``, so ``lam = 5`` is well
    outside. What must *not* happen is a finite array coming back unflagged --
    and the direct solve must still be right at the same ``lam``, which is the
    actionable advice the error gives.
    """
    mu = lebesgue([(0.0, 1.0)], 32)
    result = C.neumann_series(k_np, f_np, mu, lam=5.0)
    assert not result.converged
    assert result.spectral_radius == pytest.approx(5.0 / 3.0, rel=1e-6)
    assert np.all(np.isfinite(result.solution)), "the trap: it looks like an answer"
    assert np.abs(result.solution - separable_exact(mu.nodes[:, 0], 5.0)).max() > 1.0
    with pytest.raises(ValueError, match=r"did not converge.*needs < 1"):
        result.raise_if_diverged()
    direct = C.nystrom_solve(k_np, f_np, mu, lam=5.0)
    assert np.abs(direct - separable_exact(mu.nodes[:, 0], 5.0)).max() < 1e-12


def test_neumann_truncation_is_reported_as_a_residual() -> None:
    """Stopped early, the partial sum is not the solution, and says so."""
    mu = lebesgue([(0.0, 1.0)], 32)
    short = C.neumann_series(k_np, f_np, mu, lam=0.9, max_terms=3)
    full = C.neumann_series(k_np, f_np, mu, lam=0.9, max_terms=200)
    assert short.n_terms == 3
    assert short.residual > full.residual
    assert full.converged and not short.converged


# --------------------------------------------------------------------------- #
# Fredholm alternative
# --------------------------------------------------------------------------- #
def test_an_eigenvalue_of_the_operator_is_named_as_such() -> None:
    """``lam = 3`` makes ``1/lam`` an eigenvalue of ``K W`` for ``K = x t`` on [0,1].

    The moment is :math:`\\int_0^1 x^2 = 1/3`, so ``1 - lam/3`` vanishes at
    ``lam = 3``. Both the rank-1 moment system and the full Nystrom system are
    singular there, and both must say so.
    """
    mu = lebesgue([(0.0, 1.0)], 32)
    with pytest.raises(ValueError, match="Fredholm alternative"):
        C.degenerate_kernel_solve([(a_np, a_np)], f_np, mu, lam=3.0)
    with pytest.raises(ValueError, match="Fredholm alternative"):
        C.nystrom_solve(lambda x, t: x[:, :1] * t[:, 0][None, :], f_np, mu, lam=3.0)


def test_all_three_backends_refuse_the_same_eigenvalue() -> None:
    """The guard is part of the contract, not a numpy-only nicety."""
    mu = lebesgue([(0.0, 1.0)], 32)
    pairs_np = [(a_np, a_np)]
    with pytest.raises(ValueError, match="Fredholm alternative"):
        C.degenerate_kernel_solve(pairs_np, f_np, mu, lam=3.0)
    with pytest.raises(ValueError, match="Fredholm alternative"):
        T.degenerate_kernel_solve(
            [(lambda p: p[:, 0], lambda p: p[:, 0])], f_torch, mu, lam=3.0
        )
    with pytest.raises(ValueError, match="Fredholm alternative"):
        J.degenerate_kernel_solve(
            [(lambda p: p[:, 0], lambda p: p[:, 0])], f_jax, mu, lam=3.0
        )


def test_the_singular_case_is_a_near_miss_not_an_exact_zero() -> None:
    """Why the guard is a condition number and not ``except LinAlgError``.

    In exact arithmetic ``1 - lam/3`` is zero at ``lam = 3`` and LAPACK would
    refuse. In floating point the moment integrates to ``1/3`` up to a rounding
    error, so the matrix is merely *tiny*, the solve succeeds, and it hands back a
    vector of order ``1e15`` that looks like an answer. This test pins that
    failure mode down so the guard cannot quietly be weakened back to a
    singularity test.
    """
    mu = lebesgue([(0.0, 1.0)], 32)
    unguarded = C.degenerate_kernel_solve(
        [(a_np, a_np)], f_np, mu, lam=3.0, check_conditioning=False
    )
    assert np.isfinite(unguarded).all()
    assert np.abs(unguarded).max() > 1e10


def test_opting_out_of_the_check_is_available_on_every_backend() -> None:
    """``check_conditioning=False`` is the escape hatch for compiled regions.

    It must not change the answer on a well-conditioned problem -- the guard reads
    the matrix, it never touches the solve.
    """
    mu = lebesgue([(0.0, 1.0)], 24)
    assert np.abs(
        C.nystrom_solve(k_np, f_np, mu, lam=0.8, check_conditioning=False)
        - C.nystrom_solve(k_np, f_np, mu, lam=0.8)
    ).max() == pytest.approx(0.0, abs=0.0)
    assert np.abs(
        T.nystrom_solve(k_torch, f_torch, mu, lam=0.8, check_conditioning=False).numpy()
        - T.nystrom_solve(k_torch, f_torch, mu, lam=0.8).numpy()
    ).max() == pytest.approx(0.0, abs=0.0)
    assert np.abs(
        np.asarray(J.nystrom_solve(k_jax, f_jax, mu, lam=0.8, check_conditioning=False))
        - np.asarray(J.nystrom_solve(k_jax, f_jax, mu, lam=0.8))
    ).max() == pytest.approx(0.0, abs=0.0)


def test_the_solvability_margin_locates_the_spectrum() -> None:
    """Sweeping ``lam`` through an eigenvalue drives the margin to zero.

    ``K = x t`` on [0,1] has the single nonzero eigenvalue ``1/3``, so the margin
    of ``I - lam K W`` collapses at ``lam = 3`` and recovers on either side. That
    is the diagnostic's whole use: it says *where* a family of equations stops
    being solvable, which is where the operator's spectrum is.
    """
    mu = lebesgue([(0.0, 1.0)], 32)
    kw = np.asarray(mu.nodes)[:, :1] * (
        np.asarray(mu.nodes)[:, 0] * np.asarray(mu.weights)
    )
    margins = {
        lam: C.solvability_margin(
            np.linalg.svd(np.eye(kw.shape[0]) - lam * kw, compute_uv=False)
        )
        for lam in (1.0, 2.5, 3.0, 3.5, 5.0)
    }
    assert margins[3.0] < C.SINGULAR_RCOND
    assert margins[2.5] > margins[3.0] < margins[3.5]
    assert margins[1.0] > 0.1 and margins[5.0] > 0.1


def test_a_traced_jax_solve_steps_around_the_check() -> None:
    """The documented jax behaviour, gated so the docstring cannot go stale.

    A condition number needs a matrix, and under ``jit`` / ``grad`` there is only
    a tracer -- so the guard stands down instead of crashing on the
    concretisation. The price is that the eigenvalue is not caught under a
    transform, which is exactly what the second half asserts.
    """
    mu = lebesgue([(0.0, 1.0)], 16)
    solve = jax.jit(lambda lam: J.nystrom_solve(k_jax, f_jax, mu, lam=lam))
    expected = C.nystrom_solve(k_np, f_np, mu, lam=0.5)
    assert np.abs(np.asarray(solve(0.5)) - expected).max() < 1e-13

    pairs = [(lambda p: p[:, 0], lambda p: p[:, 0])]
    traced = jax.jit(lambda lam: J.degenerate_kernel_solve(pairs, f_jax, mu, lam=lam))
    slipped_through = np.asarray(traced(3.0))
    assert np.abs(slipped_through).max() > 1e10
    with pytest.raises(ValueError, match="Fredholm alternative"):
        J.degenerate_kernel_solve(pairs, f_jax, mu, lam=3.0)


# --------------------------------------------------------------------------- #
# cross-backend parity
# --------------------------------------------------------------------------- #
def test_all_three_backends_agree() -> None:
    mu = lebesgue([(0.0, 1.0)], 32)
    core = C.nystrom_solve(k_np, f_np, mu, lam=0.8)
    tor = T.nystrom_solve(k_torch, f_torch, mu, lam=0.8).numpy()
    jx = np.asarray(J.nystrom_solve(k_jax, f_jax, mu, lam=0.8))
    assert np.abs(tor - core).max() < 1e-13
    assert np.abs(jx - core).max() < 1e-13


def test_every_solver_has_matching_backends() -> None:
    mu = lebesgue([(0.0, 1.0)], 24)
    pairs = [
        (lambda p: p[:, 0], lambda p: p[:, 0]),
    ]
    core = C.degenerate_kernel_solve(pairs, f_np, mu, lam=0.6)
    tor = T.degenerate_kernel_solve(pairs, f_torch, mu, lam=0.6).numpy()
    jx = np.asarray(J.degenerate_kernel_solve(pairs, f_jax, mu, lam=0.6))
    assert np.abs(tor - core).max() < 1e-13
    assert np.abs(jx - core).max() < 1e-13

    core_n = C.neumann_series(k_np, f_np, mu, lam=0.4)
    tor_n, tor_rep = T.neumann_series(k_torch, f_torch, mu, lam=0.4)
    jx_n, jx_rep = J.neumann_series(k_jax, f_jax, mu, lam=0.4)
    assert np.abs(tor_n.numpy() - core_n.solution).max() < 1e-13
    assert np.abs(np.asarray(jx_n) - core_n.solution).max() < 1e-13
    assert tor_rep.converged and jx_rep.converged
    assert tor_rep.n_terms == jx_rep.n_terms == core_n.n_terms

    line = trapezoid_measure(0.0, 1.0, 65)
    core_v = C.volterra_solve(
        lambda p, q: np.ones((p.shape[0], q.shape[0])), f_np, line
    )
    tor_v = T.volterra_solve(
        lambda p, q: torch.ones(p.shape[0], q.shape[0], dtype=p.dtype), f_torch, line
    ).numpy()
    jx_v = np.asarray(
        J.volterra_solve(
            lambda p, q: jnp.ones((p.shape[0], q.shape[0])), f_jax, line
        )
    )
    assert np.abs(tor_v - core_v).max() < 1e-13
    assert np.abs(jx_v - core_v).max() < 1e-13


def test_the_cumulative_matrices_agree() -> None:
    x = np.linspace(0.0, 2.0, 33)
    core = C.cumulative_trapezoid_matrix(x)
    tor = T.cumulative_trapezoid_matrix(torch.as_tensor(x)).numpy()
    jx = np.asarray(J.cumulative_trapezoid_matrix(jnp.asarray(x)))
    assert np.abs(tor - core).max() < 1e-15
    assert np.abs(jx - core).max() < 1e-15
    assert np.allclose(np.triu(core, k=1), 0.0), "must be lower triangular"


# --------------------------------------------------------------------------- #
# autograd: the point of having backend twins at all
# --------------------------------------------------------------------------- #
def test_gradients_flow_through_the_kernel_the_source_and_lambda() -> None:
    mu = lebesgue([(0.0, 1.0)], 24)
    gain = torch.tensor(1.3, requires_grad=True)
    offset = torch.tensor(0.4, requires_grad=True)
    lam = torch.tensor(0.5, requires_grad=True)

    u = T.nystrom_solve(
        lambda p, q: gain * p[:, 0:1] * q[:, 0].unsqueeze(0),
        lambda p: offset + torch.ones(p.shape[0], dtype=p.dtype),
        mu,
        lam=lam,
    )
    u.sum().backward()
    for name, g in (("gain", gain.grad), ("offset", offset.grad), ("lam", lam.grad)):
        assert g is not None and torch.isfinite(g).all() and abs(float(g)) > 0.0, name


def test_the_gradient_is_right_not_merely_finite() -> None:
    """Checked against a central difference on the analytic solution.

    ``d/dlam`` of ``int_0^1 u`` for ``u = 1 + c(lam) x`` with
    ``c = lam/2 / (1 - lam/3)``, which is what makes this a real gradient test
    rather than a smoke test.
    """
    mu = lebesgue([(0.0, 1.0)], 32)

    def total(lam_value: float) -> float:
        u = C.nystrom_solve(k_np, f_np, mu, lam=lam_value)
        return float(mu.weights @ u)

    lam0, h = 0.5, 1e-6
    finite = (total(lam0 + h) - total(lam0 - h)) / (2.0 * h)

    lam = torch.tensor(lam0, requires_grad=True)
    weights = torch.as_tensor(mu.weights)
    u = T.nystrom_solve(k_torch, f_torch, mu, lam=lam)
    (weights @ u).backward()
    assert float(lam.grad) == pytest.approx(finite, rel=1e-6)


def test_learnable_quadrature_weights_receive_a_gradient() -> None:
    """Weights may be tensors, so the measure itself can be trained."""
    mu = lebesgue([(0.0, 1.0)], 16)
    weights = torch.as_tensor(mu.weights).clone().requires_grad_(True)
    u = T.nystrom_solve(
        k_torch, f_torch, nodes=torch.as_tensor(mu.nodes), weights=weights, lam=0.7
    )
    u.sum().backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    assert float(weights.grad.abs().max()) > 0.0


def test_jax_grad_matches_torch_grad() -> None:
    mu = lebesgue([(0.0, 1.0)], 24)
    weights_np = mu.weights

    def jax_total(lam):
        u = J.nystrom_solve(k_jax, f_jax, mu, lam=lam)
        return jnp.sum(jnp.asarray(weights_np) * u)

    jx = float(jax.grad(jax_total)(0.5))
    lam = torch.tensor(0.5, requires_grad=True)
    u = T.nystrom_solve(k_torch, f_torch, mu, lam=lam)
    (torch.as_tensor(weights_np) @ u).backward()
    assert jx == pytest.approx(float(lam.grad), rel=1e-10)


# --------------------------------------------------------------------------- #
# error paths
# --------------------------------------------------------------------------- #
def test_error_paths() -> None:
    mu = lebesgue([(0.0, 1.0)], 8)
    with pytest.raises(ValueError, match="expected \\(8, 8\\)"):
        C.nystrom_solve(lambda p, q: np.ones((3, 3)), f_np, mu)
    with pytest.raises(ValueError, match="one value per quadrature node"):
        C.nystrom_solve(k_np, lambda p: np.ones(3), mu)
    with pytest.raises(ValueError, match="at least one .a, b. factor"):
        C.degenerate_kernel_solve([], f_np, mu)
    with pytest.raises(ValueError, match="max_terms must be >= 1"):
        C.neumann_series(k_np, f_np, mu, max_terms=0)
    with pytest.raises(ValueError, match="one value per quadrature node"):
        C.fredholm_residual(np.ones(3), k_np, f_np, mu)
    with pytest.raises(ValueError, match="either `measure` or explicit"):
        T.nystrom_solve(k_torch, f_torch)


def test_a_precomputed_source_vector_is_accepted() -> None:
    """So a network's output can be fed straight in without a wrapper closure."""
    mu = lebesgue([(0.0, 1.0)], 16)
    values = np.ones(16)
    assert np.allclose(
        C.nystrom_solve(k_np, values, mu), C.nystrom_solve(k_np, f_np, mu)
    )
    tensor = torch.ones(16, dtype=torch.float64)
    assert torch.allclose(
        T.nystrom_solve(k_torch, tensor, mu), T.nystrom_solve(k_torch, f_torch, mu)
    )
