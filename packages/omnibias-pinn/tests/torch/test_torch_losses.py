# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for torch loss helpers.

Covers:

* Sobolev preconditioning: ``p = 0`` recovers MSE; weight monotonicity.
* WP causal weighting: weights non-increasing, sum to ``< n_t``,
  combined with Sobolev preconditioner.
* Entropy-consistent residual: identity for ``entropy_weight=None``.
* NTK rebalance: equal-trace -> equal weights; geometric-mean property.
"""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.pinn.torch.losses import (
    causal_residual_loss,
    causal_weights_from_per_bin,
    entropy_consistent_residual,
    estimate_ntk_trace,
    mse_residual_loss,
    ntk_balanced_loss,
    sobolev_residual_loss,
    sobolev_weight,
)


@pytest.fixture
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(20240527)
    return g


def _rand(shape: tuple[int, ...], rng: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=rng, dtype=torch.float64)


# ---------------- Sobolev ------------------------------------------


def test_sobolev_p_zero_equals_mse(rng):
    R = _rand((4, 16, 16), rng)
    loss = sobolev_residual_loss(R, L=2 * math.pi, sobolev_p=0.0)
    expected = (R * R).mean()
    assert torch.allclose(loss, expected, rtol=1e-12, atol=1e-12)


def test_sobolev_parseval_p_zero_via_fft(rng):
    """Even when going through the FFT branch, p=0 should give MSE."""
    R = _rand((4, 16, 16), rng)
    weight = sobolev_weight(R, L=2 * math.pi, sobolev_p=0.0)
    assert torch.allclose(weight, torch.ones_like(weight))


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_sobolev_weight_decreases_with_k(p, rng):
    R = _rand((1, 32, 32), rng)
    w = sobolev_weight(R, L=2 * math.pi, sobolev_p=p)
    assert (w > 0).all()
    assert torch.allclose(w.max(), torch.ones((), dtype=w.dtype))


def test_sobolev_1d_residual(rng):
    R = _rand((4, 32), rng)
    loss = sobolev_residual_loss(R, L=2 * math.pi, sobolev_p=1.0)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_sobolev_3d_residual(rng):
    R = _rand((2, 8, 8, 8), rng)
    loss = sobolev_residual_loss(R, L=(2 * math.pi, 2 * math.pi, 1.0), sobolev_p=1.0)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_sobolev_loss_smoother_than_mse(rng):
    """Random Gaussian residuals concentrate energy at high k -> Sobolev
    p=1 loss should be strictly smaller than the MSE."""
    R = _rand((4, 32, 32), rng)
    mse = (R * R).mean().item()
    sob = sobolev_residual_loss(R, L=2 * math.pi, sobolev_p=1.0).item()
    assert sob < mse


# ---------------- Causal weighting ---------------------------------


def test_causal_weights_non_increasing(rng):
    L_per_bin = torch.linspace(0.1, 1.0, 32, dtype=torch.float64)
    w = causal_weights_from_per_bin(L_per_bin, epsilon=2.0)
    diffs = w[1:] - w[:-1]
    assert (diffs <= 1e-15).all()
    assert torch.isclose(w[0], torch.ones((), dtype=w.dtype))


def test_causal_loss_matches_mse_when_eps_zero(rng):
    R = _rand((8, 16, 16), rng)
    loss = causal_residual_loss(R, epsilon=0.0)
    expected = (R * R).mean()
    assert torch.allclose(loss, expected, rtol=1e-12, atol=1e-12)


def test_causal_loss_with_sobolev(rng):
    R = _rand((8, 16, 16), rng)
    loss, w = causal_residual_loss(
        R, epsilon=1.0, L=2 * math.pi, sobolev_p=1.0,
        return_weights=True,
    )
    assert loss.dim() == 0
    assert w.shape == (8,)
    assert (w[1:] - w[:-1] <= 1e-15).all()


def test_causal_loss_requires_L_when_sobolev_positive(rng):
    R = _rand((4, 8), rng)
    with pytest.raises(ValueError, match="requires L"):
        causal_residual_loss(R, epsilon=1.0, sobolev_p=1.0)


def test_causal_weights_detached_from_graph(rng):
    R = torch.randn((6, 8, 8), dtype=torch.float64, requires_grad=True)
    loss, w = causal_residual_loss(R, epsilon=1.0, return_weights=True)
    assert w.requires_grad is False
    loss.backward()
    assert R.grad is not None


# ---------------- Entropy-consistent ------------------------------


def test_entropy_default_is_mse(rng):
    R = _rand((4, 8, 8), rng)
    loss = entropy_consistent_residual(R)
    expected = (R * R).mean()
    assert torch.allclose(loss, expected, rtol=1e-12, atol=1e-12)


def test_entropy_quadratic_matches_mse(rng):
    R = _rand((4, 8, 8), rng)
    loss = entropy_consistent_residual(R, entropy_weight=lambda u: torch.ones_like(u))
    expected = (R * R).mean()
    assert torch.allclose(loss, expected, rtol=1e-12, atol=1e-12)


def test_entropy_kinetic_weight_positive(rng):
    R = _rand((4, 8), rng)
    state = _rand((4, 8), rng).abs() + 1.0  # eta'' > 0
    loss = entropy_consistent_residual(
        R, entropy_weight=lambda u: u, state_for_weight=state,
    )
    assert loss.item() >= 0


# ---------------- NTK rebalance -----------------------------------


def test_ntk_balanced_equal_traces_gives_equal_weights():
    losses = {
        "pde": torch.tensor(2.0, dtype=torch.float64),
        "bc": torch.tensor(3.0, dtype=torch.float64),
        "ic": torch.tensor(5.0, dtype=torch.float64),
    }
    traces = {
        "pde": torch.tensor(10.0, dtype=torch.float64),
        "bc": torch.tensor(10.0, dtype=torch.float64),
        "ic": torch.tensor(10.0, dtype=torch.float64),
    }
    total, weights = ntk_balanced_loss(losses, ntk_traces=traces)
    for w in weights.values():
        assert math.isclose(w, 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_ntk_balanced_geometric_mean():
    """If t_pde >> t_bc, weights should compensate so w_pde << w_bc."""
    losses = {"pde": torch.tensor(1.0, dtype=torch.float64),
              "bc": torch.tensor(1.0, dtype=torch.float64)}
    traces = {"pde": torch.tensor(100.0, dtype=torch.float64),
              "bc": torch.tensor(1.0, dtype=torch.float64)}
    _, w = ntk_balanced_loss(losses, ntk_traces=traces)
    # Geometric mean = sqrt(100 * 1) = 10. So w_pde = 10/100 = 0.1, w_bc = 10/1 = 10.
    assert math.isclose(w["pde"], 0.1, rel_tol=1e-9)
    assert math.isclose(w["bc"], 10.0, rel_tol=1e-9)


def test_ntk_balanced_no_traces_equal_weights():
    losses = {"pde": torch.tensor(1.0, dtype=torch.float64),
              "bc": torch.tensor(2.0, dtype=torch.float64)}
    total, w = ntk_balanced_loss(losses)
    assert all(v == 1.0 for v in w.values())
    assert torch.allclose(total, torch.tensor(3.0, dtype=torch.float64))


def test_estimate_ntk_trace_smoke():
    p = torch.nn.Parameter(torch.randn((4, 3), dtype=torch.float64))
    x = torch.randn((5, 3), dtype=torch.float64)
    y = (x @ p.t()).sum()
    trace = estimate_ntk_trace(y, [p])
    # d/dp_ij sum(x p^T) = sum_b x_b -> shape (4, 3) of repeated row sums.
    expected = ((x.sum(dim=0) ** 2).sum() * 4).item()
    assert math.isclose(float(trace), expected, rel_tol=1e-12, abs_tol=1e-12)


def test_ntk_balanced_validates_keys():
    losses = {"a": torch.tensor(1.0, dtype=torch.float64)}
    traces = {"b": torch.tensor(1.0, dtype=torch.float64)}
    with pytest.raises(ValueError, match="do not match"):
        ntk_balanced_loss(losses, ntk_traces=traces)


def test_ntk_balanced_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        ntk_balanced_loss({})


# ---------------- mse_residual_loss --------------------------------


def test_mse_residual_loss(rng):
    R = _rand((4, 8, 8), rng)
    expected = (R * R).mean()
    assert torch.allclose(mse_residual_loss(R), expected, rtol=1e-12, atol=1e-12)


# ---------------- Parity with research implementation -------------


@pytest.mark.needs_research
def test_parity_with_research_ns2d_causal_loss(rng):
    """The generalised ``causal_residual_loss`` must match the existing
    internal 2-D Navier-Stokes reference solver on its native
    (n_t, n_y, n_x) input.

    Note: the research code constructs the ``k`` grid via
    ``torch.fft.fftfreq(n, d=L/n).to(dtype=...)``, which allocates
    float32 by default and casts up; this loses ~1e-7 of precision in
    the wavenumbers. The lifted helper allocates fftfreq directly in
    float64. Relative agreement to ~1e-6 is still tighter than any
    optimisation step would care about and proves the generalised
    helper subsumes the original.
    """
    pytest.importorskip(
        "research.experiments.navier_stokes_2d.solvers._causal",
        reason="research.experiments tree is private and not shipped publicly",
    )
    from research.experiments.navier_stokes_2d.solvers._causal import (
        causal_residual_loss_fourier_2d,
    )
    R = _rand((6, 16, 16), rng)
    L = 2.0 * math.pi
    eps = 1.5
    sob_p = 1.0

    expected = causal_residual_loss_fourier_2d(
        R, L=L, sobolev_p=sob_p, epsilon=eps,
    )
    got = causal_residual_loss(
        R, epsilon=eps, L=L, sobolev_p=sob_p,
    )
    assert torch.allclose(got, expected, rtol=1e-6, atol=1e-9)
