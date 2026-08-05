# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fredholm / Volterra PINN residuals (torch).

Two layers, mirroring the CCF tests. The ``*_residual_samples`` helpers are
checked against **analytic solutions**, which is where the mathematics is pinned
down; the state-level classes are then checked against an independent numpy
substitution on a real field, which is where the wiring is pinned down -- the
node re-evaluation, the weight convention, the causal pullback.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch import equations as teq
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.equations.integral import (
    fredholm_residual_samples,
    volterra_residual_samples,
)
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField

pytest.importorskip("omnibias.measure")
from omnibias.measure._core.measure import lebesgue  # noqa: E402

torch.set_default_dtype(torch.float64)


def _field(axes=("x",), time_axis=None, seed=0):
    torch.manual_seed(seed)
    return OneLayerVectorField(
        coordinate_spec=CoordinateSpec(
            axes=axes, periodicity=(False,) * len(axes), time_axis=time_axis
        ),
        components=ComponentSpec(names=("u",), groups={}),
        hidden=6,
        base="tanh",
    )


# --------------------------------------------------------------------------- #
# the mathematics: analytic solutions must zero the sample-level residual
# --------------------------------------------------------------------------- #
def test_the_exact_solution_zeroes_the_fredholm_residual() -> None:
    """``u = 1 + lam int_0^1 x t u(t) dt`` has the closed form ``1 + c x``.

    Separability collapses it: ``c = lam int_0^1 t (1 + c t) dt`` gives
    ``c = (lam/2)/(1 - lam/3)``. Feeding that exact ``u`` in must return zero --
    anything else means the contraction is not the operator claimed.
    """
    lam = 0.7
    c = (lam / 2.0) / (1.0 - lam / 3.0)
    mu = lebesgue([(0.0, 1.0)], 24)
    nodes = torch.as_tensor(mu.nodes)
    weights = torch.as_tensor(mu.weights)
    x = torch.linspace(0.0, 1.0, 17)

    residual, _ = fredholm_residual_samples(
        1.0 + c * x,
        1.0 + c * nodes[:, 0],
        x[:, None] * nodes[:, 0][None, :],
        weights,
        lam=lam,
        source=torch.ones_like(x),
    )
    assert float(residual.abs().max()) < 1e-13


def test_a_wrong_solution_does_not_zero_the_fredholm_residual() -> None:
    """The companion half: a residual zero for everything would prove nothing."""
    lam = 0.7
    mu = lebesgue([(0.0, 1.0)], 24)
    nodes = torch.as_tensor(mu.nodes)
    x = torch.linspace(0.0, 1.0, 17)
    residual, _ = fredholm_residual_samples(
        1.0 + 0.9 * x,
        1.0 + 0.9 * nodes[:, 0],
        x[:, None] * nodes[:, 0][None, :],
        torch.as_tensor(mu.weights),
        lam=lam,
        source=torch.ones_like(x),
    )
    assert float(residual.abs().max()) > 0.1


def test_the_sample_residual_agrees_with_the_measure_solver() -> None:
    """Cross-checks against ``measure``'s Nystrom solution on its own nodes.

    Evaluated at the quadrature nodes the PINN residual and the solved discrete
    system are the same object, so they must agree to round-off. That pins the
    sign, the scaling and the weight convention all at once, against code that
    was written and tested independently.
    """
    from omnibias.measure._core import integraleq as C

    mu = lebesgue([(0.0, 1.0)], 20)
    lam = 0.6

    def k_np(x, t):
        return np.exp(-np.abs(x[:, :1] - t[:, 0][None, :]))

    u_nodes = C.nystrom_solve(k_np, lambda p: np.cos(p[:, 0]), mu, lam=lam)
    nodes = torch.as_tensor(mu.nodes)
    u = torch.as_tensor(u_nodes)
    residual, _ = fredholm_residual_samples(
        u,
        u,
        torch.exp(-(nodes[:, :1] - nodes[:, 0][None, :]).abs()),
        torch.as_tensor(mu.weights),
        lam=lam,
        source=torch.cos(nodes[:, 0]),
    )
    assert float(residual.abs().max()) < 1e-12


def test_the_exact_solution_zeroes_the_volterra_residual() -> None:
    """``u(x) = 1 + int_0^x u(t) dt`` has ``u = e^x``.

    The simplest hereditary law there is, and the one that catches an
    off-by-a-factor in the pullback: ``[0, x]`` has length ``x``, so dropping the
    ``span`` factor still passes at ``x = 1`` and fails everywhere else.
    """
    mu = lebesgue([(0.0, 1.0)], 20)
    s = torch.as_tensor(mu.nodes)[:, 0]
    weights = torch.as_tensor(mu.weights)
    x = torch.linspace(0.05, 1.0, 13)
    t = x[:, None] * s[None, :]

    residual, integral = volterra_residual_samples(
        torch.exp(x),
        torch.exp(t),
        torch.ones_like(t),
        weights,
        x,
        lam=1.0,
        source=torch.ones_like(x),
    )
    assert float(residual.abs().max()) < 1e-12
    assert torch.allclose(integral, torch.exp(x) - 1.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# the wiring: the state-level classes on a real field
# --------------------------------------------------------------------------- #
def test_the_fredholm_class_matches_a_numpy_substitution() -> None:
    """The class must re-evaluate the field at the nodes and contract correctly."""
    field = _field()
    mu = lebesgue([(0.0, 1.0)], 16)
    coords = torch.linspace(0.0, 1.0, 11).reshape(-1, 1)
    nodes = torch.as_tensor(mu.nodes)
    lam = 0.5

    out = teq.fredholm(
        field(coords),
        kernel=lambda x, t: torch.exp(-(x[:, :1] - t[:, 0][None, :]) ** 2),
        measure=mu,
        lam=lam,
        source=lambda state: torch.cos(state.coords[:, 0]),
    )

    u = tops.value(field(coords), "u").detach().numpy()
    u_nodes = tops.value(field(nodes), "u").detach().numpy()
    k = np.exp(-((coords.numpy()[:, :1] - nodes.numpy()[:, 0][None, :]) ** 2))
    expected = u - lam * (k * mu.weights[None, :]) @ u_nodes - np.cos(coords.numpy()[:, 0])
    np.testing.assert_allclose(out.residual.detach().numpy(), expected, atol=1e-12)


def test_the_nonlocal_term_is_returned_for_inspection() -> None:
    """``integral`` is the expensive, quadrature-approximated half; expose it."""
    field = _field()
    mu = lebesgue([(0.0, 1.0)], 16)
    coords = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)
    out = teq.fredholm(
        field(coords),
        kernel=lambda x, t: torch.ones(x.shape[0], t.shape[0], dtype=x.dtype),
        measure=mu,
        lam=1.0,
    )
    # A constant kernel makes the integral the same number at every point.
    integral = out.integral.detach()
    assert torch.allclose(integral, integral[0].expand(5), atol=1e-13)
    u_nodes = tops.value(field(torch.as_tensor(mu.nodes)), "u").detach().numpy()
    assert float(integral[0]) == pytest.approx(float(mu.weights @ u_nodes), abs=1e-13)


def test_the_volterra_class_is_causal() -> None:
    """Nothing ahead of ``x`` may enter: the term must vanish as ``x -> a``.

    A non-causal implementation (integrating the whole domain regardless) returns
    the same value at every point, so a constant field separates the two in one
    line: the causal integral of ``1`` is ``x``.
    """
    field = _field()
    coords = torch.tensor([[0.0], [0.25], [0.5], [1.0]])
    out = teq.volterra(
        field(coords),
        kernel=lambda x, t: torch.ones(x.shape[:2], dtype=x.dtype),
        measure=lebesgue([(0.0, 1.0)], 16),
        lam=1.0,
    )
    integral = out.integral.detach()
    assert float(integral[0]) == pytest.approx(0.0, abs=1e-14)
    assert (integral[1:] > 0).all()

    # int_0^x u for a smooth u, checked against a dense independent quadrature.
    for i, x in enumerate(coords[:, 0].tolist()):
        dense = torch.linspace(0.0, x, 2001).reshape(-1, 1)
        u_dense = tops.value(field(dense), "u").detach()
        expected = torch.trapz(u_dense, dense[:, 0]) if x > 0 else torch.tensor(0.0)
        assert float(integral[i]) == pytest.approx(float(expected), abs=1e-7)


def test_a_shifted_origin_shifts_the_lower_limit() -> None:
    field = _field()
    coords = torch.tensor([[0.5], [1.0], [2.0]])
    out = teq.volterra(
        field(coords),
        kernel=lambda x, t: torch.ones(x.shape[:2], dtype=x.dtype),
        measure=lebesgue([(0.0, 1.0)], 24),
        origin=0.5,
        lam=1.0,
    )
    integral = out.integral.detach()
    assert float(integral[0]) == pytest.approx(0.0, abs=1e-14)
    for i, x in enumerate(coords[:, 0].tolist()):
        if x <= 0.5:
            continue
        dense = torch.linspace(0.5, x, 2001).reshape(-1, 1)
        u_dense = tops.value(field(dense), "u").detach()
        expected = float(torch.trapz(u_dense, dense[:, 0]))
        assert float(integral[i]) == pytest.approx(expected, abs=1e-7)


def test_the_pullback_converges_at_the_reference_rule_order() -> None:
    """Gauss-Legendre on the reference interval, so a smooth field is spectral.

    The module docstring claims the pullback buys the *rule's* convergence order
    rather than the second order a fixed cumulative-trapezoid grid would give.
    Eight nodes reaching round-off is the observable difference: a second-order
    rule cannot.
    """
    field = _field()
    coords = torch.linspace(0.1, 1.0, 7).reshape(-1, 1)
    reference = teq.volterra(
        field(coords),
        kernel=lambda x, t: torch.ones(x.shape[:2], dtype=x.dtype),
        measure=lebesgue([(0.0, 1.0)], 64),
        lam=1.0,
    ).integral.detach()

    errors = {}
    for n in (4, 8):
        got = teq.volterra(
            field(coords),
            kernel=lambda x, t: torch.ones(x.shape[:2], dtype=x.dtype),
            measure=lebesgue([(0.0, 1.0)], n),
            lam=1.0,
        ).integral.detach()
        errors[n] = float((got - reference).abs().max())
    assert errors[8] < 1e-12
    assert errors[8] < errors[4]


def test_the_causal_axis_is_chosen_and_the_others_are_frozen() -> None:
    """In space-time the causal term is a memory at fixed ``x``.

    ``int_0^t u(x, s) ds`` is only right if the spatial coordinate is held at the
    collocation point's own value while the time coordinate sweeps. Freezing the
    wrong axis, or none, moves the answer.
    """
    field = _field(axes=("x", "t"), time_axis="t")
    coords = torch.tensor([[2.0, 0.5], [3.0, 1.0], [0.5, 2.0]])
    out = teq.volterra(
        field(coords),
        kernel=lambda x, t: torch.ones(x.shape[:2], dtype=x.dtype),
        measure=lebesgue([(0.0, 1.0)], 32),
        lam=1.0,
    )
    integral = out.integral.detach()
    for i, (x, t) in enumerate(coords.tolist()):
        dense_t = torch.linspace(0.0, t, 2001)
        dense = torch.stack([torch.full_like(dense_t, x), dense_t], dim=1)
        u_dense = tops.value(field(dense), "u").detach()
        expected = float(torch.trapz(u_dense, dense_t))
        assert float(integral[i]) == pytest.approx(expected, abs=1e-6)


def test_an_ambiguous_causal_axis_must_be_named() -> None:
    field = _field(axes=("x", "y"), time_axis=None)
    coords = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="which axis is causal"):
        teq.volterra(
            field(coords),
            kernel=lambda x, t: torch.ones(x.shape[:2], dtype=x.dtype),
            measure=lebesgue([(0.0, 1.0)], 8),
        )


# --------------------------------------------------------------------------- #
# differentiability -- the reason to write a residual instead of a solve
# --------------------------------------------------------------------------- #
def test_the_residual_carries_gradients_into_a_learned_kernel() -> None:
    """A learned kernel is the case the direct solver serves less well than this."""
    theta = torch.tensor(0.4, requires_grad=True)
    field = _field()
    coords = torch.linspace(0.0, 1.0, 9).reshape(-1, 1)
    out = teq.fredholm(
        field(coords),
        kernel=lambda x, t: theta * x[:, :1] * t[:, 0][None, :],
        measure=lebesgue([(0.0, 1.0)], 12),
        lam=0.5,
    )
    (out.residual**2).mean().backward()
    assert theta.grad is not None and float(theta.grad.abs()) > 0.0
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0.0
        for p in field.parameters()
    )


def test_the_volterra_residual_carries_gradients_through_the_pullback() -> None:
    """The moving domain must not detach the field parameters from the loss."""
    field = _field()
    coords = torch.linspace(0.1, 1.0, 7).reshape(-1, 1)
    out = teq.volterra(
        field(coords),
        kernel=lambda x, t: torch.exp(-(x - t).squeeze(-1)),
        measure=lebesgue([(0.0, 1.0)], 10),
        lam=0.5,
        source=lambda state: torch.ones(state.coords.shape[0]),
    )
    (out.residual**2).mean().backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0.0
        for p in field.parameters()
    )


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_a_mismatched_measure_dimension_is_refused() -> None:
    field = _field(axes=("x", "y"))
    coords = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="the measure lives in 1D"):
        teq.fredholm(
            field(coords),
            kernel=lambda x, t: torch.ones(x.shape[0], t.shape[0], dtype=x.dtype),
            measure=lebesgue([(0.0, 1.0)], 8),
        )


def test_a_volterra_reference_measure_must_be_one_dimensional() -> None:
    field = _field(axes=("x", "y"))
    coords = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="reference measure must be 1-D"):
        teq.volterra(
            field(coords),
            kernel=lambda x, t: torch.ones(x.shape[:2], dtype=x.dtype),
            measure=lebesgue([(0.0, 1.0), (0.0, 1.0)], 4),
            axis="x",
        )


def test_a_badly_shaped_kernel_is_refused() -> None:
    field = _field()
    coords = torch.linspace(0.0, 1.0, 5).reshape(-1, 1)
    with pytest.raises(ValueError, match="the kernel returned shape"):
        teq.fredholm(
            field(coords),
            kernel=lambda x, t: torch.ones(x.shape[0], dtype=x.dtype),
            measure=lebesgue([(0.0, 1.0)], 8),
        )
