# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multi-terminal / piecewise-analytic closed-form fractional derivative.

Validates the ``piecewise_fractional_derivative`` orchestration:

* it reduces to the single-terminal op on one patch and per-patch equals the local
  single-terminal call (matched terminals);
* the short-memory restart matches a Grunwald-Letnikov derivative restarted at the
  same patch terminal;
* smooth blending gives a continuous stitched field where hard selection jumps;
* torch/jax parity and autograd in ``alpha`` and the jets.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fractional.jax.ops import analytic as ja
from omnibias.fractional.torch import ops as tfr
from omnibias.fractional.torch.ops import analytic as ta

F = torch.float64


def _global_poly(x: np.ndarray, coeffs: list[float]) -> np.ndarray:
    return sum(c * x**k for k, c in enumerate(coeffs))


def _local_jet(coeffs: list[float], a: float, order: int) -> list[float]:
    """Re-expand the global polynomial ``sum_k coeffs[k] x^k`` about ``a`` (exact)."""
    deg = len(coeffs) - 1
    jet = []
    for j in range(order + 1):
        aj = sum(
            coeffs[k] * math.comb(k, j) * a ** (k - j) for k in range(j, deg + 1)
        )
        jet.append(float(aj))
    return jet


# --------------------------------------------------------------------------- #
# Reduction / consistency with the single-terminal operator.
# --------------------------------------------------------------------------- #
def test_single_patch_equals_single_terminal() -> None:
    jet = torch.tensor([[1.0, 2.0, 0.5, -0.3]], dtype=F)  # (M=1, N+1)
    x = torch.linspace(0.2, 1.8, 20, dtype=F)
    got = ta.piecewise_fractional_derivative(jet, [0.0], x, alpha=0.5)
    ref = ta.fractional_derivative(jet[0], x, alpha=0.5, a=0.0)
    assert torch.allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_per_patch_equals_local_single_terminal() -> None:
    coeffs = [1.0, 0.5, 0.3, -0.2]
    order = len(coeffs) - 1
    terminals = [0.0, 1.0, 2.0]
    jets = torch.tensor(
        [_local_jet(coeffs, a, order) for a in terminals], dtype=F
    )  # (3, order+1)
    # points strictly inside each patch (away from terminals so the gap-clamp is inert)
    x = torch.tensor([0.3, 0.7, 1.2, 1.8, 2.4, 2.9], dtype=F)
    got = ta.piecewise_fractional_derivative(jets, terminals, x, alpha=0.6)
    for i, ai in enumerate(terminals):
        lo = ai
        hi = terminals[i + 1] if i + 1 < len(terminals) else float("inf")
        mask = (x >= lo) & (x < hi)
        ref = ta.fractional_derivative(jets[i], x[mask], alpha=0.6, a=ai)
        assert torch.allclose(got[mask], ref, rtol=1e-11, atol=1e-11)


# --------------------------------------------------------------------------- #
# vs Grunwald-Letnikov (grid) -- restart semantics.
# --------------------------------------------------------------------------- #
def test_first_patch_matches_grunwald_letnikov() -> None:
    alpha = 0.5
    coeffs = [1.0, 2.0, 0.5]
    n = 4000
    xs = np.linspace(0.0, 2.0, n)
    h = float(xs[1] - xs[0])
    f = torch.as_tensor(_global_poly(xs, coeffs), dtype=F)
    grid = tfr.grunwald_letnikov(f, alpha=alpha, h=h).detach().numpy()

    jets = torch.tensor([_local_jet(coeffs, 0.0, 2)], dtype=F)
    closed = ta.piecewise_fractional_derivative(
        jets, [0.0], torch.as_tensor(xs, dtype=F), alpha=alpha
    ).numpy()
    sl = slice(n // 4, 3 * n // 4)
    rel = np.abs(grid[sl] - closed[sl]) / (np.abs(closed[sl]) + 1e-9)
    assert np.max(rel) < 2e-2


def test_second_patch_matches_restarted_grunwald_letnikov() -> None:
    # On patch 1 the operator restarts the lower terminal at a_1 (short memory);
    # a GL run on a grid that also starts at a_1 is the matching discretisation.
    alpha = 0.5
    coeffs = [1.0, 0.5, 0.3]
    a1 = 1.0
    n = 4000
    xs = np.linspace(a1, a1 + 1.5, n)
    h = float(xs[1] - xs[0])
    f = torch.as_tensor(_global_poly(xs, coeffs), dtype=F)
    gl = tfr.grunwald_letnikov(f, alpha=alpha, h=h).detach().numpy()

    jets = torch.tensor(
        [_local_jet(coeffs, 0.0, 2), _local_jet(coeffs, a1, 2)], dtype=F
    )
    closed = ta.piecewise_fractional_derivative(
        jets, [0.0, a1], torch.as_tensor(xs, dtype=F), alpha=alpha
    ).numpy()
    sl = slice(n // 4, 3 * n // 4)
    rel = np.abs(gl[sl] - closed[sl]) / (np.abs(closed[sl]) + 1e-9)
    assert np.max(rel) < 2e-2


# --------------------------------------------------------------------------- #
# Continuity: blend smooths the restart jump.
# --------------------------------------------------------------------------- #
def test_blend_makes_field_continuous_at_boundary() -> None:
    # Continuity is a Caputo property: Caputo is regular at the (restart) terminal,
    # so the restarted patch is finite there and a smooth blend removes the jump.
    # (Riemann-Liouville is singular at each terminal, so an RL restart cannot be
    # made continuous -- that is honest fractional-calculus behaviour, not a bug.)
    coeffs = [1.0, 0.8, 0.4]
    terminals = [0.0, 1.0]
    jets = torch.tensor([_local_jet(coeffs, a, 2) for a in terminals], dtype=F)
    b = 1.0
    eps = 1e-3
    xq = torch.tensor([b - eps, b + eps], dtype=F)

    hard = ta.piecewise_fractional_derivative(jets, terminals, xq, alpha=0.5, kind="caputo")
    jump = float((hard[1] - hard[0]).abs())
    assert jump > 1e-3  # hard selection genuinely jumps at the restart

    blended = ta.piecewise_fractional_derivative(
        jets, terminals, xq, alpha=0.5, kind="caputo", blend=0.05
    )
    assert float((blended[1] - blended[0]).abs()) < jump  # blend narrows the jump

    # a fine sweep through the boundary is smooth (no big finite-difference spike)
    xs = torch.linspace(0.6, 1.4, 400, dtype=F)
    field = ta.piecewise_fractional_derivative(
        jets, terminals, xs, alpha=0.5, kind="caputo", blend=0.05
    )
    d = (field[1:] - field[:-1]).abs()
    assert float(d.max()) < 0.2


# --------------------------------------------------------------------------- #
# Cross-backend parity + autograd.
# --------------------------------------------------------------------------- #
def test_torch_jax_parity_hard_and_blend() -> None:
    coeffs = [1.0, 0.5, 0.3, -0.1]
    terminals = [0.0, 0.8, 1.7]
    jets_np = np.array([_local_jet(coeffs, a, 3) for a in terminals])
    x_np = np.linspace(0.1, 2.4, 31)

    jt = torch.as_tensor(jets_np, dtype=F)
    xt = torch.as_tensor(x_np, dtype=F)
    jj = jnp.asarray(jets_np)
    xj = jnp.asarray(x_np)

    for blend in (0.0, 0.1):
        vt = ta.piecewise_fractional_derivative(jt, terminals, xt, alpha=0.6, blend=blend)
        vj = ja.piecewise_fractional_derivative(jj, terminals, xj, alpha=0.6, blend=blend)
        assert np.allclose(vt.numpy(), np.asarray(vj), rtol=1e-9, atol=1e-10), blend


def test_autograd_alpha_and_jets() -> None:
    coeffs = [1.0, 0.5, 0.3]
    terminals = [0.0, 1.0]
    jets = torch.tensor(
        [_local_jet(coeffs, a, 2) for a in terminals], dtype=F, requires_grad=True
    )
    x = torch.tensor([0.4, 1.3], dtype=F)
    alpha = torch.tensor(0.55, dtype=F, requires_grad=True)
    out = ta.piecewise_fractional_derivative(jets, terminals, x, alpha=alpha, blend=0.05)
    out.pow(2).sum().backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad)
    assert jets.grad is not None and torch.all(torch.isfinite(jets.grad))


# --------------------------------------------------------------------------- #
# Error paths.
# --------------------------------------------------------------------------- #
def test_non_increasing_terminals_raise() -> None:
    jets = torch.zeros((2, 3), dtype=F)
    with pytest.raises(ValueError, match="strictly increasing"):
        ta.piecewise_fractional_derivative(jets, [1.0, 0.0], torch.tensor([1.5], dtype=F), alpha=0.5)


def test_wrong_terminal_count_raises() -> None:
    jets = torch.zeros((2, 3), dtype=F)
    with pytest.raises(ValueError, match="one per jet"):
        ta.piecewise_fractional_derivative(jets, [0.0], torch.tensor([0.5], dtype=F), alpha=0.5)


def test_x_below_first_terminal_raises() -> None:
    jets = torch.zeros((2, 3), dtype=F)
    with pytest.raises(ValueError, match="first terminal"):
        ta.piecewise_fractional_derivative(
            jets, [0.0, 1.0], torch.tensor([-0.5, 0.5], dtype=F), alpha=0.5
        )
