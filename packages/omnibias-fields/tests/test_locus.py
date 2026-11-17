# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equality-locus tensor twins G5/G6 (theory 01-09)."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.core.locus import EqualitySystem, UnitTerm
from omnibias.core.locus import newton_project as newton_core
from omnibias.fields.locus.torch import newton_project, newton_project_unrolled


def _case_b() -> EqualitySystem:
    return EqualitySystem(
        (
            UnitTerm(1, 1.0, (1.0, 0.0), 0.0),
            UnitTerm(2, -2.0, (0.0, 1.0), 0.0),
        )
    )


def _isolated() -> EqualitySystem:
    # 2 tanh(x) = tanh(1), with the independent closed-form root below.
    return EqualitySystem((UnitTerm(0, 2.0, (1.0,)), UnitTerm(0, 1.0, (0.0,), 1.0)))


def _flat_locus() -> EqualitySystem:
    # F(x, y) = tanh(x). Every point (0, y) is a root.
    return EqualitySystem((UnitTerm(0, 1.0, (1.0, 0.0)), UnitTerm(0, 0.0, (0.0, 0.0))))


def test_isolated_ift_matches_analytic_and_parameter_finite_differences() -> None:
    sys = _isolated()
    x0 = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([2.0, 1.0], dtype=torch.float64, requires_grad=True)
    result = newton_project(sys, x0, weights=w, tol=1e-14)
    t = math.tanh(1.0)
    expected_root = math.atanh(t / 2.0)
    assert result.item() == pytest.approx(expected_root, abs=1e-14)
    result.sum().backward()
    denominator = 1.0 - (t / 2.0) ** 2
    expected_gradient = [-t / (4.0 * denominator), t / (2.0 * denominator)]
    assert w.grad.tolist() == pytest.approx(expected_gradient, abs=1e-12)
    assert x0.grad.tolist() == [0.0]
    eps = 1e-5
    finite_difference = []
    for index in range(2):
        delta = torch.zeros_like(w)
        delta[index] = eps
        plus = newton_project(sys, x0.detach(), weights=w.detach() + delta, tol=1e-14)
        minus = newton_project(sys, x0.detach(), weights=w.detach() - delta, tol=1e-14)
        finite_difference.append((plus.item() - minus.item()) / (2 * eps))
    assert w.grad.tolist() == pytest.approx(finite_difference, abs=1e-10)
    plus_seed = newton_project(sys, x0.detach() + eps, tol=1e-14)
    minus_seed = newton_project(sys, x0.detach() - eps, tol=1e-14)
    assert ((plus_seed - minus_seed) / (2 * eps)).item() == pytest.approx(0.0, abs=1e-10)


def test_underdetermined_ift_refuses_the_old_zero_tangent_gradient() -> None:
    sys = _flat_locus()
    x0 = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="solution-selection rule"):
        newton_project(sys, x0)
    # Forward landing is retained when an implicit derivative is not requested.
    with torch.no_grad():
        landed = newton_project(sys, x0)
    assert landed.tolist() == pytest.approx([0.0, 0.7], abs=1e-12)
    # The executed iteration preserves the tangent coordinate and its gradient.
    weights = torch.tensor([1.0, 0.0], dtype=torch.float64)
    iterated = newton_project_unrolled(sys, x0, weights=weights)
    iterated[1].backward()
    assert x0.grad.tolist() == pytest.approx([0.0, 1.0], abs=1e-12)


def test_implicit_derivative_refuses_nonconvergence_and_rank_loss() -> None:
    x0 = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="converged root"):
        newton_project(_isolated(), x0, max_iter=1, tol=1e-15)
    degenerate = EqualitySystem((UnitTerm(0, 1.0, (0.0,)), UnitTerm(0, 1.0, (0.0,))))
    with pytest.raises(ValueError, match="regular root"):
        newton_project(degenerate, x0)


def test_unrolled_nonregular_nonroot_refuses_infinite_step() -> None:
    sys = EqualitySystem((UnitTerm(0, 1.0, (0.0,), 1.0), UnitTerm(0, 1.0, (0.0,))))
    with pytest.raises(ValueError, match="regular Jacobian"):
        newton_project_unrolled(
            sys, torch.tensor([0.0], dtype=torch.float64), weights=torch.ones(2, dtype=torch.float64)
        )


@pytest.mark.parametrize("bad_seed", [[float("nan")], [float("inf")]])
def test_nonfinite_seed_refused(bad_seed: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        newton_project(_isolated(), torch.tensor(bad_seed, dtype=torch.float64))


def test_g6_torch_jax_newton_parity() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.fields.locus.jax import newton_project as newton_jax

    jax.config.update("jax_enable_x64", True)
    sys = _case_b()
    x0_t = torch.tensor([0.0, 0.20], dtype=torch.float64)
    x_t = newton_project(sys, x0_t)
    x_j = newton_jax(sys, jnp.asarray([0.0, 0.20], dtype=jnp.float64))
    assert x_t.detach().cpu().numpy() == pytest.approx(x_j.tolist(), rel=0, abs=0)
    core = newton_core(sys, (0.0, 0.20))
    assert x_t[1].item() == pytest.approx(core.point[1], rel=0, abs=0)
    assert math.isfinite(float(x_t[1].detach()))


def test_jax_manual_vjp_restricts_ift_and_matches_analytic() -> None:
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from omnibias.fields.locus.jax import newton_ift_vjp

    jax.config.update("jax_enable_x64", True)
    t = math.tanh(1.0)
    root = jnp.asarray([math.atanh(t / 2.0)], dtype=jnp.float64)
    weights = jnp.asarray([2.0, 1.0], dtype=jnp.float64)
    gradient = newton_ift_vjp(_isolated(), root, weights, jnp.ones(1, dtype=jnp.float64))
    denominator = 1.0 - (t / 2.0) ** 2
    assert gradient.tolist() == pytest.approx([-t / (4 * denominator), t / (2 * denominator)], abs=1e-12)
    with pytest.raises(ValueError, match="solution-selection rule"):
        newton_ift_vjp(_flat_locus(), jnp.asarray([0.0, 0.7]), weights, jnp.ones(2))
    with pytest.raises(ValueError, match="converged root"):
        newton_ift_vjp(_isolated(), jnp.asarray([0.0]), weights, jnp.ones(1))
    degenerate = EqualitySystem((UnitTerm(0, 1.0, (0.0,)), UnitTerm(0, 1.0, (0.0,))))
    with pytest.raises(ValueError, match="regular root"):
        newton_ift_vjp(degenerate, jnp.asarray([0.0]), jnp.ones(2), jnp.ones(1))
