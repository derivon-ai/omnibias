# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Generalized divergences + optimal-transport extensions (PyTorch).

Truths, the family limit relations (Renyi -> KL, entropy limits, f-divergence
specialisations), error guards, and autograd differentiability. Cross-backend
parity against JAX lives in the JAX test module.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.torch.information import (  # noqa: E402
    chi_squared_divergence,
    entropy,
    f_divergence,
    hellinger_distance,
    kl_divergence,
    renyi_divergence,
    renyi_entropy,
    sinkhorn_distance,
    sliced_wasserstein,
    total_variation_distance,
    tsallis_entropy,
    wasserstein1,
    wasserstein2_gaussian,
    wassersteinp,
)


@pytest.fixture(autouse=True)
def _use_float64() -> object:
    """Exercise the divergence oracles in double precision, test-locally.

    Scoped to this module so the float64 default never leaks into dtype-sensitive
    suites (e.g. ``test_fastpath_stability``) during collection.
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


# Module-level constants are built at import time (before the fixture runs), so
# pin their dtype explicitly rather than relying on the process default.
_P = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
_Q = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)


def test_total_variation_truth() -> None:
    assert float(total_variation_distance(_P, _Q)) == pytest.approx(
        0.5 * float((_P - _Q).abs().sum())
    )
    assert float(total_variation_distance(_P, _P)) == pytest.approx(0.0, abs=1e-15)


def test_hellinger_truth_and_bounds() -> None:
    bc = float((_P * _Q).sqrt().sum())
    assert float(hellinger_distance(_P, _Q)) == pytest.approx(math.sqrt(1.0 - bc), rel=1e-12)
    disjoint = float(hellinger_distance(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])))
    assert disjoint == pytest.approx(1.0, rel=1e-12)


def test_chi_squared_truth() -> None:
    truth = float((((_P - _Q) ** 2) / _Q).sum())
    assert float(chi_squared_divergence(_P, _Q)) == pytest.approx(truth, rel=1e-12)


def test_renyi_limit_is_kl_and_half_is_hellinger() -> None:
    assert float(renyi_divergence(_P, _Q, 1.0 + 1e-6)) == pytest.approx(
        float(kl_divergence(_P, _Q)), rel=1e-3
    )
    h2 = float(hellinger_distance(_P, _Q)) ** 2
    assert float(renyi_divergence(_P, _Q, 0.5)) == pytest.approx(
        -2.0 * math.log(1.0 - h2), rel=1e-12
    )


def test_entropy_limits() -> None:
    assert float(renyi_entropy(_P, 1.0 + 1e-6)) == pytest.approx(float(entropy(_P)), rel=1e-3)
    assert float(renyi_entropy(_P, 2.0)) == pytest.approx(-math.log(float((_P**2).sum())), rel=1e-12)
    assert float(tsallis_entropy(_P, 1.0 + 1e-6)) == pytest.approx(float(entropy(_P)), rel=1e-3)
    assert float(tsallis_entropy(_P, 2.0)) == pytest.approx(1.0 - float((_P**2).sum()), rel=1e-12)


def test_f_divergence_specialisations() -> None:
    assert float(f_divergence(_P, _Q, lambda t: t * torch.log(t))) == pytest.approx(
        float(kl_divergence(_P, _Q)), rel=1e-12
    )
    assert float(f_divergence(_P, _Q, lambda t: (t - 1.0) ** 2)) == pytest.approx(
        float(chi_squared_divergence(_P, _Q)), rel=1e-12
    )
    assert float(f_divergence(_P, _Q, lambda t: (t - 1.0).abs())) == pytest.approx(
        2.0 * float(total_variation_distance(_P, _Q)), rel=1e-12
    )


def test_renyi_rejects_alpha_one() -> None:
    with pytest.raises(ValueError, match="alpha != 1"):
        renyi_divergence(_P, _Q, 1.0)
    with pytest.raises(ValueError, match="order != 1"):
        tsallis_entropy(_P, 1.0)


# ----- optimal transport ----------------------------------------------------


def test_wassersteinp_matches_w1_and_rms() -> None:
    rng = np.random.default_rng(0)
    u = torch.tensor(rng.normal(size=40))
    v = torch.tensor(rng.normal(size=40))
    assert float(wassersteinp(u, v, p=1.0)) == pytest.approx(float(wasserstein1(u, v)), rel=1e-12)
    us = np.sort(u.numpy())
    vs = np.sort(v.numpy())
    assert float(wassersteinp(u, v, p=2.0)) == pytest.approx(
        math.sqrt(float(np.mean((us - vs) ** 2))), rel=1e-12
    )


def test_wasserstein2_gaussian() -> None:
    assert float(wasserstein2_gaussian(0.0, 1.0, 3.0, 2.0)) == pytest.approx(math.sqrt(10.0))


def test_sliced_identity_zero() -> None:
    rng = np.random.default_rng(1)
    X = torch.tensor(rng.normal(size=(25, 3)))
    dirs = rng.normal(size=(7, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    assert float(sliced_wasserstein(X, X, torch.tensor(dirs))) == pytest.approx(0.0, abs=1e-12)


def test_sinkhorn_matches_w1_for_1d_abs_cost() -> None:
    rng = np.random.default_rng(2)
    n = 8
    xs = np.sort(rng.normal(size=n))
    ys = np.sort(rng.normal(size=n))
    cost = torch.tensor(np.abs(xs[:, None] - ys[None, :]))
    a = torch.full((n,), 1.0 / n)
    w1 = float(wasserstein1(torch.tensor(xs), torch.tensor(ys)))
    fine = float(sinkhorn_distance(a, a, cost, epsilon=0.005, num_iters=2000))
    assert fine == pytest.approx(w1, abs=1e-3)


def test_ot_error_guards() -> None:
    with pytest.raises(ValueError, match="p >= 1"):
        wassersteinp(torch.zeros(3), torch.zeros(3), p=0.5)
    with pytest.raises(ValueError, match="cost must have shape"):
        sinkhorn_distance(torch.full((3,), 1 / 3), torch.full((2,), 0.5), torch.zeros((3, 3)))


# ----- autograd -------------------------------------------------------------


def test_operators_are_autograd_differentiable() -> None:
    p = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)
    renyi_divergence(p, _Q, 0.5).backward()
    assert p.grad is not None and bool(torch.isfinite(p.grad).all())

    loc = torch.tensor(0.7, requires_grad=True)
    wasserstein2_gaussian(loc, 1.0, 0.0, 1.0).backward()
    assert loc.grad is not None and float(loc.grad.abs()) > 0.0

    cost = torch.tensor(np.random.default_rng(0).normal(size=(4, 4)) ** 2, requires_grad=True)
    a = torch.full((4,), 0.25)
    sinkhorn_distance(a, a, cost, epsilon=0.2, num_iters=100).backward()
    assert cost.grad is not None and bool(torch.isfinite(cost.grad).all())
