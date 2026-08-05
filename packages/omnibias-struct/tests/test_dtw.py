# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soft Dynamic Time Warping: recursion == brute force, marginals == autodiff, certified gap.

The recursive softmin soft-DTW must equal the flat softmin over every monotonic path
(``lse_beta`` distributes over additive path costs), sandwich hard DTW by the closed-form
``log(N)/beta`` gap (min sense), expose closed-form alignment marginals that equal autodiff,
and be bit-identical across the PyTorch and JAX twins.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    DTWLattice,
    brute_force_dtw,
    brute_force_soft_dtw,
    certify_soft_dp,
    hard_dtw,
)
from omnibias.struct.jax import dtw as jdtw  # noqa: E402
from omnibias.struct.torch import dtw as tdtw  # noqa: E402

torch.set_default_dtype(torch.float64)


def _cost(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    return np.abs(rng.standard_normal((n, m))) + 0.1


def test_hard_dtw_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    for n, m in ((3, 3), (4, 2), (2, 5), (4, 4)):
        cost = _cost(rng, n, m)
        assert abs(hard_dtw(cost) - brute_force_dtw(cost)) < 1e-9


def test_recursive_soft_dtw_equals_flat_softmin_over_paths() -> None:
    rng = np.random.default_rng(1)
    cost = _cost(rng, 4, 3)
    ct = torch.tensor(cost)
    for beta in (0.5, 1.0, 4.0, 16.0):
        recursive = float(tdtw.soft_dtw(ct, beta))
        brute = brute_force_soft_dtw(cost, beta)
        assert abs(recursive - brute) < 1e-9


def test_soft_dtw_sandwiches_hard_dtw_with_log_n_gap() -> None:
    rng = np.random.default_rng(2)
    cost = _cost(rng, 4, 4)
    ct = torch.tensor(cost)
    hard = hard_dtw(cost)
    num_paths = DTWLattice(4, 4).count_paths()
    prev_gap = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0):
        soft = float(tdtw.soft_dtw(ct, beta))
        cert = certify_soft_dp(hard, soft, num_paths, beta, sense="min", brute_force_value=brute_force_dtw(cost))
        assert cert.is_sound  # V* - log(N)/beta <= V_beta <= V*
        assert cert.agrees_with_bruteforce
        assert soft <= hard + 1e-9  # softmin never exceeds the hard min
        assert cert.absolute_gap <= prev_gap + 1e-12  # gap shrinks with beta
        prev_gap = cert.absolute_gap


def test_soft_dtw_marginals_equal_autograd_torch() -> None:
    rng = np.random.default_rng(3)
    cost = _cost(rng, 4, 3)
    ct = torch.tensor(cost, requires_grad=True)
    tdtw.soft_dtw(ct, 3.0).backward()
    closed = tdtw.soft_dtw_marginals(torch.tensor(cost), 3.0)
    assert torch.max(torch.abs(ct.grad - closed)).item() < 1e-9


def test_soft_dtw_marginals_equal_grad_jax() -> None:
    rng = np.random.default_rng(4)
    cost = _cost(rng, 3, 4)
    cj = jnp.asarray(cost)
    grad = jax.grad(lambda c: jdtw.soft_dtw(c, 3.0))(cj)
    closed = jdtw.soft_dtw_marginals(cj, 3.0)
    assert float(jnp.max(jnp.abs(grad - closed))) < 1e-9


def test_soft_dtw_source_and_sink_marginals_are_one() -> None:
    rng = np.random.default_rng(5)
    cost = _cost(rng, 4, 4)
    e = tdtw.soft_dtw_marginals(torch.tensor(cost), 2.0)
    assert abs(float(e[0, 0]) - 1.0) < 1e-9
    assert abs(float(e[-1, -1]) - 1.0) < 1e-9


def test_soft_dtw_torch_jax_parity() -> None:
    rng = np.random.default_rng(6)
    cost = _cost(rng, 4, 3)
    ct, cj = torch.tensor(cost), jnp.asarray(cost)
    for beta in (1.0, 8.0):
        v_t = float(tdtw.soft_dtw(ct, beta))
        v_j = float(jdtw.soft_dtw(cj, beta))
        assert abs(v_t - v_j) < 1e-9
        m_t = tdtw.soft_dtw_marginals(ct, beta).numpy()
        m_j = np.asarray(jdtw.soft_dtw_marginals(cj, beta))
        assert np.max(np.abs(m_t - m_j)) < 1e-9


def test_soft_dtw_batched_matches_loop() -> None:
    rng = np.random.default_rng(7)
    costs = np.stack([_cost(rng, 3, 4) for _ in range(5)])
    ct, cj = torch.tensor(costs), jnp.asarray(costs)
    beta = 4.0
    bt = tdtw.soft_dtw_batched(ct, beta).numpy()
    bj = np.asarray(jdtw.soft_dtw_batched(cj, beta))
    loop = np.array([float(tdtw.soft_dtw(torch.tensor(c), beta)) for c in costs])
    assert np.max(np.abs(bt - loop)) < 1e-9
    assert np.max(np.abs(bj - loop)) < 1e-9
