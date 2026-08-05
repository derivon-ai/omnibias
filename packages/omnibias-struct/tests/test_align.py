# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Needleman-Wunsch alignment on the shared DAG substrate: oracle, gap, marginals, parity.

Global alignment is a longest-path DAG, so the soft aligner reuses ``soft_shortest_path``.
The hard NW DP must equal brute-force enumeration; the soft score must anneal to it with the
closed-form ``log(N)/beta`` gap; the closed-form parameter gradients (substitution / gap
usage) must equal autodiff; and the PyTorch / JAX twins must be bit-identical.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    AlignmentLattice,
    brute_force_align,
    brute_force_soft_align,
    certify_soft_dp,
    hard_align,
)
from omnibias.struct.jax import align as jalign  # noqa: E402
from omnibias.struct.torch import align as talign  # noqa: E402

torch.set_default_dtype(torch.float64)


def _blosum_like(k: int, rng: np.random.Generator) -> np.ndarray:
    s = rng.standard_normal((k, k))
    s = 0.5 * (s + s.T)
    s[np.arange(k), np.arange(k)] += 2.0  # reward matches on the diagonal
    return s


def test_hard_align_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    sub = _blosum_like(4, rng)
    gap = -1.0
    for _ in range(5):
        a = rng.integers(0, 4, size=int(rng.integers(2, 5)))
        b = rng.integers(0, 4, size=int(rng.integers(2, 5)))
        assert abs(hard_align(a, b, sub, gap) - brute_force_align(a, b, sub, gap)) < 1e-9


def test_soft_align_matches_flat_softmax_and_anneals() -> None:
    rng = np.random.default_rng(1)
    sub = _blosum_like(4, rng)
    gap = -1.0
    a = np.array([0, 1, 2, 1])
    b = np.array([0, 2, 1])
    st, gt = torch.tensor(sub), torch.tensor(gap)
    hard = hard_align(a, b, sub, gap)
    for beta in (0.5, 1.0, 4.0, 16.0):
        soft = float(talign.soft_align(a, b, st, gt, beta))
        assert abs(soft - brute_force_soft_align(a, b, sub, gap, beta)) < 1e-9
        assert soft >= hard - 1e-9  # softmax >= max


def test_soft_align_certified_gap() -> None:
    rng = np.random.default_rng(2)
    sub = _blosum_like(4, rng)
    gap = -1.0
    a = np.array([0, 1, 2, 3])
    b = np.array([0, 2, 3])
    st, gt = torch.tensor(sub), torch.tensor(gap)
    hard = hard_align(a, b, sub, gap)
    num_paths = AlignmentLattice(len(a), len(b)).build_dag()[0].count_paths()
    prev = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0):
        soft = float(talign.soft_align(a, b, st, gt, beta))
        cert = certify_soft_dp(hard, soft, num_paths, beta, brute_force_value=brute_force_align(a, b, sub, gap))
        assert cert.is_sound and cert.agrees_with_bruteforce
        assert cert.absolute_gap <= prev + 1e-12
        prev = cert.absolute_gap


def test_soft_align_marginals_equal_autograd_torch() -> None:
    rng = np.random.default_rng(3)
    sub = _blosum_like(4, rng)
    a = np.array([0, 1, 2, 1])
    b = np.array([0, 2, 1])
    beta = 3.0
    st = torch.tensor(sub, requires_grad=True)
    gt = torch.tensor(-1.0, requires_grad=True)
    talign.soft_align(a, b, st, gt, beta).backward()
    grad_sub, grad_gap = talign.soft_align_marginals(a, b, st.detach(), gt.detach(), beta)
    assert torch.max(torch.abs(st.grad - grad_sub)).item() < 1e-9
    assert abs(float(gt.grad) - float(grad_gap)) < 1e-9


def test_soft_align_marginals_equal_grad_jax() -> None:
    rng = np.random.default_rng(4)
    sub = _blosum_like(4, rng)
    a = np.array([0, 1, 2, 1])
    b = np.array([0, 2, 1])
    beta = 3.0
    sj, gj = jnp.asarray(sub), jnp.asarray(-1.0)
    g_sub = jax.grad(lambda s: jalign.soft_align(a, b, s, gj, beta))(sj)
    g_gap = jax.grad(lambda g: jalign.soft_align(a, b, sj, g, beta))(gj)
    grad_sub, grad_gap = jalign.soft_align_marginals(a, b, sj, gj, beta)
    assert float(jnp.max(jnp.abs(g_sub - grad_sub))) < 1e-9
    assert abs(float(g_gap) - float(grad_gap)) < 1e-9


def test_soft_align_torch_jax_parity() -> None:
    rng = np.random.default_rng(5)
    sub = _blosum_like(5, rng)
    a = np.array([0, 3, 2, 1])
    b = np.array([0, 2, 4, 1])
    for beta in (1.0, 8.0):
        v_t = float(talign.soft_align(a, b, torch.tensor(sub), torch.tensor(-0.7), beta))
        v_j = float(jalign.soft_align(a, b, jnp.asarray(sub), jnp.asarray(-0.7), beta))
        assert abs(v_t - v_j) < 1e-9
