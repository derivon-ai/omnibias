# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Smith-Waterman local + Gotoh affine-gap alignment on the shared shortest-path substrate.

Both reuse ``soft_shortest_path`` on an augmented DAG (free start/end 0-edges for local; a
3-state M/Ix/Iy lattice for affine gaps). The hard classic DP must equal brute-force
enumeration; the soft score must anneal to it with a sound ``log(N)/beta`` gap; the
closed-form parameter gradients (substitution / gap-open / gap-extend usage) must equal
autodiff; and the PyTorch / JAX twins must be bit-identical.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    brute_force_gotoh,
    brute_force_local_align,
    brute_force_soft_gotoh,
    brute_force_soft_local_align,
    build_gotoh_dag,
    build_local_dag,
    certify_soft_dp,
    hard_gotoh,
    hard_local_align,
)
from omnibias.struct.jax import align as jalign  # noqa: E402
from omnibias.struct.torch import align as talign  # noqa: E402

torch.set_default_dtype(torch.float64)

SEEDS = range(6)


def _sub(k: int, rng: np.random.Generator) -> np.ndarray:
    s = rng.standard_normal((k, k))
    s = 0.5 * (s + s.T)
    s[np.arange(k), np.arange(k)] += 2.0  # reward matches on the diagonal
    return s


def _seqs(rng: np.random.Generator, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    n = int(rng.integers(1, 5))
    m = int(rng.integers(1, 5))
    return rng.integers(0, k, size=n), rng.integers(0, k, size=m)


# ---------------------------------------------------------------------------
# Smith-Waterman local alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_local_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sub = _sub(4, rng)
    gap = -1.0
    a, b = _seqs(rng)
    assert abs(hard_local_align(a, b, sub, gap) - brute_force_local_align(a, b, sub, gap)) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_local_matches_brute_and_anneals(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sub = _sub(4, rng)
    gap = -1.0
    a, b = _seqs(rng)
    st, gt = torch.tensor(sub), torch.tensor(gap)
    hard = hard_local_align(a, b, sub, gap)
    for beta in (0.5, 1.0, 4.0, 16.0):
        soft = float(talign.soft_local_align(a, b, st, gt, beta))
        assert abs(soft - brute_force_soft_local_align(a, b, sub, gap, beta)) < 1e-9
        assert soft >= hard - 1e-9  # softmax >= max


def test_soft_local_certified_gap() -> None:
    rng = np.random.default_rng(2)
    sub = _sub(4, rng)
    gap = -1.0
    a = np.array([0, 1, 2, 3])
    b = np.array([0, 2, 3])
    st, gt = torch.tensor(sub), torch.tensor(gap)
    hard = hard_local_align(a, b, sub, gap)
    num_paths = build_local_dag(len(a), len(b))[0].count_paths()
    prev = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0):
        soft = float(talign.soft_local_align(a, b, st, gt, beta))
        cert = certify_soft_dp(
            hard, soft, num_paths, beta, brute_force_value=brute_force_local_align(a, b, sub, gap)
        )
        assert cert.is_sound and cert.agrees_with_bruteforce
        assert cert.absolute_gap <= prev + 1e-12
        prev = cert.absolute_gap


def test_soft_local_marginals_equal_autograd() -> None:
    rng = np.random.default_rng(3)
    sub = _sub(4, rng)
    a = np.array([0, 1, 2, 1])
    b = np.array([0, 2, 1])
    beta = 3.0
    # torch closed-form == autograd
    st = torch.tensor(sub, requires_grad=True)
    gt = torch.tensor(-1.0, requires_grad=True)
    talign.soft_local_align(a, b, st, gt, beta).backward()
    g_sub, g_gap = talign.soft_local_align_marginals(a, b, st.detach(), gt.detach(), beta)
    assert torch.max(torch.abs(st.grad - g_sub)).item() < 1e-9
    assert abs(float(gt.grad) - float(g_gap)) < 1e-9
    # jax closed-form == jax.grad
    sj, gj = jnp.asarray(sub), jnp.asarray(-1.0)
    jg_sub = jax.grad(lambda s: jalign.soft_local_align(a, b, s, gj, beta))(sj)
    jg_gap = jax.grad(lambda g: jalign.soft_local_align(a, b, sj, g, beta))(gj)
    cf_sub, cf_gap = jalign.soft_local_align_marginals(a, b, sj, gj, beta)
    assert float(jnp.max(jnp.abs(jg_sub - cf_sub))) < 1e-9
    assert abs(float(jg_gap) - float(cf_gap)) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_local_torch_jax_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sub = _sub(5, rng)
    a, b = _seqs(rng, 5)
    for beta in (1.0, 8.0):
        v_t = float(talign.soft_local_align(a, b, torch.tensor(sub), torch.tensor(-0.7), beta))
        v_j = float(jalign.soft_local_align(a, b, jnp.asarray(sub), jnp.asarray(-0.7), beta))
        assert abs(v_t - v_j) < 1e-9


# ---------------------------------------------------------------------------
# Gotoh affine-gap alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_gotoh_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sub = _sub(4, rng)
    a, b = _seqs(rng)
    go, ge = -1.5, -0.4
    assert abs(hard_gotoh(a, b, sub, go, ge) - brute_force_gotoh(a, b, sub, go, ge)) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_gotoh_matches_brute_and_anneals(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sub = _sub(4, rng)
    a, b = _seqs(rng)
    go, ge = -1.5, -0.4
    st = torch.tensor(sub)
    hard = hard_gotoh(a, b, sub, go, ge)
    for beta in (0.5, 1.0, 4.0, 16.0):
        soft = float(talign.soft_gotoh(a, b, st, torch.tensor(go), torch.tensor(ge), beta))
        assert abs(soft - brute_force_soft_gotoh(a, b, sub, go, ge, beta)) < 1e-9
        assert soft >= hard - 1e-9


def test_soft_gotoh_certified_gap() -> None:
    rng = np.random.default_rng(2)
    sub = _sub(4, rng)
    a = np.array([0, 1, 2, 3])
    b = np.array([0, 2, 3])
    go, ge = -1.5, -0.4
    st = torch.tensor(sub)
    hard = hard_gotoh(a, b, sub, go, ge)
    num_paths = build_gotoh_dag(len(a), len(b))[0].count_paths()
    prev = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0):
        soft = float(talign.soft_gotoh(a, b, st, torch.tensor(go), torch.tensor(ge), beta))
        cert = certify_soft_dp(
            hard, soft, num_paths, beta, brute_force_value=brute_force_gotoh(a, b, sub, go, ge)
        )
        assert cert.is_sound and cert.agrees_with_bruteforce
        assert cert.absolute_gap <= prev + 1e-12
        prev = cert.absolute_gap


def test_soft_gotoh_marginals_equal_autograd() -> None:
    rng = np.random.default_rng(3)
    sub = _sub(4, rng)
    a = np.array([0, 1, 2, 1])
    b = np.array([0, 2, 1])
    beta = 3.0
    # torch closed-form == autograd
    st = torch.tensor(sub, requires_grad=True)
    got = torch.tensor(-1.5, requires_grad=True)
    get = torch.tensor(-0.4, requires_grad=True)
    talign.soft_gotoh(a, b, st, got, get, beta).backward()
    g_sub, g_open, g_ext = talign.soft_gotoh_marginals(a, b, st.detach(), got.detach(), get.detach(), beta)
    assert torch.max(torch.abs(st.grad - g_sub)).item() < 1e-9
    assert abs(float(got.grad) - float(g_open)) < 1e-9
    assert abs(float(get.grad) - float(g_ext)) < 1e-9
    # jax closed-form == jax.grad
    sj, oj, ej = jnp.asarray(sub), jnp.asarray(-1.5), jnp.asarray(-0.4)
    jg_sub = jax.grad(lambda s: jalign.soft_gotoh(a, b, s, oj, ej, beta))(sj)
    jg_open = jax.grad(lambda o: jalign.soft_gotoh(a, b, sj, o, ej, beta))(oj)
    jg_ext = jax.grad(lambda e: jalign.soft_gotoh(a, b, sj, oj, e, beta))(ej)
    cf_sub, cf_open, cf_ext = jalign.soft_gotoh_marginals(a, b, sj, oj, ej, beta)
    assert float(jnp.max(jnp.abs(jg_sub - cf_sub))) < 1e-9
    assert abs(float(jg_open) - float(cf_open)) < 1e-9
    assert abs(float(jg_ext) - float(cf_ext)) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_gotoh_torch_jax_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sub = _sub(5, rng)
    a, b = _seqs(rng, 5)
    for beta in (1.0, 8.0):
        v_t = float(talign.soft_gotoh(a, b, torch.tensor(sub), torch.tensor(-1.2), torch.tensor(-0.3), beta))
        v_j = float(
            jalign.soft_gotoh(a, b, jnp.asarray(sub), jnp.asarray(-1.2), jnp.asarray(-0.3), beta)
        )
        assert abs(v_t - v_j) < 1e-9
