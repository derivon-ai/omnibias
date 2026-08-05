# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch <-> JAX bit-identical parity (float64) for every soft-DP layer and its marginals.

Parity is measured *across seeds* (the anti-overfitting rule) at a uniform ``1e-9``, plus
a pinned well-determined instance held to a tight ``1e-11``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import omnibias.struct.jax as sj  # noqa: E402
import omnibias.struct.torch as st  # noqa: E402
from _struct_helpers import dag_weight_matrix, random_chain, random_dag, sample_ctc  # noqa: E402

SEEDS = range(8)
ACROSS_SEED_TOL = 1e-9
PINNED_TOL = 1e-11


def _max_abs_diff(a: object, b: object) -> float:
    an = a.detach().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    bn = b.detach().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    return float(np.max(np.abs(an - bn)))


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_viterbi_parity(seed: int) -> None:
    trellis = random_chain(seed)
    args_t = (torch.tensor(trellis.emissions), torch.tensor(trellis.transitions))
    args_j = (jnp.asarray(trellis.emissions), jnp.asarray(trellis.transitions))
    start_t, start_j = torch.tensor(trellis.start), jnp.asarray(trellis.start)
    for beta in (1.0, 4.0, 16.0):
        vt = st.soft_viterbi(*args_t, beta, start=start_t)
        vj = sj.soft_viterbi(*args_j, beta, start=start_j)
        assert abs(float(vt) - float(vj)) < ACROSS_SEED_TOL
        gt = st.soft_viterbi_marginals(*args_t, beta, start=start_t)
        gj = sj.soft_viterbi_marginals(*args_j, beta, start=start_j)
        assert _max_abs_diff(gt, gj) < ACROSS_SEED_TOL


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_shortest_path_parity(seed: int) -> None:
    dag = random_dag(seed)
    wt, wj = torch.tensor(dag_weight_matrix(dag)), jnp.asarray(dag_weight_matrix(dag))
    for beta in (1.0, 8.0):
        assert abs(float(st.soft_shortest_path(wt, dag, beta)) - float(sj.soft_shortest_path(wj, dag, beta))) < ACROSS_SEED_TOL
        xt = st.soft_shortest_path_marginals(wt, dag, beta)
        xj = sj.soft_shortest_path_marginals(wj, dag, beta)
        assert _max_abs_diff(xt, xj) < ACROSS_SEED_TOL


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_ctc_parity(seed: int) -> None:
    lattice, log_probs = sample_ctc(seed)
    lpt, lpj = torch.tensor(log_probs), jnp.asarray(log_probs)
    for beta in (1.0, 8.0):
        assert abs(float(st.soft_ctc(lpt, lattice, beta)) - float(sj.soft_ctc(lpj, lattice, beta))) < ACROSS_SEED_TOL


def test_pinned_instance_is_bit_identical_to_tight_tol() -> None:
    # A single well-determined instance held to a tight tolerance (not seed-tuned).
    trellis = random_chain(0)
    vt = st.soft_viterbi(torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), 4.0, start=torch.tensor(trellis.start))
    vj = sj.soft_viterbi(jnp.asarray(trellis.emissions), jnp.asarray(trellis.transitions), 4.0, start=jnp.asarray(trellis.start))
    assert abs(float(vt) - float(vj)) < PINNED_TOL
    gt = st.soft_viterbi_marginals(torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), 4.0, start=torch.tensor(trellis.start))
    gj = sj.soft_viterbi_marginals(jnp.asarray(trellis.emissions), jnp.asarray(trellis.transitions), 4.0, start=jnp.asarray(trellis.start))
    assert _max_abs_diff(gt, gj) < PINNED_TOL
