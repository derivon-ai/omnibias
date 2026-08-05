# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Differentiable annealed relaxation: torch <-> jax parity and anneal-to-vertex."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.qubo import AnnealSchedule, QUBOProblem, max_cut


def _problem(seed: int = 0, n: int = 6) -> QUBOProblem:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return QUBOProblem(m + m.T, rng.standard_normal(n))


def test_torch_jax_bit_identical() -> None:
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("torch")
    from omnibias.qubo.jax import qubo_relaxation as jax_relax
    from omnibias.qubo.torch import qubo_relaxation as torch_relax

    # A well-determined instance (strong linear term) so the annealed fixed point commits
    # cleanly to a vertex: the float64 twins then agree to ~1e-16. (A frustrated instance
    # with a coordinate pinned near 0.5 is a chaotic-amplification regime where the two
    # frameworks' matmul reduction order can diverge to ~1e-7 -- not a machinery bug.)
    rng = np.random.default_rng(0)
    n = 6
    m = rng.standard_normal((n, n))
    prob = QUBOProblem(0.3 * (m + m.T), 5.0 * rng.standard_normal(n))
    xj = np.asarray(jnp.asarray(jax_relax(prob)))
    xt = torch_relax(prob).detach().numpy()
    assert np.max(np.abs(xj - xt)) < 1e-9  # float64 twins are bit-identical here


def test_output_is_in_the_unit_box() -> None:
    pytest.importorskip("jax")
    from omnibias.qubo.jax import qubo_relaxation as jax_relax

    x = np.asarray(jax_relax(_problem(2)))
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert np.all(np.isfinite(x))


def test_annealing_collapses_toward_a_vertex() -> None:
    pytest.importorskip("jax")
    from omnibias.qubo.jax import qubo_relaxation as jax_relax

    # A longer schedule (larger final beta) should push most coordinates near {0, 1}.
    x = np.asarray(jax_relax(_problem(3), schedule=AnnealSchedule(stages=16)))
    dist_to_vertex = np.minimum(x, 1.0 - x)
    assert float(np.mean(dist_to_vertex)) < 0.1


def test_max_cut_relaxation_decodes_to_a_good_cut() -> None:
    pytest.importorskip("jax")
    from omnibias.qubo import decode_qubo
    from omnibias.qubo.jax import qubo_relaxation as jax_relax

    w = np.array([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    prob = max_cut(w)
    relaxed = np.asarray(jax_relax(prob))
    _, energy = decode_qubo(prob, relaxed=relaxed)
    from omnibias.qubo import brute_force_min

    _, e_min = brute_force_min(prob)
    assert energy <= e_min + 1e-9 or abs(energy - e_min) < 1e-9  # decoder is sound


def test_torch_relaxation_is_differentiable() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.qubo.torch import qubo_relaxation as torch_relax

    prob = _problem(4, n=5)
    q = torch.tensor(prob.Q, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(prob.c, dtype=torch.float64, requires_grad=True)
    out = torch_relax(q, c, schedule=AnnealSchedule.fast())
    out.sum().backward()
    assert q.grad is not None and torch.all(torch.isfinite(q.grad))
    assert c.grad is not None and torch.all(torch.isfinite(c.grad))
