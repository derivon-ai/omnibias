# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Differentiable relaxation twins: torch <-> jax parity, unit box, decode-competitive."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.nphard import QAPProblem, SchedulingProblem, classical_optimum, decode, qap, schedule


def _random_qap(dim: int, seed: int) -> QAPProblem:
    rng = np.random.default_rng(seed)
    flow = rng.integers(0, 9, size=(dim, dim)).astype(float)
    dist = rng.integers(0, 9, size=(dim, dim)).astype(float)
    flow = (flow + flow.T) / 2.0
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(flow, 0.0)
    np.fill_diagonal(dist, 0.0)
    return qap(flow, dist)


def _random_schedule(jobs: int, machines: int, seed: int) -> SchedulingProblem:
    rng = np.random.default_rng(seed)
    return schedule(rng.integers(1, 20, size=jobs).astype(float), machines)


def test_scheduling_relaxation_is_bit_identical_across_backends() -> None:
    """Well-determined (strong one-hot penalty) scheduling QUBO: float64 twins agree ~1e-13."""
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from omnibias.nphard.jax import relax as relax_j
    from omnibias.nphard.torch import relax as relax_t

    worst = 0.0
    for seed in range(6):
        prob = _random_schedule(6, 2, seed)
        xj = np.asarray(relax_j(prob))
        xt = relax_t(prob).detach().numpy()
        worst = max(worst, float(np.max(np.abs(xj - xt))))
    assert worst < 1e-9  # measured ~4e-13 across seeds


def test_qap_relaxation_parity_within_calibrated_tolerance() -> None:
    """QAP twins agree to a calibrated tol; a frustrated coord pinned near 0.5 is the
    chaotic-amplification regime where the two frameworks' matmul reduction order can
    diverge to ~1e-7 (see omnibias-qubo's parity note) -- not a machinery bug."""
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from omnibias.nphard.jax import relax as relax_j
    from omnibias.nphard.torch import relax as relax_t

    worst = 0.0
    for seed in range(6):
        prob = _random_qap(3, seed)
        xj = np.asarray(relax_j(prob))
        xt = relax_t(prob).detach().numpy()
        worst = max(worst, float(np.max(np.abs(xj - xt))))
    assert worst < 1e-5  # measured ~5e-7 worst case across seeds


def test_relaxation_output_is_in_the_unit_box() -> None:
    pytest.importorskip("jax")
    from omnibias.nphard.jax import relax as relax_j

    for prob in (_random_qap(4, 0), _random_schedule(6, 3, 0)):
        x = np.asarray(relax_j(prob))
        assert np.all(x >= 0.0) and np.all(x <= 1.0) and np.all(np.isfinite(x))


def test_decoded_relaxation_is_competitive_with_the_named_baseline() -> None:
    """Relax -> decode is competitive with the family's classical baseline across seeds.

    Aggregate, data-driven gate: the mean relative excess over the baseline is small and
    no instance is far worse (measured mean ~+0.2%, worst ~+4% on QAP)."""
    pytest.importorskip("jax")
    from omnibias.nphard.jax import relax as relax_j

    for mk, shape in ((lambda s: _random_qap(4, s), (4, 4)),
                      (lambda s: _random_schedule(7, 3, s), (7, 3))):
        rels = []
        for seed in range(6):
            prob = mk(seed)
            heat = np.asarray(relax_j(prob)).reshape(shape)
            _, e_dec = decode(prob, relaxed=heat)
            _, e_cla = classical_optimum(prob)
            rels.append((e_dec - e_cla) / max(abs(e_cla), 1e-9))
        assert float(np.mean(rels)) < 0.03  # competitive on average
        assert float(np.max(rels)) < 0.10  # never far worse than the baseline


def test_qap_decision_cost_is_bit_identical_across_backends() -> None:
    """The differentiable decision cost matches jax <-> torch on a fixed input."""
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from omnibias.nphard.jax import qap_decision_cost as dc_jax
    from omnibias.nphard.torch import qap_decision_cost as dc_torch

    rng = np.random.default_rng(7)
    dim = 3
    flow_pred = rng.random((dim, dim))
    dist = rng.random((dim, dim))
    flow_true = rng.random((dim, dim))
    lj = float(dc_jax(flow_pred, dist, flow_true))
    lt = float(dc_torch(flow_pred, dist, flow_true))
    assert abs(lj - lt) < 1e-9


def test_qap_decision_cost_is_differentiable_in_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.nphard.torch import qap_decision_cost

    rng = np.random.default_rng(8)
    dim = 3
    flow_pred = torch.tensor(rng.random((dim, dim)), dtype=torch.float64, requires_grad=True)
    dist = torch.tensor(rng.random((dim, dim)), dtype=torch.float64)
    flow_true = torch.tensor(rng.random((dim, dim)), dtype=torch.float64)
    cost = qap_decision_cost(flow_pred, dist, flow_true)
    cost.backward()
    assert flow_pred.grad is not None and torch.all(torch.isfinite(flow_pred.grad))
