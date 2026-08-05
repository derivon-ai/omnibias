# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The annealed #SAT relaxation: torch <-> jax parity, unit box, and model finding."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.logic import count_enclosure, decode, exact_model_count, model_count

_CLAUSES = [[1, -2], [2, 3], [-1, -3], [1, 2, 3]]


def test_torch_jax_bit_identical_across_instances() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from omnibias.logic.jax import sat_relaxation as jax_relax
    from omnibias.logic.torch import sat_relaxation as torch_relax

    worst = 0.0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 7))
        m = int(rng.integers(2, 7))
        clauses = []
        for _ in range(m):
            k = int(rng.integers(1, 4))
            variables = rng.choice(np.arange(1, n + 1), size=min(k, n), replace=False)
            signs = rng.choice([-1, 1], size=len(variables))
            clauses.append([int(s * v) for s, v in zip(signs, variables, strict=True)])
        mc = model_count(clauses, n_vars=n)
        xj = np.asarray(jax_relax(mc))
        xt = torch_relax(mc).detach().numpy()
        worst = max(worst, float(np.max(np.abs(xj - xt))))
    assert worst < 1e-8, f"torch<->jax parity {worst} exceeds tol"


def test_output_is_in_the_unit_box() -> None:
    pytest.importorskip("jax")
    from omnibias.logic.jax import sat_relaxation as jax_relax

    x = np.asarray(jax_relax(model_count(_CLAUSES)))
    assert np.all(x >= 0.0) and np.all(x <= 1.0) and np.all(np.isfinite(x))


def test_relaxation_is_differentiable_in_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.discrete import AnnealSchedule
    from omnibias.logic.torch import sat_relaxation as torch_relax

    out = torch_relax(model_count(_CLAUSES), AnnealSchedule.fast())
    assert out.shape[0] == 3 and torch.all(torch.isfinite(out))


def test_decoded_relaxation_is_a_model_when_satisfiable() -> None:
    pytest.importorskip("torch")
    from omnibias.logic.torch import sat_relaxation as torch_relax

    mc = model_count(_CLAUSES)
    x_soft = torch_relax(mc).detach().numpy()
    z, energy = decode(mc, relaxed=x_soft, n_starts=16)
    assert energy == 0.0
    assert mc.is_model(np.asarray(z, dtype=float))


def test_multistart_witnesses_give_a_sound_lower_bound() -> None:
    # multi-start decode collects distinct models; they must not exceed the exact count and
    # must sit inside the certified enclosure.
    mc = model_count(_CLAUSES)
    exact = exact_model_count(mc)
    witnesses = []
    for seed in range(24):
        z, energy = decode(mc, n_starts=4, seed=seed)
        if energy == 0.0:
            witnesses.append(z)
    distinct = {tuple(w) for w in witnesses}
    assert len(distinct) <= exact
    enc = count_enclosure(mc, order=1, witnesses=np.array(list(distinct), dtype=float))
    assert enc.lower <= exact
    assert enc.contains(exact)
