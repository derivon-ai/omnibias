# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""MaxSAT relaxation: torch <-> jax parity and unit-box output."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.discrete.maxsat import max_sat

_CLAUSES = [[1, -2], [2, 3], [-1, -3], [1, 2, 3]]


def test_torch_jax_bit_identical() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    from omnibias.discrete.maxsat.jax import maxsat_relaxation as jax_relax
    from omnibias.discrete.maxsat.torch import maxsat_relaxation as torch_relax

    prob = max_sat(_CLAUSES, weights=[1.0, 2.0, 0.5, 1.5])
    xj = np.asarray(jax_relax(prob))
    xt = torch_relax(prob).detach().numpy()
    assert np.max(np.abs(xj - xt)) < 1e-8


def test_output_is_in_the_unit_box() -> None:
    pytest.importorskip("jax")
    from omnibias.discrete.maxsat.jax import maxsat_relaxation as jax_relax

    x = np.asarray(jax_relax(max_sat(_CLAUSES)))
    assert np.all(x >= 0.0) and np.all(x <= 1.0) and np.all(np.isfinite(x))


def test_relaxation_is_differentiable_in_torch() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.discrete import AnnealSchedule
    from omnibias.discrete.maxsat.torch import maxsat_relaxation as torch_relax

    out = torch_relax(max_sat(_CLAUSES), AnnealSchedule.fast())
    assert out.shape[0] == 3 and torch.all(torch.isfinite(out))
