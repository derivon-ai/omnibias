# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Guided C∞ Fourier hunt for periodic 1-D viscous Burgers."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn.solver import (
    burgers_residual_periodic,
    cinf_fourier_basis,
    guided_cinf_burgers_hunt,
    guided_cinf_smoke,
)
from omnibias.pinn.solver._core.guided_cinf import _topk_mask


def test_cinf_fourier_basis_columns() -> None:
    x = np.linspace(0.0, 1.0, 32, endpoint=False)
    basis, names = cinf_fourier_basis(x, 1)
    assert names == ["1", "cos_1", "sin_1"]
    np.testing.assert_allclose(basis[:, 0], 1.0)
    np.testing.assert_allclose(basis[:, 1], np.cos(2.0 * np.pi * x))
    np.testing.assert_allclose(basis[:, 2], np.sin(2.0 * np.pi * x))


def test_topk_mask_picks_largest_importance() -> None:
    mask = _topk_mask(np.array([0.1, 3.0, 0.2]), 1)
    assert mask.tolist() == [False, True, False]


def test_constant_cole_hopf_hunt_seals_in_q() -> None:
    viscosity = 0.05
    report = guided_cinf_burgers_hunt(
        n_grid=32,
        n_modes=3,
        k_terms=1,
        viscosity=viscosity,
        outer_iters=3,
        use_picard=False,
        try_exact_seal=True,
    )
    assert report.q_reconstruction_seal is True
    assert report.burgers_residual == 0.0
    assert report.honesty["navier_stokes_proof_claim"] is False
    assert report.honesty["q_seal_is_not_pde_identity"] is True
    x = np.linspace(0.0, 1.0, 32, endpoint=False)
    dx = float(x[1] - x[0])
    residual = burgers_residual_periodic(
        np.full(32, 2.0 * viscosity), viscosity=viscosity, dx=dx
    )
    assert float(np.max(np.abs(residual))) == 0.0


def test_guided_cinf_smoke_wraps_numpy_bools() -> None:
    smoke = guided_cinf_smoke()
    assert smoke["q_reconstruction_seal"] is True
    assert type(smoke["q_reconstruction_seal"]) is bool
    assert type(smoke["monotonically_improved"]) is bool
    assert smoke["burgers_residual"] == 0.0
    assert smoke["navier_stokes_proof_claim"] is False
    assert smoke["honesty"]["navier_stokes_proof_claim"] is False


def test_matroid_path_same_honesty() -> None:
    pytest.importorskip("omnibias.combinatorics")
    pytest.importorskip("omnibias.combinatorics.torch")
    pytest.importorskip("torch")
    report = guided_cinf_burgers_hunt(
        n_grid=32,
        n_modes=3,
        k_terms=1,
        viscosity=0.05,
        outer_iters=3,
        use_picard=False,
        try_exact_seal=True,
    )
    assert report.honesty["combinatorics_guided"] is True
    assert report.q_reconstruction_seal is True
    assert report.burgers_residual == 0.0
    assert report.honesty["navier_stokes_proof_claim"] is False
