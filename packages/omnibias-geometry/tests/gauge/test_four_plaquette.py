# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Four-plaquette KS Hamiltonian: census and basis only.

The certified gap of the 153-D ``j_max=1`` matrix is an ensemble-laws
L2 probe (~20s).  Do not put that gap in default omnibias CI.
"""

from __future__ import annotations

import numpy as np
from omnibias.geometry.gauge.transfer.hamiltonian import (
    COUPLING_LOCK,
    FOUR_PLAQUETTE_ELECTRIC,
    four_plaquette_basis,
    legal_chain4,
    rebuild_hamiltonian,
    su2_four_plaquette_hamiltonian,
)


def test_edge_census_is_locked() -> None:
    assert FOUR_PLAQUETTE_ELECTRIC == (3, 2, 2, 3, 1, 1, 1)
    assert sum(FOUR_PLAQUETTE_ELECTRIC) == 13


def test_j_max_1_basis_is_153_and_legal() -> None:
    basis = four_plaquette_basis(1)
    assert len(basis) == 153
    for t1, t2, t3, t4, s12, s23, s34 in basis:
        assert legal_chain4(t1, t2, t3, t4, s12, s23, s34, two_j_max=2)
    hamiltonian = su2_four_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    assert hamiltonian.dimension == 153
    assert hamiltonian.model == "su2_four_plaquette"
    assert hamiltonian.parameters["n_plaquettes"] == 4
    assert hamiltonian.basis == basis


def test_matrix_is_symmetric() -> None:
    hamiltonian = su2_four_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    mid = np.array(
        [[0.5 * (c.lo + c.hi) for c in row] for row in hamiltonian.entries]
    )
    np.testing.assert_allclose(mid, mid.T, atol=1e-12)


def test_rebuild_round_trips_parameters() -> None:
    hamiltonian = su2_four_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    rebuilt = rebuild_hamiltonian(hamiltonian.parameters)
    assert rebuilt.dimension == hamiltonian.dimension
    assert rebuilt.model == hamiltonian.model
    assert rebuilt.basis == hamiltonian.basis
