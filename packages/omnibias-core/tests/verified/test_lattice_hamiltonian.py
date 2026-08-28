# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named finite-lattice Hamiltonian families, symmetry sectors, and the ED oracle.

Soundness oracles per family (each is *independent* of this package's code
path -- either a symbolic/hand diagonalization external to
:mod:`omnibias.core.verified.lattice_hamiltonian`, or a standard textbook
closed form):

* **TFIM** (2-site open chain): the explicit ``4x4`` matrix ``-J sigma^z_1
  sigma^z_2 - h(sigma^x_1+sigma^x_2)`` has symbolically-derived eigenvalues
  ``{J, -J, sqrt(J^2+4h^2), -sqrt(J^2+4h^2)}`` (``sympy`` re-derivation, not
  this module).
* **Heisenberg XXZ** (2-site dimer): the standard spin-1/2 dimer result
  ``H = J_xy(S^x_1 S^x_2+S^y_1 S^y_2) + J_z S^z_1 S^z_2`` has eigenvalues
  ``{J_z/4 (x2), J_xy/2-J_z/4, -J_xy/2-J_z/4}``; isotropic recovers the
  textbook singlet/triplet split ``-3J/4`` / ``+J/4``.
* **Hubbard** (2-site dimer, half filling): the standard closed form
  ``E_0 = U/2 - sqrt(U^2+16t^2)/2``.
* **t-J** (2 sites): at ``N=1`` (one particle) the exchange term cannot act
  (needs both sites occupied) and the Hamiltonian reduces to plain 2-site
  hopping, giving ``{-t,-t,+t,+t}``; at ``N=2`` (half filling) hopping cannot
  act (no empty site) and the Hamiltonian reduces to the Heisenberg dimer
  shifted by the density term, giving ``{-J, 0, 0, 0}``.

All four reference computations were re-derived from the physical definition
independently of this module (see the comments inline) before being encoded
as literal expected numbers below.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.verified import eig_operator
from omnibias.core.verified.lattice_hamiltonian import (
    HamiltonianMatrix,
    NumericalSpectrum,
    chain_bonds,
    ground_state_energy,
    hamiltonian_to_matrix,
    heisenberg_hamiltonian,
    hubbard_hamiltonian,
    numerical_exact_diagonalization,
    particle_number,
    restrict_to_indices,
    restrict_via_basis,
    sector_indices,
    spin_resolved_occupation,
    tfim_hamiltonian,
    tj_hamiltonian,
    total_sz_doubled,
    z2_parity_basis,
)
from omnibias.core.verified.linalg import identity_matrix


# --------------------------------------------------------------------------- #
# Lattice geometry.
# --------------------------------------------------------------------------- #
def test_chain_bonds_open() -> None:
    assert chain_bonds(1, "open") == []
    assert chain_bonds(4, "open") == [(0, 1), (1, 2), (2, 3)]


def test_chain_bonds_periodic() -> None:
    assert chain_bonds(2, "periodic") == [(0, 1)]  # no duplicate wraparound
    assert chain_bonds(3, "periodic") == [(0, 1), (1, 2), (2, 0)]
    assert chain_bonds(4, "periodic") == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_chain_bonds_invalid_raises() -> None:
    with pytest.raises(ValueError):
        chain_bonds(0, "open")
    with pytest.raises(ValueError):
        chain_bonds(1, "periodic")
    with pytest.raises(ValueError):
        chain_bonds(3, "diagonal")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# TFIM: cross-check against a symbolically re-derived (independent) 2-site spectrum.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("coupling,field", [(1.0, 0.5), (2.0, 0.3), (0.7, 1.4)])
def test_tfim_two_site_matches_independent_closed_form(coupling: float, field: float) -> None:
    h = tfim_hamiltonian(2, "open", coupling=coupling, field=field)
    spectrum = numerical_exact_diagonalization(h.dense)
    expected = sorted(
        [coupling, -coupling, np.sqrt(coupling**2 + 4 * field**2), -np.sqrt(coupling**2 + 4 * field**2)]
    )
    np.testing.assert_allclose(sorted(spectrum.eigenvalues), expected, atol=1e-10)


def test_tfim_dimension_and_hermiticity() -> None:
    h = tfim_hamiltonian(6, "periodic", coupling=1.0, field=0.8)
    assert h.dense.shape == (64, 64)
    np.testing.assert_allclose(h.dense, h.dense.T, atol=1e-12)


# --------------------------------------------------------------------------- #
# Heisenberg XXZ: cross-check against the standard spin-1/2 dimer result.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("j_xy,j_z", [(1.0, 1.0), (1.3, 0.7), (0.5, 2.0)])
def test_heisenberg_dimer_matches_textbook_result(j_xy: float, j_z: float) -> None:
    h = heisenberg_hamiltonian(2, "open", j_xy=j_xy, j_z=j_z)
    spectrum = numerical_exact_diagonalization(h.dense)
    expected = sorted([j_z / 4, j_z / 4, j_xy / 2 - j_z / 4, -j_xy / 2 - j_z / 4])
    np.testing.assert_allclose(sorted(spectrum.eigenvalues), expected, atol=1e-10)


def test_heisenberg_isotropic_dimer_singlet_triplet_split() -> None:
    """The textbook antiferromagnetic Heisenberg dimer: singlet -3J/4, triplet +J/4 (x3)."""
    j = 1.0
    h = heisenberg_hamiltonian(2, "open", j_xy=j, j_z=j)
    spectrum = numerical_exact_diagonalization(h.dense)
    np.testing.assert_allclose(sorted(spectrum.eigenvalues), [-0.75, 0.25, 0.25, 0.25], atol=1e-10)


# --------------------------------------------------------------------------- #
# Hubbard: cross-check against the standard 2-site half-filling closed form.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t,u", [(1.0, 2.0), (1.3, 2.7), (0.5, 5.0)])
def test_hubbard_dimer_half_filling_matches_closed_form(t: float, u: float) -> None:
    h = hubbard_hamiltonian(2, "open", hopping=t, u=u)
    n_up, n_down = spin_resolved_occupation(2)
    idx = sector_indices(n_up + n_down, 2)
    sub = restrict_to_indices(h.dense, idx)
    ground = ground_state_energy(sub)
    expected = u / 2 - np.sqrt(u**2 + 16 * t**2) / 2
    assert ground == pytest.approx(expected, abs=1e-10)


def test_hubbard_atomic_limit_t_zero() -> None:
    """At ``t=0`` every doubly-occupied site contributes exactly ``U``, else ``0``.

    Each site independently has 4 local configurations (empty, up, down,
    doubly-occupied); exactly 1 of those 4 is doubly-occupied. Choosing which
    ``k`` of the 3 sites are doubly-occupied gives multiplicity
    ``C(3, k) * 3**(3 - k)`` for energy ``k * U`` (the other ``3 - k`` sites
    each range freely over their 3 non-doubly-occupied local states).
    """
    h = hubbard_hamiltonian(3, "open", hopping=0.0, u=4.0)
    spectrum = numerical_exact_diagonalization(h.dense)
    unique, counts = np.unique(np.round(spectrum.eigenvalues, 8), return_counts=True)
    np.testing.assert_allclose(unique, [0.0, 4.0, 8.0, 12.0])
    np.testing.assert_array_equal(counts, [27, 27, 9, 1])
    assert int(np.sum(counts)) == 4**3


# --------------------------------------------------------------------------- #
# t-J: cross-check against the two independently-solvable 2-site sectors.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t,j", [(1.6, 0.9), (1.0, 1.0), (2.0, 0.3)])
def test_tj_two_site_n1_sector_is_pure_hopping(t: float, j: float) -> None:
    """At N=1 the exchange term cannot act; the spectrum is the pure 2-site hop {+-t, +-t}."""
    h = tj_hamiltonian(2, "open", hopping=t, j_coupling=j)
    n_up, n_down = spin_resolved_occupation(2)
    n_full = n_up + n_down
    n_sub = n_full[h.basis_states]
    idx = sector_indices(n_sub, 1)
    sub = restrict_to_indices(h.dense, idx)
    spectrum = numerical_exact_diagonalization(sub)
    np.testing.assert_allclose(sorted(spectrum.eigenvalues), sorted([t, t, -t, -t]), atol=1e-10)


@pytest.mark.parametrize("t,j", [(1.6, 0.9), (1.0, 1.0), (2.0, 0.3)])
def test_tj_two_site_n2_sector_is_heisenberg_like(t: float, j: float) -> None:
    """At N=2 (half filling) hopping cannot act; spectrum reduces to {-J, 0, 0, 0}."""
    h = tj_hamiltonian(2, "open", hopping=t, j_coupling=j)
    n_up, n_down = spin_resolved_occupation(2)
    n_full = n_up + n_down
    n_sub = n_full[h.basis_states]
    idx = sector_indices(n_sub, 2)
    sub = restrict_to_indices(h.dense, idx)
    spectrum = numerical_exact_diagonalization(sub)
    np.testing.assert_allclose(sorted(spectrum.eigenvalues), sorted([-j, 0.0, 0.0, 0.0]), atol=1e-10)


def test_tj_hilbert_space_dimension_is_3_to_the_n() -> None:
    for n in range(1, 5):
        h = tj_hamiltonian(n, "open", hopping=1.0, j_coupling=0.5)
        assert h.dense.shape == (3**n, 3**n)
        assert h.basis_states.shape == (3**n,)


# --------------------------------------------------------------------------- #
# Symmetry sectors: full-space spectrum equals the union of sector spectra.
# --------------------------------------------------------------------------- #
def test_heisenberg_sz_sector_union_reproduces_full_spectrum() -> None:
    n = 4
    h = heisenberg_hamiltonian(n, "periodic", j_xy=1.0, j_z=0.6)
    full_spectrum = numerical_exact_diagonalization(h.dense).eigenvalues
    labels = total_sz_doubled(n)
    sector_eigs: list[float] = []
    for target in sorted(set(labels.tolist())):
        idx = sector_indices(labels, target)
        sub = restrict_to_indices(h.dense, idx)
        sector_eigs.extend(numerical_exact_diagonalization(sub).eigenvalues.tolist())
    np.testing.assert_allclose(sorted(full_spectrum), sorted(sector_eigs), atol=1e-10)


def test_tfim_z2_parity_sector_union_reproduces_full_spectrum() -> None:
    n = 5
    h = tfim_hamiltonian(n, "periodic", coupling=1.0, field=0.7)
    full_spectrum = numerical_exact_diagonalization(h.dense).eigenvalues
    basis_plus = z2_parity_basis(n, 1)
    basis_minus = z2_parity_basis(n, -1)
    assert basis_plus.shape == (2**n, 2 ** (n - 1))
    assert basis_minus.shape == (2**n, 2 ** (n - 1))
    eig_plus = numerical_exact_diagonalization(restrict_via_basis(h.dense, basis_plus)).eigenvalues
    eig_minus = numerical_exact_diagonalization(restrict_via_basis(h.dense, basis_minus)).eigenvalues
    combined = np.concatenate([eig_plus, eig_minus])
    np.testing.assert_allclose(sorted(full_spectrum), sorted(combined), atol=1e-10)


def test_z2_parity_basis_is_orthonormal() -> None:
    basis = z2_parity_basis(4, 1)
    gram = basis.T @ basis
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-12)


def test_hubbard_spin_resolved_sector_union_reproduces_full_spectrum() -> None:
    n = 3
    h = hubbard_hamiltonian(n, "open", hopping=1.1, u=2.2)
    full_spectrum = numerical_exact_diagonalization(h.dense).eigenvalues
    n_up, n_down = spin_resolved_occupation(n)
    sector_eigs: list[float] = []
    for nu in range(n + 1):
        for nd in range(n + 1):
            idx = np.flatnonzero((n_up == nu) & (n_down == nd)).astype(np.intp)
            if idx.size == 0:
                continue
            sub = restrict_to_indices(h.dense, idx)
            sector_eigs.extend(numerical_exact_diagonalization(sub).eigenvalues.tolist())
    np.testing.assert_allclose(sorted(full_spectrum), sorted(sector_eigs), atol=1e-10)


def test_particle_number_labels_match_popcount() -> None:
    labels = particle_number(4)
    for state in range(16):
        assert labels[state] == bin(state).count("1")


def test_sector_indices_empty_selection_raises_on_restrict() -> None:
    labels = total_sz_doubled(2)
    idx = sector_indices(labels, 99)  # unreachable target
    assert idx.size == 0
    with pytest.raises(ValueError):
        restrict_to_indices(np.eye(4), idx)


# --------------------------------------------------------------------------- #
# Sparse action correctness.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "make_hamiltonian",
    [
        lambda: tfim_hamiltonian(5, "periodic", coupling=1.0, field=0.7),
        lambda: heisenberg_hamiltonian(4, "periodic", j_xy=1.0, j_z=0.6),
        lambda: hubbard_hamiltonian(3, "open", hopping=1.1, u=2.2),
        lambda: tj_hamiltonian(3, "open", hopping=1.0, j_coupling=0.5),
    ],
    ids=["tfim", "heisenberg", "hubbard", "tj"],
)
def test_sparse_matvec_matches_dense(make_hamiltonian: object) -> None:
    h = make_hamiltonian()  # type: ignore[operator]
    rng = np.random.default_rng(0)
    v = rng.standard_normal(h.dense.shape[0])
    np.testing.assert_allclose(h.sparse.matvec(v), h.dense @ v, atol=1e-10)
    np.testing.assert_allclose(h.sparse.to_dense(), h.dense, atol=1e-12)


# --------------------------------------------------------------------------- #
# Numerical (not certified) exact-diagonalization oracle.
# --------------------------------------------------------------------------- #
def test_numerical_exact_diagonalization_returns_dataclass() -> None:
    h = heisenberg_hamiltonian(2, "open", j_xy=1.0, j_z=1.0)
    spectrum = numerical_exact_diagonalization(h.dense)
    assert isinstance(spectrum, NumericalSpectrum)
    assert spectrum.eigenvalues.shape == (4,)
    assert spectrum.eigenvectors.shape == (4, 4)
    # Reconstructs H from its own eigendecomposition (definition check, not a new oracle).
    reconstructed = spectrum.eigenvectors @ np.diag(spectrum.eigenvalues) @ spectrum.eigenvectors.T
    np.testing.assert_allclose(reconstructed, h.dense, atol=1e-10)


def test_numerical_exact_diagonalization_rejects_asymmetric_input() -> None:
    with pytest.raises(ValueError):
        numerical_exact_diagonalization(np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        numerical_exact_diagonalization(np.zeros((2, 3)))


def test_ground_state_energy_matches_eigh_minimum() -> None:
    h = tfim_hamiltonian(4, "open", coupling=1.0, field=0.3)
    assert ground_state_energy(h.dense) == pytest.approx(float(np.linalg.eigvalsh(h.dense)[0]))


# --------------------------------------------------------------------------- #
# Interval-matrix conversion: round-trip and direct eig_operator smoke test.
# --------------------------------------------------------------------------- #
def test_hamiltonian_to_matrix_round_trips_exactly() -> None:
    h = heisenberg_hamiltonian(2, "open", j_xy=1.0, j_z=1.0)
    matrix = hamiltonian_to_matrix(h.dense)
    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)
    round_tripped = np.array([[float(x) for x in row] for row in matrix])
    np.testing.assert_array_equal(round_tripped, h.dense)


def test_hamiltonian_to_matrix_rejects_non_square() -> None:
    with pytest.raises(ValueError):
        hamiltonian_to_matrix(np.zeros((2, 3)))


def test_hamiltonian_to_matrix_feeds_eig_operator_directly() -> None:
    """The future certified-ground-state integration point: works today, not just in theory."""
    h = heisenberg_hamiltonian(2, "open", j_xy=1.0, j_z=1.0)
    matrix = hamiltonian_to_matrix(h.dense)
    n = len(matrix)
    identity = identity_matrix(n)

    inertia = eig_operator.interval_ldlt_inertia(matrix)
    assert inertia is not None
    true_eigenvalues = sorted(np.linalg.eigvalsh(h.dense))
    assert inertia.negative == sum(1 for e in true_eigenvalues if e < 0.0)
    assert inertia.positive == sum(1 for e in true_eigenvalues if e > 0.0)

    for threshold in (-1.0, -0.5, 0.0, 0.5, 1.0):
        count = eig_operator.count_eigenvalues_below(matrix, identity, threshold)
        expected_count = sum(1 for e in true_eigenvalues if e < threshold)
        assert count == expected_count


@pytest.mark.parametrize(
    "make_hamiltonian",
    [
        lambda: tfim_hamiltonian(3, "open", coupling=1.0, field=0.5).dense,
        lambda: heisenberg_hamiltonian(3, "open", j_xy=1.0, j_z=0.5).dense,
        lambda: hubbard_hamiltonian(2, "open", hopping=1.0, u=2.0).dense,
        lambda: tj_hamiltonian(2, "open", hopping=1.0, j_coupling=0.5).dense,
    ],
    ids=["tfim", "heisenberg", "hubbard", "tj"],
)
def test_hamiltonian_to_matrix_count_eigenvalues_below_matches_eigh(make_hamiltonian: object) -> None:
    """Thresholds are midpoints strictly between distinct eigenvalues (never *on* the spectrum).

    ``count_eigenvalues_below`` honestly returns ``None`` when an *unpivoted*
    ``LDL^T`` pivot straddles ``0`` -- for an indefinite matrix this can
    happen even at a threshold that is not itself an eigenvalue (a vanishing
    leading principal minor at that shift, independent of whether the shifted
    matrix as a whole is singular). That is a known, documented limitation of
    the no-pivoting certified primitive itself (see
    :func:`omnibias.core.verified.eig_operator.interval_ldlt_inertia`), not a
    defect introduced here, so this test only requires agreement whenever a
    definite answer *is* returned, plus at least one certified threshold per
    instance (proving the integration is not vacuously ``None`` everywhere).
    """
    dense = make_hamiltonian()  # type: ignore[operator]
    matrix = hamiltonian_to_matrix(dense)
    identity = identity_matrix(len(matrix))
    true_eigenvalues = np.sort(np.linalg.eigvalsh(dense))
    distinct = np.unique(np.round(true_eigenvalues, 9))
    thresholds = [distinct[0] - 1.0, *(0.5 * (distinct[:-1] + distinct[1:])), distinct[-1] + 1.0]
    certified = 0
    for threshold in thresholds:
        count = eig_operator.count_eigenvalues_below(matrix, identity, float(threshold))
        if count is None:
            continue
        certified += 1
        expected = int(np.sum(true_eigenvalues < threshold))
        assert count == expected
    assert certified > 0


def test_hamiltonian_matrix_dataclass_fields() -> None:
    h = tfim_hamiltonian(2, "open", coupling=1.0, field=0.5)
    assert isinstance(h, HamiltonianMatrix)
    np.testing.assert_allclose(h.sparse.to_dense(), h.dense, atol=1e-12)
