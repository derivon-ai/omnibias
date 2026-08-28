# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact Pauli / Jordan-Wigner fermionic operator algebra on a finite lattice.

Soundness oracle: exact ``(anti)commutation`` relations checked to machine
precision (every matrix entry is ``0``, ``+-1``, or ``+-i``, so equality
should be exact modulo floating-point roundoff in the ``kron`` products).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from omnibias.core.verified.lattice_operators import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    SIGMA_LOWER,
    SIGMA_RAISE,
    SparseCOO,
    apply_fermion_op,
    apply_fermion_string,
    coo_add,
    coo_from_triplets,
    dense_to_coo,
    embed_operator_string,
    embed_single_site_operator,
    fermion_annihilation_operator,
    fermion_creation_operator,
    jordan_wigner_string,
    pauli_matrices,
    pauli_operator,
    spin_lowering_operator,
    spin_raising_operator,
)


def test_pauli_commutation_relations() -> None:
    sx, sy, sz = pauli_matrices()
    np.testing.assert_allclose(sx @ sy - sy @ sx, 2j * sz, atol=1e-14)
    np.testing.assert_allclose(sy @ sz - sz @ sy, 2j * sx, atol=1e-14)
    np.testing.assert_allclose(sz @ sx - sx @ sz, 2j * sy, atol=1e-14)


def test_pauli_squares_to_identity() -> None:
    sx, sy, sz = pauli_matrices()
    for s in (sx, sy, sz):
        np.testing.assert_allclose(s @ s, np.eye(2), atol=1e-14)


def test_sigma_plus_minus_match_textbook_ladder_formula() -> None:
    """``sigma^+=(sx+i sy)/2`` and ``sigma^-=(sx-i sy)/2`` per the module's basis convention."""
    sigma_plus = (PAULI_X + 1j * PAULI_Y) / 2
    sigma_minus = (PAULI_X - 1j * PAULI_Y) / 2
    np.testing.assert_allclose(sigma_plus, SIGMA_LOWER, atol=1e-14)
    np.testing.assert_allclose(sigma_minus, SIGMA_RAISE, atol=1e-14)


def test_multisite_embedding_bit_order() -> None:
    """Site 0 is the least-significant bit of the computational-basis index."""
    n = 3
    for site in range(n):
        sz = pauli_operator("z", site, n)
        for state in range(2**n):
            vec = np.zeros(2**n, dtype=complex)
            vec[state] = 1.0
            eigenvalue = (vec.conj() @ sz @ vec).real
            expected = 1.0 if not (state >> site) & 1 else -1.0
            assert eigenvalue == pytest.approx(expected)


def test_pauli_operator_invalid_axis_raises() -> None:
    with pytest.raises(ValueError):
        pauli_operator("q", 0, 2)  # type: ignore[arg-type]


def test_embed_operator_string_rejects_bad_site() -> None:
    with pytest.raises(ValueError):
        embed_operator_string({5: PAULI_X}, 3)
    with pytest.raises(ValueError):
        embed_operator_string({0: PAULI_X}, 0)


def test_spin_raising_lowering_act_on_sigma_z_eigenstates() -> None:
    n = 1
    sp = spin_raising_operator(0, n)
    sm = spin_lowering_operator(0, n)
    up = np.array([1.0, 0.0], dtype=complex)  # sigma_z eigenvalue +1
    down = np.array([0.0, 1.0], dtype=complex)  # sigma_z eigenvalue -1
    np.testing.assert_allclose(sp @ down, up, atol=1e-14)
    np.testing.assert_allclose(sp @ up, np.zeros(2), atol=1e-14)
    np.testing.assert_allclose(sm @ up, down, atol=1e-14)
    np.testing.assert_allclose(sm @ down, np.zeros(2), atol=1e-14)


def test_embed_single_site_operator_matches_embed_operator_string() -> None:
    n = 4
    for site in range(n):
        a = embed_single_site_operator(PAULI_X, site, n)
        b = embed_operator_string({site: PAULI_X}, n)
        np.testing.assert_allclose(a, b, atol=1e-14)


def test_jordan_wigner_string_is_product_of_sigma_z() -> None:
    n = 3
    string = jordan_wigner_string([0, 2], n)
    expected = pauli_operator("z", 0, n) @ pauli_operator("z", 2, n)
    np.testing.assert_allclose(string, expected, atol=1e-14)


@pytest.mark.parametrize("n_orbitals", [1, 2, 3, 4])
def test_fermion_anticommutation_relations(n_orbitals: int) -> None:
    """``{c_i, c_j^dagger} = delta_ij`` and ``{c_i, c_j} = 0`` for every ``i, j``."""
    identity = np.eye(2**n_orbitals)
    c_ops = [fermion_annihilation_operator(k, n_orbitals) for k in range(n_orbitals)]
    cdag_ops = [fermion_creation_operator(k, n_orbitals) for k in range(n_orbitals)]
    for i, j in itertools.product(range(n_orbitals), repeat=2):
        anticommutator_cdag = c_ops[i] @ cdag_ops[j] + cdag_ops[j] @ c_ops[i]
        expected = identity if i == j else np.zeros((2**n_orbitals, 2**n_orbitals))
        np.testing.assert_allclose(anticommutator_cdag, expected, atol=1e-12)
        anticommutator_c = c_ops[i] @ c_ops[j] + c_ops[j] @ c_ops[i]
        np.testing.assert_allclose(anticommutator_c, np.zeros((2**n_orbitals, 2**n_orbitals)), atol=1e-12)


def test_fermion_creation_is_conjugate_transpose_of_annihilation() -> None:
    n = 3
    for k in range(n):
        c = fermion_annihilation_operator(k, n)
        cdag = fermion_creation_operator(k, n)
        np.testing.assert_allclose(cdag, c.conj().T, atol=1e-14)


def test_fermion_number_operator_eigenvalues() -> None:
    """``c_k^dagger c_k`` has eigenvalue equal to bit ``k`` of the basis state."""
    n = 3
    for k in range(n):
        number_op = fermion_creation_operator(k, n) @ fermion_annihilation_operator(k, n)
        for state in range(2**n):
            vec = np.zeros(2**n, dtype=complex)
            vec[state] = 1.0
            eigenvalue = (vec.conj() @ number_op @ vec).real
            expected = float((state >> k) & 1)
            assert eigenvalue == pytest.approx(expected)


def test_fermion_operator_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        fermion_annihilation_operator(3, 3)
    with pytest.raises(ValueError):
        fermion_annihilation_operator(-1, 3)


@pytest.mark.parametrize("n_orbitals", [2, 3, 4])
def test_apply_fermion_string_matches_matrix_construction(n_orbitals: int) -> None:
    """`apply_fermion_string` sign/occupation bookkeeping matches explicit matrix products."""
    dim = 2**n_orbitals
    for i, j in itertools.product(range(n_orbitals), repeat=2):
        c_i = fermion_annihilation_operator(i, n_orbitals)
        cdag_j = fermion_creation_operator(j, n_orbitals)
        matrix = c_i @ cdag_j
        for state in range(dim):
            vec = np.zeros(dim, dtype=complex)
            vec[state] = 1.0
            matrix_result = matrix @ vec
            step = apply_fermion_string(state, [(j, True), (i, False)])
            if step is None:
                np.testing.assert_allclose(matrix_result, np.zeros(dim), atol=1e-12)
                continue
            new_state, sign = step
            expected = np.zeros(dim, dtype=complex)
            expected[new_state] = sign
            np.testing.assert_allclose(matrix_result, expected, atol=1e-12)


def test_apply_fermion_op_rejects_forbidden_transitions() -> None:
    # orbital 0 unoccupied: annihilation must fail.
    assert apply_fermion_op(0b00, 0, dagger=False) is None
    # orbital 0 occupied: creation must fail.
    assert apply_fermion_op(0b01, 0, dagger=True) is None


class TestSparseCOO:
    def test_to_dense_matches_hand_built_matrix(self) -> None:
        coo = coo_from_triplets([0, 1], [1, 0], [2.0, 3.0], (2, 2))
        np.testing.assert_allclose(coo.to_dense(), [[0.0, 2.0], [3.0, 0.0]])

    def test_matvec_matches_dense(self) -> None:
        rng = np.random.default_rng(0)
        dense = rng.standard_normal((5, 5))
        coo = dense_to_coo(dense)
        v = rng.standard_normal(5)
        np.testing.assert_allclose(coo.matvec(v), dense @ v, atol=1e-12)

    def test_duplicate_entries_are_summed(self) -> None:
        coo = coo_from_triplets([0, 0], [0, 0], [1.0, 2.0], (1, 1))
        np.testing.assert_allclose(coo.to_dense(), [[3.0]])
        coalesced = coo.coalesce()
        assert coalesced.rows.size == 1
        np.testing.assert_allclose(coalesced.values, [3.0])

    def test_coo_add_concatenates_and_coalesces(self) -> None:
        a = coo_from_triplets([0], [0], [1.0], (2, 2))
        b = coo_from_triplets([0], [0], [4.0], (2, 2))
        combined = coo_add(a, b)
        np.testing.assert_allclose(combined.to_dense(), [[5.0, 0.0], [0.0, 0.0]])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            SparseCOO(
                rows=np.array([0], dtype=np.intp),
                cols=np.array([0], dtype=np.intp),
                values=np.array([1.0]),
                shape=(-1, 1),
            )
        with pytest.raises(ValueError):
            coo_from_triplets([5], [0], [1.0], (2, 2))

    def test_dense_to_coo_drops_below_tolerance(self) -> None:
        dense = np.array([[1.0, 1e-15], [0.0, 2.0]])
        coo = dense_to_coo(dense, tol=1e-9)
        assert coo.rows.size == 2
