# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent Haar/CG contractions of the two-vertex theta graph.

The oracle integrates representation matrix elements with Schur orthogonality
and Clebsch--Gordan coefficients. It never evaluates a 6j symbol.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product

import numpy as np
import pytest
from omnibias.geometry.gauge.transfer.hamiltonian import physical_basis
from omnibias.geometry.gauge.transfer.sixj import (
    magnetic_sixj_amplitude,
    racah_sixj_squared,
)


def test_character_sector_has_unit_amplitudes() -> None:
    for a in range(12):
        forward = magnetic_sixj_amplitude(a, a, 0, a + 1, a + 1)
        reverse = magnetic_sixj_amplitude(a + 1, a + 1, 0, a, a)
        assert forward.lo <= 1 <= forward.hi
        assert reverse.lo <= 1 <= reverse.hi
        assert forward.width < 1e-12
    assert not magnetic_sixj_amplitude(0, 0, 0, 1, 1).contains(2**0.5)


def test_exact_square_and_illegal_labels() -> None:
    assert racah_sixj_squared(0, 0, 0, 1, 1, 1) == Fraction(1, 2)
    assert racah_sixj_squared(2, 2, 2, 2, 2, 2) == Fraction(1, 36)
    assert racah_sixj_squared(1, 1, 1, 1, 1, 1) == 0
    with pytest.raises(ValueError):
        racah_sixj_squared(True, 0, 0, 1, 1, 1)


def test_nontrivial_spectator_matches_exact_haar_projectors() -> None:
    sym = pytest.importorskip("sympy")
    from sympy.physics.wigner import clebsch_gordan

    @lru_cache(None)
    def cg(a, b, s, m, n, r):
        return clebsch_gordan(*(sym.Rational(x, 2) for x in (a, b, s, m, n, r)))

    @lru_cache(None)
    def projector(a, b, s):
        labels = tuple(product(range(-a, a + 1, 2), range(-b, b + 1, 2)))
        return labels, {
            (i, j): sum(cg(a, b, s, *left, r) * cg(a, b, s, *right, r)
                        for r in range(-s, s + 1, 2))
            for i, left in enumerate(labels) for j, right in enumerate(labels)
        }

    # b_state=Tr(P_s(D^a(U) tensor D^b(V)))/(2s+1).
    # Its Haar Gram diagonal is 1/[(2a+1)(2b+1)(2s+1)].
    def haar_bilinear(a, b, s, ap, sp):
        old_labels, old = projector(a, b, s)
        new_labels, new = projector(ap, b, sp)
        new_index = {labels: i for i, labels in enumerate(new_labels)}
        total = sym.S.Zero
        for i, (m, p) in enumerate(old_labels):
            for j, (n, q) in enumerate(old_labels):
                if old[j, i] == 0:
                    continue
                for sigma in (-1, 1):
                    M, N = m + sigma, n + sigma
                    if (M, p) not in new_index or (N, q) not in new_index:
                        continue
                    factor = cg(a, 1, ap, m, sigma, M) * cg(a, 1, ap, n, sigma, N)
                    total += old[j, i] * new[new_index[N, q], new_index[M, p]] * factor
        return sym.simplify(total / ((ap + 1) * (b + 1) * (s + 1) * (sp + 1)))

    comparisons = 0
    # Include both nontrivial spectator values and all allowed ± transitions.
    states = [(a, b, s) for a, b in product(range(3), repeat=2)
              for s in range(abs(a - b), a + b + 1, 2)]
    for a, b, s in states:
        if b == 0:
            continue
        for ap, bp, sp in states:
            if bp != b or abs(ap - a) != 1 or abs(sp - s) != 1:
                continue
            expected = haar_bilinear(a, b, s, ap, sp)
            actual = racah_sixj_squared(a, s, b, sp, ap, 1) / (b + 1)
            assert expected == sym.Rational(actual.numerator, actual.denominator)
            comparisons += 1
    assert comparisons >= 20


def test_character_multiplication_is_symmetric_and_bounded() -> None:
    for cutoff in (1, 2, 3):
        basis = physical_basis(cutoff)
        for active in (0, 1):
            matrix = np.zeros((len(basis), len(basis)))
            for i, state in enumerate(basis):
                for j, other in enumerate(basis):
                    spectator = 1 - active
                    if (state[spectator] != other[spectator]
                            or abs(state[active] - other[active]) != 1
                            or abs(state[2] - other[2]) != 1):
                        continue
                    amp = magnetic_sixj_amplitude(state[active], state[2],
                                                   state[spectator], other[active], other[2])
                    matrix[i, j] = amp.mid
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-14, rtol=1e-14)
            assert np.max(np.abs(np.linalg.eigvalsh(matrix))) <= 2 + 1e-12
