# SPDX-License-Identifier: Apache-2.0
"""Independent exact original-edge tensor normalization and residual identities.

No production source-gate imports. The complete finite Fourier supports are
verified by sparse singular-value polynomial identities and direct analytic
matrix insertions; no sampled truncation is used as a spectral proof.
"""

from __future__ import annotations

from fractions import Fraction as Q
from itertools import pairwise, product
from typing import Any

import pytest
import sympy as sp  # type: ignore[import-untyped]
from sympy.physics.wigner import wigner_3j  # type: ignore[import-untyped]


def strip(n: int) -> Any:
    edges = [(("b", i), ("b", i + 1)) for i in range(n)]
    edges += [(("t", i), ("t", i + 1)) for i in range(n)]
    edges += [(("b", i), ("t", i)) for i in range(n + 1)]
    return edges


def vertex_tensor(labels: Any, incoming: Any) -> Any:
    ranges = [range(int(2 * j + 1)) for j in labels]
    out = []
    for indices in product(*ranges):
        ms = [j - i for j, i in zip(labels, indices, strict=True)]
        if len(labels) == 2:
            assert labels[0] == labels[1]
            coefficient = (-1) ** int(labels[0] - ms[0]) if ms[0] == -ms[1] else 0
        elif len(labels) == 3:
            coefficient = wigner_3j(*labels, *ms)
        else:
            raise ValueError("only the complete bivalent/trivalent local supports are used")
        if not coefficient:
            continue
        converted = list(indices)
        for k, is_incoming in enumerate(incoming):
            if is_incoming:
                coefficient *= (-1) ** int(labels[k] - ms[k])
                converted[k] = int(2 * labels[k]) - indices[k]
        out.append((tuple(converted), sp.simplify(coefficient)))
    return out


def coefficient_matrix(edges: Any, spins: Any) -> Any:
    # Remove trivial legs before constructing the unique invariant tensors.
    active = [e for e, s in enumerate(spins) if s]
    dims = [int(2 * spins[e] + 1) for e in active]
    vertices = sorted({v for e in active for v in edges[e]})
    factors = []
    for vertex in vertices:
        local = [e for e in active if vertex in edges[e]]
        incoming = [edges[e][1] == vertex for e in local]
        factors.append((local, incoming, vertex_tensor([spins[e] for e in local], incoming)))
    rows = {}
    for chosen in product(*(factor[2] for factor in factors)):
        rr: dict[int, int] = {}
        cc: dict[int, int] = {}
        coefficient = sp.Integer(1)
        for (local, incoming, _), (indices, value) in zip(factors, chosen, strict=True):
            coefficient *= value
            for edge, is_incoming, index in zip(local, incoming, indices, strict=True):
                (cc if is_incoming else rr)[edge] = index
        row = col = 0
        for edge, dim in zip(active, dims, strict=True):
            row = row * dim + rr[edge]
            col = col * dim + cc[edge]
        rows[row, col] = sp.simplify(coefficient)
    size = sp.prod(dims)
    matrix = sp.MutableSparseMatrix(size, size, rows)
    # Inserting a trivial shared leg changes the normalization at its two
    # now-bivalent vertices: theta b_(a,b,0) uses normalized tensors there.
    for vertex in vertices:
        original = [e for e, s in enumerate(spins) if vertex in edges[e]]
        local = [e for e in active if vertex in edges[e]]
        if len(original) == 3 and len(local) == 2:
            matrix /= sp.sqrt(2 * spins[local[0]] + 1)
    diagonal = sp.simplify(sp.trace(matrix))
    expected = int(2 * spins[active[0]] + 1) if len(edges) == 4 else 1
    assert abs(diagonal) == expected, (diagonal, expected)
    matrix *= expected / diagonal
    return sp.SparseMatrix(matrix), active, dims


def point(index: int) -> Any:
    pauli = (sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.diag(1, -1))
    z = sp.Rational(index + 1, index + 3)
    return (1 - z * z) / (1 + z * z) * sp.eye(2) + sp.I * 2 * z / (1 + z * z) * pauli[index % 3]


def representation(matrix: Any, spin: Any) -> Any:
    if spin == sp.Rational(1, 2):
        return matrix
    assert spin == 1
    cg = sp.Matrix([[1, 0, 0], [0, 1 / sp.sqrt(2), 0], [0, 1 / sp.sqrt(2), 0], [0, 0, 1]])
    return sp.simplify(cg.H * sp.kronecker_product(matrix, matrix) * cg)


def evaluate(matrix: Any, active: Any, dims: Any, spins: Any, links: Any) -> Any:
    reps = {e: representation(links[e], spins[e]) for e in active}

    def decode(index: int) -> list[int]:
        values = []
        for d in reversed(dims):
            values.append(index % d)
            index //= d
        return list(reversed(values))

    total = 0
    for (r, c), value in matrix.todok().items():
        rr, cc = decode(r), decode(c)
        total += value * sp.prod(reps[e][i, j] for e, i, j in zip(active, rr, cc, strict=True))
    return sp.simplify(total)


def line_distance(edges: Any, source: int, target: int) -> int:
    frontier = [source]
    seen = {source}
    distance = 0
    while target not in frontier:
        frontier = [
            j
            for i in frontier
            for j in range(len(edges))
            if j not in seen and set(edges[i]) & set(edges[j])
        ]
        seen.update(frontier)
        distance += 1
        assert frontier
    return distance


@pytest.mark.parametrize("spin, nuclear", [(sp.Rational(1, 2), 8), (sp.Integer(1), 27)])
def test_complete_square_fourier_matrix_has_exact_singular_values(spin: Any, nuclear: int) -> None:
    edges = strip(1)
    spins = [spin] * 4
    coefficient, active, dims = coefficient_matrix(edges, spins)
    gram = coefficient * coefficient.H
    dimension = int(2 * spin + 1)
    assert (gram * gram - dimension**2 * gram).applyfunc(sp.simplify) == sp.zeros(gram.rows)
    assert sp.trace(gram) == dimension**4
    assert dimension**2 * dimension == nuclear
    for shift in (0, 2):
        links = [point(e + shift) for e in range(4)]
        holonomy = links[0] * links[3] * links[1].H * links[2].H
        expected = sp.trace(representation(holonomy, spin))
        assert sp.simplify(evaluate(coefficient, active, dims, spins, links) - expected) == 0


@pytest.mark.parametrize("shared", [sp.Integer(0), sp.Integer(1)])
def test_strip_pair_coefficients_are_rank_sixteen_partial_isometries(shared: Any) -> None:
    edges = strip(2)
    spins = [sp.Rational(1, 2)] * 7
    spins[5] = shared
    coefficient, active, dims = coefficient_matrix(edges, spins)
    gram = coefficient * coefficient.H
    assert (gram * gram - gram).applyfunc(sp.simplify) == sp.zeros(gram.rows)
    assert sp.trace(gram) == 16
    assert Q(16, int(sp.prod(dims))) == Q(1, 4 * int(2 * shared + 1))
    for shift in (0, 2):
        links = [point(e + shift) for e in range(7)]
        x = links[0].H * links[4] * links[2]
        y = links[1] * links[6] * links[3].H
        z = links[5]
        outer = sp.trace(x * y.H)
        expected = (
            outer / 2 if shared == 0 else (sp.trace(x * z.H) * sp.trace(y * z.H) - outer / 2) / 3
        )
        assert sp.simplify(evaluate(coefficient, active, dims, spins, links) - expected) == 0
    assert max(line_distance(edges, i, j) for i in active for j in active) == 3


def _perpendicular_pair(first_sign: int, second_sign: int) -> Any:
    origin, end = (0, 0, 0), (0, 0, 1)
    edges = []
    for axis, sign in ((0, first_sign), (1, second_sign)):
        lower = [0, 0, 0]
        lower[axis] = sign
        upper = list(lower)
        upper[2] = 1
        path = [origin, tuple(lower), tuple(upper), end]
        for a, b in pairwise(path):
            edges.append(tuple(sorted((a, b))))
    edges.append((origin, end))
    return edges


@pytest.mark.parametrize("signs", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
@pytest.mark.parametrize("shared", [sp.Integer(0), sp.Integer(1)])
def test_all_perpendicular_orientation_patterns_obey_original_coefficient_bound(
    signs: tuple[int, int],
    shared: Any,
) -> None:
    edges = _perpendicular_pair(*signs)
    spins = [sp.Rational(1, 2)] * 6 + [shared]
    coefficient, active, _ = coefficient_matrix(edges, spins)
    gram = coefficient * coefficient.H
    same_side = signs[0] == signs[1]
    squared_singular = (
        sp.Rational(4, 3)
        if same_side and shared == 1
        else (sp.Integer(4) if same_side else sp.Integer(1))
    )
    rank = 12 if same_side and shared == 1 else (4 if same_side else 16)
    assert (gram * gram - squared_singular * gram).applyfunc(sp.simplify) == sp.zeros(gram.rows)
    assert sp.trace(gram) == rank * squared_singular == 16
    assert rank**2 * squared_singular <= 16**2
    assert max(line_distance(edges, i, j) for i in active for j in active) <= 3


def test_exact_shared_generator_insertions_match_both_residual_channels() -> None:
    pauli = (sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.diag(1, -1))
    for shift in (0, 1, 3):
        x, y, z = (point(shift + i) for i in range(3))
        xp, yq, outer = sp.trace(x * z.H), sp.trace(y * z.H), sp.trace(x * y.H)
        cross = sum(
            (sp.trace(-x * sp.I * t * z.H / 2) * sp.trace(-y * sp.I * t * z.H / 2) for t in pauli),
            sp.Integer(0),
        )
        singlet, triplet = outer / 2, (xp * yq - outer / 2) / 3
        assert sp.simplify(cross - sp.Rational(3, 4) * (singlet - triplet)) == 0
        assert sp.simplify(cross - outer / 2 + xp * yq / 4) == 0
        assert Q(9, 2) / 27 == Q(13, 2) / 39 == Q(1, 6)
        # C R_2 equals the nonconstant Gamma source, including the cross factor2/9.
        assert sp.simplify(singlet / 6 - triplet / 6 - sp.Rational(2, 9) * cross) == 0
        diagonal = sum((sp.trace(-x * sp.I * t * z.H / 2) ** 2 for t in pauli), sp.Integer(0))
        assert sp.simplify(4 * diagonal - 4 + xp**2) == 0
        assert Q(8) / 72 == Q(1, 9)


def _cubic_faces(length: int) -> list[frozenset[Any]]:
    faces = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for base in product(range(length + 1), repeat=3):
            if base[a] == length or base[b] == length:
                continue
            aa, bb, both = list(base), list(base), list(base)
            aa[a] += 1
            bb[b] += 1
            both[a] += 1
            both[b] += 1
            vertices = [base, tuple(aa), tuple(both), tuple(bb)]
            faces.append(
                frozenset(tuple(sorted((vertices[i], vertices[(i + 1) % 4]))) for i in range(4))
            )
    return faces


def test_cubic_anchor_pair_bound_counts_every_pair_once() -> None:
    for length in (1, 2, 4):
        faces = _cubic_faces(length)
        edges = set().union(*faces)
        pairs = [
            (p, q)
            for p in range(len(faces))
            for q in range(p + 1, len(faces))
            if faces[p] & faces[q]
        ]
        assert all(len(faces[p] & faces[q]) == 1 for p, q in pairs)
        maximum = 0
        for edge in edges:
            q_e = sum(edge in face for face in faces)
            touching = sum(edge in faces[p] | faces[q] for p, q in pairs)
            assert q_e <= 4
            assert touching <= 12 * q_e - q_e * (q_e - 1) // 2 <= 42
            maximum = max(maximum, touching)
        if length == 4:
            assert maximum == 42


def test_strip_anchor_norm_is_exact_on_a_four_square_interior_vertical() -> None:
    for n in range(1, 7):
        edges = strip(n)
        faces = [frozenset((i, n + i, 2 * n + i, 2 * n + i + 1)) for i in range(n)]
        pairs = [(i, i + 1) for i in range(n - 1)]
        base = Q(5, 4)
        rows = []
        for edge in range(len(edges)):
            self_count = sum(edge in face for face in faces)
            pair_count = sum(edge in faces[i] | faces[j] for i, j in pairs)
            rows.append(3 * self_count * base**2 + Q(8, 3) * pair_count * base**3)
        assert max(rows) <= 6 * base**2 + 8 * base**3
        if n >= 4:
            assert max(rows) == 6 * base**2 + 8 * base**3
