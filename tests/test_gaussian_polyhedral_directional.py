# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Critical-cone target derivatives versus exact QPs and nonlinear secants.

The reference QP uses local rational Gaussian elimination, independent of the
core response compiler. Secants enumerate original nonlinear Gaussian KKT
systems and compare one-sided limits, including active-set switches.
"""

from fractions import Fraction as F
from itertools import combinations

import pytest
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_directional import (
    certify_gaussian_polyhedral_directional,
    compile_gaussian_critical_face,
)
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)


def _dot(left, right):
    return sum((a*b for a, b in zip(left, right, strict=True)), F(0))


def _solve(matrix, rhs):
    """Small independent Fraction Gauss-Jordan reference, not a core solve."""
    size = len(rhs)
    rows = [[F(value) for value in row]+[F(value)] for row, value in zip(matrix, rhs, strict=True)]
    for column in range(size):
        pivot = next((j for j in range(column, size) if rows[j][column]), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value/scale for value in rows[column]]
        for j in range(size):
            if j != column:
                scale = rows[j][column]
                rows[j] = [a-scale*b for a, b in zip(rows[j], rows[column], strict=True)]
    return tuple(row[-1] for row in rows)


def _fixture(kind):
    if kind == "coupled":
        source = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0))
        equalities, rhs, target, point = (), (), (0, 0), (0, 0)
        inequalities, upper = ((1, 1), (-1, 0), (0, -1)), (0, 2, 3)
        strong, weak = (), (0,)
    else:
        source = GaussianHeatMap(((-1, -2, 0), (1, 2, 0)), (F(9, 4), F(1, 4)))
        equalities, rhs, target, point = ((0, 1, 0),), (1,), (0, 0, 1), (0, 1, 0)
        inequalities, upper = ((0, 0, 1), (1, 0, 1), (-1, 0, 0)), (0, 0, 2)
        strong, weak = (0,), (1,)
    return source, equalities, rhs, inequalities, upper, target, point, strong, weak


def _equilibrium(kind):
    data = _fixture(kind)
    source, equalities, rhs, inequalities, upper, target, point, strong, _ = data
    base = restrict_gaussian_heat(source, equalities, rhs)
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, inequalities, upper, active_indices=strong)
    assert face is not None
    equilibrium = certify_gaussian_polyhedral_equilibrium(face, target, point)
    assert equilibrium is not None
    return equilibrium, data


def _exact_hessian(source, point):
    scores = tuple(bias-sum((x-a)**2 for x, a in zip(point, center, strict=True))/(4*source.viscosity*source.time)
                   for bias, center in zip(source.biases, source.centers, strict=True))
    assert len(set(scores)) == 1  # Uniform posterior is established independently.
    mean = tuple(sum(row[i] for row in source.centers)/source.mode_count for i in range(source.dimension))
    covariance = tuple(tuple(sum((row[i]-mean[i])*(row[j]-mean[j]) for row in source.centers)/source.mode_count
                             for j in range(source.dimension)) for i in range(source.dimension))
    return tuple(tuple(F(int(i == j))/source.time-covariance[i][j]/(2*source.viscosity*source.time**2)
                       for j in range(source.dimension)) for i in range(source.dimension))


def _critical_qp(data, direction):
    source, equalities, _, inequalities, _, _, point, strong, weak = data
    hessian = _exact_hessian(source, point)
    dimension = source.dimension
    fixed = equalities+tuple(inequalities[i] for i in strong)
    candidates = []
    for size in range(len(weak)+1):
        for chosen in combinations(weak, size):
            constraints = fixed+tuple(inequalities[i] for i in chosen)
            kkt = tuple(tuple(hessian[i])+tuple(row[i] for row in constraints) for i in range(dimension))
            kkt += tuple(tuple(map(F, row))+(F(0),)*len(constraints) for row in constraints)
            solved = _solve(kkt, tuple(direction)+(F(0),)*len(constraints))
            if solved is None:
                continue
            value, multipliers = solved[:dimension], solved[dimension:]
            if any(_dot(inequalities[i], value) > 0 for i in weak):
                continue
            if any(multiplier < 0 for multiplier in multipliers[len(fixed):]):
                continue
            objective = sum(value[i]*_dot(hessian[i], value) for i in range(dimension))/2-_dot(direction, value)
            candidates.append((objective, chosen, value, multipliers))
    assert candidates
    candidates.sort(key=lambda entry: entry[0])
    assert all(candidate[2] == candidates[0][2] for candidate in candidates)
    return candidates[0]


def _mp(mp, value):
    if isinstance(value, (int, float, F)):
        exact = F(value)
        return mp.mpf(exact.numerator)/exact.denominator
    return value


def _original_velocity(mp, source, point):
    age, nu = _mp(mp, source.time), _mp(mp, source.viscosity)
    centers = tuple(tuple(_mp(mp, value) for value in row) for row in source.centers)
    scores = tuple(_mp(mp, bias)-sum((x-a)**2 for x, a in zip(point, row, strict=True))/(4*nu*age)
                   for bias, row in zip(source.biases, centers, strict=True))
    weights = tuple(mp.exp(score-max(scores)) for score in scores)
    mean = tuple(sum(weight*row[i] for weight, row in zip(weights, centers, strict=True))/sum(weights)
                 for i in range(source.dimension))
    return tuple((x-mu)/age for x, mu in zip(point, mean, strict=True))


def _nonlinear_minimum(mp, data, wanted):
    """Enumerate original nonlinear KKT faces, without using critical data."""
    source, equalities, rhs, inequalities, upper, _, _, _, _ = data
    dimension = source.dimension
    feasible = []
    for size in range(dimension-len(equalities)+1):
        for active in combinations(range(len(inequalities)), size):
            constraints = equalities+tuple(inequalities[i] for i in active)
            right = rhs+tuple(upper[i] for i in active)
            if constraints:
                matrix = mp.matrix(constraints)
                gram = matrix*matrix.T
                if mp.det(gram) == 0:
                    continue
                anchor = tuple(matrix.T*mp.lu_solve(gram, mp.matrix(right)))
            else:
                anchor = (mp.mpf(0),)*dimension

            def equations(*state, constraints=constraints, right=right):
                point, multipliers = state[:dimension], state[dimension:]
                velocity = _original_velocity(mp, source, point)
                return tuple(velocity[i]-wanted[i]+sum(row[i]*multiplier for row, multiplier in
                                                      zip(constraints, multipliers, strict=True)) for i in range(dimension)) + tuple(
                    sum(a*x for a, x in zip(row, point, strict=True))-bound for row, bound in zip(constraints, right, strict=True)
                )

            state = tuple(mp.findroot(equations, anchor+(mp.mpf(0),)*len(constraints), tol=mp.eps*32, maxsteps=100))
            assert max(abs(value) for value in equations(*state)) < mp.sqrt(mp.eps)
            point, multipliers = state[:dimension], state[dimension:]
            slack = tuple(bound-sum(a*x for a, x in zip(row, point, strict=True))
                          for row, bound in zip(inequalities, upper, strict=True))
            if min(slack) >= -mp.mpf("1e-75") and min(multipliers[len(equalities):], default=0) >= -mp.mpf("1e-75"):
                feasible.append(point)
    assert feasible
    assert all(max(abs(a-b) for a, b in zip(feasible[0], point, strict=True)) < mp.mpf("1e-70") for point in feasible)
    return feasible[0]


@pytest.mark.parametrize("kind,direction", [
    ("coupled", (1, 0)), ("coupled", (-1, 0)), ("coupled", (0, 1)), ("coupled", (0, -1)),
    ("coupled", (1, -2)), ("coupled", (1, 1)), ("mixed", (1, 0, 0)), ("mixed", (-1, 0, 0)),
])
def test_exact_critical_response_matches_independent_qp_and_nonlinear_one_sided_secants(kind, direction):
    mp = pytest.importorskip("mpmath")
    equilibrium, data = _equilibrium(kind)
    quadratic, selected, expected, multipliers = _critical_qp(data, direction)
    face = compile_gaussian_critical_face(equilibrium, selected)
    assert face is not None
    result = certify_gaussian_polyhedral_directional(face, direction)
    assert result is not None and result.value == expected
    assert result.equality_multipliers == multipliers and result.quadratic_value == quadratic
    source, equalities, _, inequalities, _, target, point, strong, weak = data
    assert face.constraints == equalities+tuple(inequalities[i] for i in strong+selected)
    assert all(_dot(row, direction) >= 0 for row in face.region_inequalities)
    for i in range(len(inequalities)):
        assert result.weak_slacks[i] == (-_dot(inequalities[i], expected) if i in weak else 0)
        assert result.weak_multipliers[i] == (multipliers[len(equalities)+len(strong)+selected.index(i)] if i in selected else 0)
    if kind == "mixed":
        assert source.contraction_bound == F(5, 2)
        assert _exact_hessian(source, point)[1][1] == -1
    with mp.workdps(100):
        errors = []
        for step in (F(1, 10), F(1, 100), F(1, 1000)):
            epsilon = _mp(mp, step)
            wanted = tuple(_mp(mp, value)+epsilon*_mp(mp, delta) for value, delta in zip(target, direction, strict=True))
            root = _nonlinear_minimum(mp, data, wanted)
            secant = tuple((x-_mp(mp, base))/epsilon for x, base in zip(root, point, strict=True))
            errors.append(mp.sqrt(sum((value-_mp(mp, exact))**2 for value, exact in zip(secant, expected, strict=True))))
        assert errors[-1] < mp.mpf("2e-6")
        if errors[0] > mp.mpf("1e-50"):
            assert errors[1] < errors[0]/50 and errors[2] < errors[1]/50


def test_coupled_weak_face_is_positively_homogeneous_but_not_additive():
    equilibrium, _ = _equilibrium("coupled")
    active = compile_gaussian_critical_face(equilibrium, (0,))
    free = compile_gaussian_critical_face(equilibrium, ())
    positive = certify_gaussian_polyhedral_directional(active, (1, 0))
    negative = certify_gaussian_polyhedral_directional(free, (-1, 0))
    zero = certify_gaussian_polyhedral_directional(active, (0, 0))
    assert positive.value == (F(2, 3), F(-2, 3)) and negative.value == (-2, 0)
    assert zero.value == (0, 0)
    assert tuple(a+b for a, b in zip(positive.value, negative.value, strict=True)) != zero.value
    for scale in (F(0), F(1, 3), F(2), F(7, 2)):
        result = certify_gaussian_polyhedral_directional(active, (scale, 0))
        assert result is not None
        assert result.value == tuple(scale*value for value in positive.value)
        assert result.weak_multipliers == tuple(scale*value for value in positive.weak_multipliers)
        assert result.weak_slacks == tuple(scale*value for value in positive.weak_slacks)
        assert result.quadratic_value == scale**2*positive.quadratic_value


def test_boundary_direction_has_the_same_exact_value_from_both_adjacent_faces():
    equilibrium, _ = _equilibrium("coupled")
    active = compile_gaussian_critical_face(equilibrium, (0,))
    free = compile_gaussian_critical_face(equilibrium, ())
    on_active = certify_gaussian_polyhedral_directional(active, (1, -2))
    on_free = certify_gaussian_polyhedral_directional(free, (1, -2))
    assert on_active is not None and on_free is not None
    assert on_active.value == on_free.value == (2, -2)
    assert on_active.weak_multipliers == on_free.weak_multipliers == (0, 0, 0)
    assert on_active.weak_slacks == on_free.weak_slacks == (0, 0, 0)
    assert certify_gaussian_polyhedral_directional(active, (-1, 0)) is None
    assert certify_gaussian_polyhedral_directional(free, (1, 0)) is None


def test_mixed_strong_weak_geometry_removes_large_exact_base_normal_directions():
    equilibrium, data = _equilibrium("mixed")
    selected = compile_gaussian_critical_face(equilibrium, (1,))
    free = compile_gaussian_critical_face(equilibrium, ())
    assert selected.constraints == ((0, 1, 0), (0, 0, 1), (1, 0, 1))
    assert free.constraints == ((0, 1, 0), (0, 0, 1))
    result = certify_gaussian_polyhedral_directional(free, (-1, 2**200, -(2**100)))
    assert result is not None and result.value == (-2, 0, 0)
    assert result.value == _critical_qp(data, (-1, 2**200, -(2**100)))[2]
    assert result.weak_slacks == (0, 2, 0)
    assert result.weak_multipliers == (0, 0, 0)
    clamped = certify_gaussian_polyhedral_directional(selected, (1, 0, 0))
    assert clamped is not None and clamped.value == (0, 0, 0)
