# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
import random
from dataclasses import replace
from fractions import Fraction as Q

import pytest
from omnibias.core.realization.algebraic import RealAlgebraicField, root_count
from omnibias.core.realization.membership import (
    classify_binary_quadratic,
    classify_linear,
    classify_scalar_quadratic,
)
from omnibias.core.realization.polynomial import (
    AlgebraBudget,
    AlgebraBudgetExceeded,
    PolynomialNetworkSpec,
    SparsePolynomial,
    compile_coefficient_map,
)
from omnibias.core.realization.rank import certify_generic_rank, certify_matrix_rank
from omnibias.core.realization.witness import LaurentPolynomial, witness_from_payload


def _forward(spec, parameters, x):
    offset = 0
    values = tuple(x)
    for layer, (n, m) in enumerate(zip(spec.dims[:-1], spec.dims[1:], strict=True)):
        w = [parameters[offset + i * n : offset + (i + 1) * n] for i in range(m)]
        offset += n * m
        b = parameters[offset : offset + m] if spec.biases else (Q(0),) * m
        offset += m if spec.biases else 0
        values = tuple(
            sum(a * v for a, v in zip(row, values, strict=True)) + z
            for row, z in zip(w, b, strict=True)
        )
        if layer < len(spec.activations):
            values = tuple(
                sum(c * v**k for k, c in enumerate(spec.activations[layer])) for v in values
            )
    return values


@pytest.mark.parametrize(
    "spec",
    [
        PolynomialNetworkSpec((2, 2, 1), ((Q(1, 3), Q(-2, 5), 1),)),
        PolynomialNetworkSpec((1, 2, 1, 1), ((1, 0, 1), (0, 1, Q(1, 2)))),
        PolynomialNetworkSpec.monomial((2, 2, 2), 3),
        PolynomialNetworkSpec((1, 1, 1), ((0,),), False),
    ],
)
def test_compiler_matches_independent_dense_rational_forward(spec):
    source = compile_coefficient_map(spec)
    rng = random.Random(842)
    for _ in range(8):
        theta = tuple(Q(rng.randrange(-3, 4), rng.randrange(1, 4)) for _ in range(spec.n_params))
        x = tuple(Q(rng.randrange(-3, 4), 3) for _ in range(spec.dims[0]))
        coeffs = source.evaluate(theta)
        actual = tuple(
            sum(c * _monomial(x, idx) for c, idx in zip(row, source.input_indices, strict=True))
            for row in coeffs
        )
        assert actual == _forward(spec, theta, x)


def _monomial(x, powers):
    value = Q(1)
    for z, p in zip(x, powers, strict=True):
        value *= z**p
    return value


def test_exact_type_and_budget_boundaries():
    with pytest.raises(TypeError):
        SparsePolynomial(1, {(1,): 0.1})
    with pytest.raises(TypeError):
        PolynomialNetworkSpec((1, 1, 1), ((False,),))
    with pytest.raises(AlgebraBudgetExceeded):
        compile_coefficient_map(
            PolynomialNetworkSpec.monomial((2, 3, 1)), budget=AlgebraBudget(max_terms=3)
        )
    with pytest.raises(AlgebraBudgetExceeded):
        compile_coefficient_map(
            PolynomialNetworkSpec.monomial((1, 1, 1), 5), budget=AlgebraBudget(max_degree=2)
        )


@pytest.mark.parametrize(
    "a,rank",
    [
        ([[0, 0], [0, 0]], 0),
        ([[1, 2, 3], [2, 4, 6]], 1),
        ([[0, 2, 1], [2, 0, -1]], 2),
        ([[1, 0], [0, 1]], 2),
    ],
)
def test_source_rank_factorization_and_minor(a, rank):
    witness = certify_matrix_rank(a)
    assert witness.rank == rank and witness.verify()
    changed = (
        tuple(x + (1 if i == 0 else 0) for i, x in enumerate(witness.source[0])),
        *witness.source[1:],
    )
    assert not replace(witness, source=changed).verify()
    assert not replace(witness, minor_value=witness.minor_value + 1).verify()
    assert not replace(witness, source=()).verify()
    assert not replace(witness, source=((Q(1),), ())).verify()
    assert not replace(witness, rank=True).verify()


def test_generic_rank_uses_symbolic_minors_not_one_singular_sample():
    x = SparsePolynomial.variable(2, 0)
    y = SparsePolynomial.variable(2, 1)
    report = certify_generic_rank((x * x, x * y), (1, 2))
    assert report.complete and report.lower == report.upper == 2
    singular = certify_generic_rank((x * x, x * y), (0, 0))
    assert singular.lower >= 1 and singular.upper == 2
    dependent = certify_generic_rank((x * x, 2 * x * x), (1, 0))
    assert dependent.complete and dependent.lower == dependent.upper == 1


def test_algebraic_root_selection_equality_sign_and_inverse():
    assert root_count((-2, 0, 1), 1, 2) == 1
    with pytest.raises(ValueError):
        RealAlgebraicField((-2, 0, 1), -2, 2)
    with pytest.raises(ValueError):
        RealAlgebraicField((1, -2, 1), 0, 2)
    field = RealAlgebraicField.sqrt(2)
    t = field.generator
    assert (t * t - 2).is_zero()
    assert (t - Q(7, 5)).sign() == 1
    assert (t - Q(3, 2)).sign() == -1
    assert (t * t.inverse() - 1).is_zero()
    assert (RealAlgebraicField.sqrt(Q(9, 4)).generator - Q(3, 2)).is_zero()
    # Reducible primitives select a root; equality is not coefficient equality.
    reducible = RealAlgebraicField((0, -1, 0, 1), Q(1, 2), Q(3, 2))
    assert (reducible.generator - 1).is_zero()
    with pytest.raises(ZeroDivisionError):
        (reducible.generator + 1).inverse()


@pytest.mark.parametrize("hidden", [(), (2,), (3, 2, 4)])
def test_linear_membership_arbitrary_depth(hidden):
    report = classify_linear([[1, 2, 3], [-1, 2, 0]], hidden)
    assert report.status == "realizable" and report.witness.verify(report.source)
    assert not replace(report.witness, target=tuple(t + 1 for t in report.target)).verify(
        report.source
    )
    assert classify_linear([[1, 0], [0, 1]], (1, 3)).status == "excluded"


def test_scalar_quadratic_congruence_handles_zero_diagonal_and_rank():
    rng = random.Random(529)
    for n in (2, 3, 4):
        for _ in range(7):
            a = [[Q(rng.randrange(-3, 4)) for _ in range(n)] for _ in range(n)]
            a = [[a[min(i, j)][max(i, j)] for j in range(n)] for i in range(n)]
            report = classify_scalar_quadratic(a, n)
            assert report.status == "realizable" and report.witness.verify(report.source)
    assert classify_scalar_quadratic([[0, 1], [1, 0]], 2).witness is not None
    assert classify_scalar_quadratic([[1, 0], [0, 1]], 1).status == "excluded"


@pytest.mark.parametrize(
    "c,status",
    [
        ([[1, 0, 0], [0, 1, 0]], "closure_only"),
        ([[0, 0, 1], [0, 1, 0]], "closure_only"),
        ([[1, 0, -1], [0, 1, 0]], "excluded"),
        ([[1, 0, 2], [0, 1, 0]], "realizable"),
        ([[1, 0, 0], [0, 0, 1]], "realizable"),
        ([[0, 1, 0], [0, 2, 0]], "realizable"),
        ([[0, 0, 0], [0, 0, 0]], "realizable"),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], "excluded"),
        ([[1, 0, 0], [0, 1, 0], [2, -3, 0]], "closure_only"),
    ],
)
def test_complete_binary_quadratic_real_vs_closure(c, status):
    report = classify_binary_quadratic(c)
    assert report.status == status
    if report.witness:
        assert report.witness.verify(report.source)
    if report.closure_path:
        path = report.closure_path
        assert path.verify(report.source)
        error = path.coefficient_error_bounds(report.source, Q(1, 10))
        theta = tuple(p.evaluate(Q(1, 10)) for p in path.parameters)
        actual = tuple(x for row in report.source.evaluate(theta) for x in row)
        assert all(abs(a - b) <= e for a, b, e in zip(actual, report.target, error, strict=True))
        assert not replace(path, target=tuple(x + 1 for x in path.target)).verify(report.source)
        divergent = replace(path, parameters=(LaurentPolynomial({-1: 1}), *path.parameters[1:]))
        assert not divergent.verify(report.source)


def test_random_real_binary_networks_are_never_excluded():
    rng = random.Random(802)
    for _ in range(16):
        a, b, c, d = [Q(rng.randrange(-3, 4)) for _ in range(4)]
        rows = []
        for _ in range(3):
            u, v = [Q(rng.randrange(-2, 3)) for _ in range(2)]
            rows.append([u * a * a + v * c * c, 2 * (u * a * b + v * c * d), u * b * b + v * d * d])
        report = classify_binary_quadratic(rows)
        assert report.status == "realizable" and report.witness.verify(report.source)


def test_portable_rational_algebraic_and_closure_witnesses_bind_source():
    import json

    for report in (
        classify_linear([[1, 2], [0, 1]], (2,)),
        classify_binary_quadratic([[1, 0, 2], [0, 1, 0]]),
        classify_binary_quadratic([[1, 0, 0], [0, 1, 0]]),
    ):
        witness = report.witness or report.closure_path
        payload = json.loads(json.dumps(witness.to_payload()))
        assert witness_from_payload(payload, report.source).verify(report.source)
        payload["target"][0] = [100, 1]
        with pytest.raises(ValueError):
            witness_from_payload(payload, report.source)


def test_binary_two_output_complex_closure_has_full_ambient_dimension():
    # Full Jacobian rank six independently anchors the full-space ED endpoint.
    source = compile_coefficient_map(PolynomialNetworkSpec.monomial((2, 2, 2)))
    report = certify_generic_rank(source.polynomials, (1, 0, 0, 1, 1, 0, 0, 1))
    assert len(source.polynomials) == 6
    assert report.complete and report.lower == report.upper == 6
