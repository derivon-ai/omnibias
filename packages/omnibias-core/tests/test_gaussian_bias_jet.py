# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent formal-composition checks for exact joint Gaussian series."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from math import factorial

import pytest
from omnibias.core._rational_linear import RationalLDLT
from omnibias.core.gaussian_bias_jet import (
    GaussianBiasInverseSeries,
    _build_equal_logits,
    gaussian_bias_inverse_series,
)
from omnibias.core.gaussian_jet import GaussianInverseSeries, _Arithmetic
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _scalar():
    return GaussianHeatMap(((-1,), (1,)), (F(1, 4),)*2)


def _multiply(left, right, degree):
    output = [F(0)]*(degree+1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if i+j <= degree:
                output[i+j] += a*b
    return output


def _exp_zero(poly, degree):
    assert poly[0] == 0
    total, power = [F(1)]+[F(0)]*degree, [F(1)]+[F(0)]*degree
    for k in range(1, degree+1):
        power = _multiply(power, poly, degree)
        total = [a+b/F(factorial(k)) for a, b in zip(total, power, strict=True)]
    return total


def _inverse(poly, degree):
    small = [F(0)]+[-value/poly[0] for value in poly[1:]]
    total, power = [F(1)]+[F(0)]*degree, [F(1)]+[F(0)]*degree
    for _ in range(degree):
        power = _multiply(power, small, degree)
        total = [a+b for a, b in zip(total, power, strict=True)]
    return [value/poly[0] for value in total]


def _posterior_by_full_gaussian_composition(model, x, bias, degree):
    # Independently expand the original squared distances, including the
    # common x(t)^2 term. Exponentials use finite factorial powers, and
    # normalization uses a geometric reciprocal rather than the recurrence.
    weights = []
    for j, center in enumerate(model.centers):
        energy = [F(bias[k][j]) for k in range(degree+1)]
        for axis, origin in enumerate(center):
            delta = [F(x[k][axis]) for k in range(degree+1)]
            delta[0] -= origin
            square = _multiply(delta, delta, degree)
            energy = [a-b/(4*model.viscosity*model.time) for a, b in zip(energy, square, strict=True)]
        energy[0] = F(0)  # Every base logit is equal; cancel that common factor.
        weights.append(_exp_zero(energy, degree))
    partition = [sum(row[k] for row in weights) for k in range(degree+1)]
    reciprocal = _inverse(partition, degree)
    probabilities = [_multiply(row, reciprocal, degree) for row in weights]
    return tuple(tuple(row[k] for row in probabilities) for k in range(degree+1))


def _eliminate(matrix, rhs):
    rows = [[F(value) for value in row]+[F(value)] for row, value in zip(matrix, rhs, strict=True)]
    dimension = len(rows)
    for j in range(dimension):
        pivot = next(i for i in range(j, dimension) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [value/scale for value in rows[j]]
        for i in range(dimension):
            if i != j:
                multiple = rows[i][j]
                rows[i] = [a-multiple*b for a, b in zip(rows[i], rows[j], strict=True)]
    return tuple(row[-1] for row in rows)


def _independent_series(model, point, targets, biases):
    dimension, count, degree = model.dimension, model.mode_count, len(targets)-1
    coefficients = [tuple(map(F, point))]
    for n in range(1, degree+1):
        def residual(trial, n=n):
            x = (*coefficients, trial)
            probabilities = _posterior_by_full_gaussian_composition(model, x, biases, n)
            return tuple(trial[axis]-model.time*targets[n][axis]
                         -sum(probabilities[n][j]*model.centers[j][axis] for j in range(count))
                         for axis in range(dimension))

        zero = residual((F(0),)*dimension)
        columns = [tuple(a-b for a, b in zip(residual(tuple(F(int(i == j)) for i in range(dimension))), zero, strict=True))
                   for j in range(dimension)]
        matrix = tuple(tuple(column[i] for column in columns) for i in range(dimension))
        coefficients.append(_eliminate(matrix, tuple(-value for value in zero)))
    return tuple(coefficients), _posterior_by_full_gaussian_composition(model, coefficients, biases, degree)


def test_cubic_contact_fixture_has_zero_first_and_second_orders_but_nonzero_third():
    model = _scalar()
    result = gaussian_bias_inverse_series(model, (0,), ((-1,),)+((0,),)*6, ((0, 2),)+((0, 0),)*6)
    assert result.coefficients == ((0,), (0,), (0,), (F(-2, 3),), (0,), (F(14, 15),), (0,), (F(-538, 315),))
    assert result.target_coefficients == ((0,), (-1,), *((0,),)*6)
    assert result.bias_coefficients[0] == model.biases
    assert result.relative_bias_coefficients[1] == (0, 2)
    assert result.posterior_coefficients[1] == (F(-1, 2), F(1, 2))
    assert tuple(sum(row) for row in result.posterior_coefficients) == (1, 0, 0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("coupled", [False, True])
def test_independent_original_gaussian_composition_matches_every_coefficient(coupled):
    if coupled:
        model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (0, F(1, 4), F(1, 4)))
        point = (0, 0)
        targets = ((1, -2), (F(1, 3), F(1, 2)), (0, 1), (0, 0), (F(-1, 7), 0))
        biases = ((0, 1, -1), (F(1, 3), 0, 2), (0, -1, 0), (1, 2, 3), (0, 0, 0))
    else:
        model, point = _scalar(), (0,)
        targets = ((-1,), (0,), (F(1, 4),), (0,), (0,))
        biases = ((0, 2), (F(2, 3), F(-1, 2)), (0, 0), (0, 1), (0, 0))
    result = gaussian_bias_inverse_series(model, point, targets, biases)
    coefficients, posterior = _independent_series(model, point, result.target_coefficients, result.bias_coefficients)
    assert result.coefficients == coefficients
    assert result.posterior_coefficients == posterior


def test_independent_high_precision_original_inverse_agrees_to_the_expected_truncation_order():
    mp = pytest.importorskip("mpmath")
    result = gaussian_bias_inverse_series(_scalar(), (0,), ((-1,),)+((0,),)*6, ((0, 2),)+((0, 0),)*6)
    with mp.workdps(90):
        errors = []
        for t in (mp.mpf("0.01"), mp.mpf("0.005")):
            root = mp.findroot(lambda x, t=t: x+t-mp.tanh(x/2+t), (0, -t**3))
            polynomial = sum(mp.mpf(row[0].numerator)/row[0].denominator*t**n for n, row in enumerate(result.coefficients))
            errors.append(abs(root-polynomial))
        assert 490 < errors[0]/errors[1] < 520  # First omitted nonzero term is order nine.
        assert errors[0] < mp.mpf("1e-16")


def test_zero_bias_path_exactly_matches_existing_fixed_bias_series():
    model = GaussianHeatMap(((0, 0), (1, 0), (0, 1)), (0, F(1, 4), F(1, 4)))
    target = ((1, 2), (F(-1, 3), F(2, 7)), (0, 0), (0, 0), (1, 0))
    old = GaussianInverseSeries.from_equal_logits(model, (0, 0), target)
    result = gaussian_bias_inverse_series(model, (0, 0), target, ((0, 0, 0),)*len(target))
    assert result.target_coefficients == old.target_coefficients
    assert result.coefficients == old.coefficients


def test_exact_common_bias_and_spatial_gauges_do_not_change_higher_coefficients():
    order, shift, gauge = 5, 2**2500, 2**3000
    targets = ((-1,),)+((0,),)*(order-1)
    bias = ((0, 2),)+((0, 0),)*(order-1)
    baseline = gaussian_bias_inverse_series(_scalar(), (0,), targets, bias)
    shifted = GaussianHeatMap(((shift-1,), (shift+1,)), (gauge+F(1, 4),)*2)
    gauged_bias = tuple(tuple(value+gauge+n for value in row) for n, row in enumerate(bias))
    result = gaussian_bias_inverse_series(shifted, (shift,), targets, gauged_bias)
    assert result.coefficients[0] == (shift,) and result.coefficients[1:] == baseline.coefficients[1:]
    assert result.posterior_coefficients == baseline.posterior_coefficients
    assert result.relative_bias_coefficients == baseline.relative_bias_coefficients
    assert result.target_coefficients == baseline.target_coefficients


def test_binding_does_not_square_a_huge_common_base_distance():
    # Identical centers make equal logits independent of the base point;
    # absolute distance squared would exceed the exact bit limit here.
    point = 2**2500
    model = GaussianHeatMap(((0,), (0,)), (2**3000,)*2)
    result = gaussian_bias_inverse_series(model, (point,), ((1,), (F(2, 3),)), ((0, 1), (1, -1)))
    assert result.target_coefficients[0] == (point,)
    assert result.coefficients == ((point,), (1,), (F(2, 3),))
    assert tuple(sum(row) for row in result.posterior_coefficients) == (1, 0, 0)


def test_one_reused_factor_and_one_solve_per_order_with_replayed_solution(monkeypatch):
    module = import_module("omnibias.core.gaussian_bias_jet")
    original_factor, original_solve = module.factor_positive_definite, RationalLDLT.solve
    factors, solves = [], []

    def factor(matrix, arithmetic):
        result = original_factor(matrix, arithmetic)
        factors.append(result)
        return result

    def solve(self, rhs, arithmetic):
        solves.append(self)
        return original_solve(self, rhs, arithmetic)

    monkeypatch.setattr(module, "factor_positive_definite", factor)
    monkeypatch.setattr(RationalLDLT, "solve", solve)
    result = gaussian_bias_inverse_series(_scalar(), (0,), ((-1,),)+((0,),)*6, ((0, 2),)+((0, 0),)*6)
    assert len(factors) == 1 and len(solves) == result.order == 7
    assert all(item is factors[0] for item in solves)
    monkeypatch.setattr(RationalLDLT, "solve", lambda self, rhs, arithmetic: (F(0),)*len(rhs))
    with pytest.raises(ArithmeticError, match="replay"):
        gaussian_bias_inverse_series(_scalar(), (0,), ((1,),), ((0, 0),))


def test_order_zero_binds_base_without_factorization(monkeypatch):
    module = import_module("omnibias.core.gaussian_bias_jet")
    monkeypatch.setattr(module, "factor_positive_definite", lambda *_: pytest.fail("order zero factored a matrix"))
    model = GaussianHeatMap(((0,), (1,), (3,)), (0, F(1, 32), F(9, 32)), time=8)
    result = gaussian_bias_inverse_series(model, (0,), (), ())
    assert result.order == 0 and result.dimension == 1 and result.mode_count == 3
    assert result.target_coefficients == ((F(-1, 6),),)
    assert result.posterior_coefficients == ((F(1, 3),)*3,)
    assert result.relative_bias_coefficients == (model.biases,)


def test_exact_unequal_logits_and_contraction_boundary_are_refused():
    unequal = GaussianHeatMap(((-1,), (1,)), (F(1, 4), F(1, 4)+F(1, 2**100)))
    with pytest.raises(ValueError, match="exactly equal"):
        gaussian_bias_inverse_series(unequal, (0,), (), ())
    boundary = GaussianHeatMap(((-1,), (1,)), (F(1, 2),)*2, time=F(1, 2))
    assert boundary.contraction_bound == 1
    with pytest.raises(ValueError, match="q<1"):
        gaussian_bias_inverse_series(boundary, (0,), (), ())


def test_snapshot_properties_and_exact_evaluation():
    model = _scalar()
    point, target, bias = [0], [[-1], [0], [0]], [[0, 2], [0, 0], [0, 0]]
    result = gaussian_bias_inverse_series(model, point, target, bias)
    point[0] = target[0][0] = bias[0][0] = 99
    assert result.model is model and result.base_point == (0,)
    assert result.target_coefficients[1] == (-1,) and result.bias_coefficients[1] == (0, 2)
    assert result.evaluate(F(1, 2)) == (F(-1, 12),)
    assert result.evaluate(0) == result.base_point
    with pytest.raises(TypeError):
        GaussianBiasInverseSeries()
    with pytest.raises(FrozenInstanceError):
        result.coefficients = ()
    for value in (True, 0.5, complex(0)):
        with pytest.raises(TypeError):
            result.evaluate(value)


def test_no_transcendental_or_numerical_inverse_dependency(monkeypatch):
    model = _scalar()

    def fail(*args, **kwargs):
        pytest.fail("exact series used a numerical inverse or exponentials")

    for name, functions in (("omnibias.core.verified.inverse_gaussian_heat", ("inverse_gaussian_heat", "certify_gaussian_inverse")),
                            ("omnibias.core.verified.mean_map", ("_probabilities",))):
        module = import_module(name)
        for function in functions:
            monkeypatch.setattr(module, function, fail)
    monkeypatch.setattr(transcend, "exp_iv", fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = gaussian_bias_inverse_series(model, (0,), ((-1,),)*3, ((0, 2),)*3)
    assert result.order == 3 and transcend.strict_backend() and transcend.libm_fallback_used()


@pytest.mark.parametrize("position,bad,error", [
    (0, None, TypeError), (1, None, TypeError), (1, (), ValueError), (1, (0, 0), ValueError),
    (1, (True,), TypeError), (1, (0.0,), TypeError),
    (2, None, TypeError), (2, "", TypeError), (2, ((True,),), TypeError), (2, ((0.0,),), TypeError),
    (2, ((0, 0),), ValueError), (2, (), ValueError),
    (3, None, TypeError), (3, ((0,),), ValueError), (3, ((0, 0.0),), TypeError),
    (3, ((0, True),), TypeError), (3, ((0, complex(0)),), TypeError), (3, (), ValueError),
])
def test_public_input_contracts(position, bad, error):
    arguments = [_scalar(), (0,), ((1,),), ((0, 0),)]
    arguments[position] = bad
    with pytest.raises(error):
        gaussian_bias_inverse_series(*arguments)


@pytest.mark.parametrize("budget", [0, {}, "budget"])
def test_invalid_budget_type(budget):
    with pytest.raises(TypeError):
        gaussian_bias_inverse_series(_scalar(), (0,), (), (), budget=budget)


@pytest.mark.parametrize("limits", [RidgeBudget(max_order=0), RidgeBudget(max_matrix_entries=1), RidgeBudget(max_elimination_work=1)])
def test_resource_preflight_precedes_factorization(limits, monkeypatch):
    model = _scalar()
    module = import_module("omnibias.core.gaussian_bias_jet")
    monkeypatch.setattr(module, "factor_positive_definite", lambda *_: pytest.fail("preflight missed factorization"))
    with pytest.raises(ValueError):
        gaussian_bias_inverse_series(model, (0,), ((1,),), ((0, 0),), budget=limits)


def test_new_budget_checks_original_model_metadata_and_intermediate_growth():
    with pytest.raises(ValueError, match="bit"):
        gaussian_bias_inverse_series(_scalar(), (0,), ((255,),), ((0, 0),), budget=RidgeBudget(max_coefficient_bits=8))
    model = GaussianHeatMap(((-1,), (1,)), (2**100+F(1, 4),)*2)
    with pytest.raises(ValueError, match="bit"):
        gaussian_bias_inverse_series(model, (0,), (), (), budget=RidgeBudget(max_coefficient_bits=64))


def test_private_builder_reuses_the_callers_precharged_meter():
    model = _scalar()
    limits = RidgeBudget(max_elimination_work=10000)
    arithmetic = _Arithmetic(limits)
    arithmetic._work(9000)
    with pytest.raises(ValueError, match="work"):
        _build_equal_logits(model, (0,), ((1,),)*4, ((0, 0),)*4, limits, arithmetic)
    assert arithmetic.used > limits.max_elimination_work
    other = RidgeBudget(max_elimination_work=20000)
    with pytest.raises(ValueError, match="agree"):
        _build_equal_logits(model, (0,), (), (), other, _Arithmetic(limits))


def test_binding_recurrence_and_ldlt_all_share_one_cumulative_counter(monkeypatch):
    model = _scalar()
    observed = []

    def record(self, count):
        observed.append(count)
        assert count <= self.max_elimination_work

    monkeypatch.setattr(RidgeBudget, "_work", record)
    result = gaussian_bias_inverse_series(model, (0,), ((-1,),)+((0,),)*6, ((0, 2),)+((0, 0),)*6)
    assert result.order == 7 and len(observed) > 100
    assert all(a < b for a, b in zip(observed, observed[1:], strict=False))
