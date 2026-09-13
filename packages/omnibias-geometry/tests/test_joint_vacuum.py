# SPDX-License-Identifier: Apache-2.0
"""Independent finite algebra checks; these samples are not analytic proofs."""

from __future__ import annotations

from fractions import Fraction as Q
from math import comb, sqrt
from random import Random
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from omnibias.geometry.gauge.transfer.joint_vacuum import (
    ThetaState,
    su2_theta_norm,
    su2_theta_product_characters,
    su2_theta_product_reference_bound,
    su2_theta_vacuum_tail,
)

Matrix = NDArray[np.complex128]
BiPolynomial = dict[tuple[int, int], Q]


def _spin_generators(n: int) -> tuple[Matrix, Matrix, Matrix]:
    j = n / 2
    raising = np.zeros((n + 1, n + 1), dtype=np.complex128)
    for k in range(n):
        raising[k + 1, k] = sqrt((n - k) * (k + 1))
    lowering = raising.conj().T
    return (
        (raising + lowering) / 2,
        (raising - lowering) / (2j),
        np.diag(np.arange(n + 1, dtype=float) - j).astype(np.complex128),
    )


def _projectors(a: int, b: int) -> dict[int, Matrix]:
    ia, ib = np.eye(a + 1), np.eye(b + 1)
    identity = np.eye((a + 1) * (b + 1), dtype=np.complex128)
    total = sum(
        (
            (np.kron(x, ib) + np.kron(ia, y)) @ (np.kron(x, ib) + np.kron(ia, y))
            for x, y in zip(_spin_generators(a), _spin_generators(b), strict=True)
        ),
        np.zeros_like(identity),
    )
    labels = list(range(abs(a - b), a + b + 1, 2))
    result = {}
    for c in labels:
        projector = identity.copy()
        for other in labels:
            if other != c:
                projector = projector @ (
                    (total - other * (other + 2) / 4 * identity)
                    / (c * (c + 2) / 4 - other * (other + 2) / 4)
                )
        result[c] = projector
    return result


def _unitary(x: float, y: float, z: float) -> Matrix:
    scale = 1 + x * x + y * y + z * z
    q0, q1, q2, q3 = (1 - x * x - y * y - z * z) / scale, 2*x/scale, 2*y/scale, 2*z/scale
    return np.array(
        [[q0 + 1j*q3, q2 + 1j*q1], [-q2 + 1j*q1, q0 - 1j*q3]],
        dtype=np.complex128,
    )


def _representation(n: int, unitary: Matrix) -> Matrix:
    # Direct symmetric powers of the fundamental polynomial representation.
    result = np.zeros((n + 1, n + 1), dtype=np.complex128)
    for k in range(n + 1):
        for ell in range(n + 1):
            for i in range(max(0, ell - k), min(n - k, ell) + 1):
                result[ell, k] += (
                    sqrt(comb(n, k) / comb(n, ell))
                    * comb(n - k, i) * comb(k, ell - i)
                    * unitary[0, 0]**(n - k - i) * unitary[1, 0]**i
                    * unitary[0, 1]**(k - ell + i) * unitary[1, 1]**(ell - i)
                )
    return result


def _chebyshev(n: int, trace: complex) -> complex:
    if n == 0:
        return 1
    previous, current = complex(1), trace
    for _ in range(1, n):
        previous, current = current, trace * current - previous
    return current


def test_nuclear_norm_from_independent_casimir_projectors() -> None:
    rng = Random(811)
    for a in range(4):
        for b in range(4):
            projectors = _projectors(a, b)
            for _ in range(3):
                coefficients = {
                    (a, b, c): Q(rng.randint(-9, 9), 13) for c in projectors
                }
                matrix = sum(
                    (float(coefficients[a, b, c]) * p / (c + 1) for c, p in projectors.items()),
                    np.zeros(((a + 1)*(b + 1), (a + 1)*(b + 1)), dtype=np.complex128),
                )
                nuclear = float(np.linalg.svd(matrix, compute_uv=False).sum())
                assert nuclear == pytest.approx(float(su2_theta_norm(coefficients, norm_order=0)))
                assert float(su2_theta_norm(coefficients)) == pytest.approx(
                    nuclear * (1 + a + b)**2
                )


def test_product_character_values_on_grid_and_random_group_elements() -> None:
    rng = Random(491)
    points: list[tuple[float, ...]] = [
        (x / 3, 0.0, 0.0, 0.0, y / 3, 0.0) for x in range(-3, 4) for y in range(-3, 4)
    ]
    points += [tuple(rng.uniform(-1, 1) for _ in range(6)) for _ in range(16)]
    for a, b in ((0, 0), (1, 1), (1, 2), (2, 2), (3, 2)):
        coefficients = su2_theta_product_characters({a: Q(2, 7)}, {b: Q(-3, 5)})
        projectors = _projectors(a, b)
        for point in points:
            u, v = _unitary(*point[:3]), _unitary(*point[3:])
            representation = np.kron(_representation(a, u), _representation(b, v))
            actual = sum(
                float(value) * np.trace(projectors[c] @ representation) / (c + 1)
                for (_, _, c), value in coefficients.items()
            )
            expected = -6 / 35 * _chebyshev(a, complex(np.trace(u))) * _chebyshev(
                b, complex(np.trace(v))
            )
            assert actual == pytest.approx(expected, abs=2e-12)


def test_product_and_physical_derivative_bounds() -> None:
    rng = Random(725)
    a, b = 2, 1
    projectors = _projectors(a, b)
    coefficients = {(a, b, c): Q(rng.randint(-7, 7), 11) for c in projectors}
    matrix = sum(
        (float(coefficients[a, b, c]) * p / (c + 1) for c, p in projectors.items()),
        np.zeros(((a + 1)*(b + 1), (a + 1)*(b + 1)), dtype=np.complex128),
    )
    n1 = float(su2_theta_norm(coefficients, norm_order=1))
    n2 = float(su2_theta_norm(coefficients, norm_order=2))
    for _ in range(32):
        u, v = (_unitary(*(rng.uniform(-1, 1) for _ in range(3))) for _ in range(2))
        du, dv = _representation(a, u), _representation(b, v)
        gu = np.array([
            np.trace(matrix @ np.kron(du @ (1j*x), dv)).real
            for x in _spin_generators(a)
        ])
        gv = np.array([
            np.trace(matrix @ np.kron(du, dv @ (1j*x))).real
            for x in _spin_generators(b)
        ])
        assert np.linalg.norm(np.concatenate((gu, gv))) <= n1 / 2 + 1e-12
        physical_gradient = sqrt(3*np.dot(gu, gu) + 3*np.dot(gv, gv) + np.dot(gu+gv, gu+gv))
        assert physical_gradient <= n1 + 1e-12
        product_value = np.trace(matrix @ np.kron(du, dv))
        assert abs((a*(a+2) + b*(b+2))/4 * product_value) <= n2 / 4 + 1e-12
        physical_laplacian = sum(
            float(value) * (3*a*(a+2) + 3*b*(b+2) + c*(c+2))/4
            * np.trace(projectors[c] @ np.kron(du, dv))/(c+1)
            for (_, _, c), value in coefficients.items()
        )
        assert abs(physical_laplacian) <= n2 + 1e-12


@pytest.mark.parametrize("n", range(17))
def test_exact_dimension_lift_polynomial(n: int) -> None:
    direct = sum((a + 1)**2 * (b + 1)**2 for a in range(n + 1) for b in range(n - a + 1))
    assert direct == Q((n+1)*(n+2)*(n+3)*(n+4)*(2*n*n+10*n+15), 360)
    assert comb(n+3, 3)**2 - direct == Q(n*(n+1)*(n+2)*(n+3)*(2*n+5)*(4*n+11), 360)


@pytest.mark.parametrize("order", (0, 1, 2))
@pytest.mark.parametrize("ratio", (Q(0), Q(1, 7), Q(3, 8)))
def test_rational_tail_against_independent_positive_sum(order: int, ratio: Q) -> None:
    first, end = 5, 80
    tail = su2_theta_vacuum_tail(ratio, first_omitted=first, circle_norm=3, norm_order=order)
    partial = 3*sum(
        (ratio**n * (n+1)**order * comb(n+3, 3) for n in range(first, end)), Q(0)
    )
    # Beyond end, the coefficient ratio decreases and is bounded by this
    # first ratio. This bounds the unsummed infinite positive series.
    term = 3*ratio**end * (end+1)**order * comb(end+3, 3)
    ratio_bound = ratio * Q(end+2, end+1)**order * Q(end+4, end+1)
    assert ratio_bound < 1
    assert partial <= tail <= partial + term / (1-ratio_bound)


def test_tail_known_initial_coefficients_and_monotonicity() -> None:
    t = Q(1, 6)
    for order in (0, 1, 2):
        previous = su2_theta_vacuum_tail(t, first_omitted=0, circle_norm=9, norm_order=order)
        for n in range(12):
            following = su2_theta_vacuum_tail(t, first_omitted=n+1, circle_norm=9, norm_order=order)
            assert previous - following == 9*t**n*(n+1)**order*comb(n+3, 3)
            assert 0 < following < previous
            previous = following
    assert su2_theta_vacuum_tail(0, first_omitted=0, circle_norm=9) == 9
    assert su2_theta_vacuum_tail(0, first_omitted=1, circle_norm=9) == 0


def _biproduct(left: BiPolynomial, right: BiPolynomial) -> BiPolynomial:
    result: BiPolynomial = {}
    for (a, b), x in left.items():
        for (c, d), y in right.items():
            for i in range(abs(a-c), a+c+1, 2):
                for j in range(abs(b-d), b+d+1, 2):
                    result[i, j] = result.get((i, j), Q(0)) + x*y
    return {key: value for key, value in result.items() if value}


def _binorm(values: BiPolynomial, order: int = 2) -> Q:
    return sum(
        ((1+a+b)**order*(a+1)*(b+1)*abs(v) for (a, b), v in values.items()), Q(0)
    )


def _difference(left: BiPolynomial, right: BiPolynomial) -> BiPolynomial:
    return {key: left.get(key, Q(0))-right.get(key, Q(0)) for key in left.keys() | right.keys()}


def _to_theta(values: BiPolynomial) -> dict[ThetaState, Q]:
    return {
        (a, b, c): (c+1)*value
        for (a, b), value in values.items()
        for c in range(abs(a-b), a+b+1, 2)
    }


@pytest.mark.parametrize("order", (0, 1, 2))
def test_density_bounds_against_exact_positive_nonproduct_vacua(order: int) -> None:
    rng = Random(670)
    for _ in range(24):
        p: BiPolynomial = {
            (0, 0): Q(1),
            (1, 0): Q(rng.randint(-4, 4), 160),
            (0, 1): Q(rng.randint(-4, 4), 160),
            (1, 1): Q(rng.randint(-4, 4), 400),
            (2, 0): Q(rng.randint(-2, 2), 500),
        }
        extra = Q(rng.randint(-3, 3), 700)
        actual = dict(p)
        actual[0, 2] = extra
        # These actual test wavefunctions are everywhere strictly positive.
        assert _binorm(_difference(actual, {(0, 0): Q(1)}), 0) < 1
        bound = su2_theta_product_reference_bound(
            _to_theta(p),
            vacuum_norm_error=3**(order+1)*abs(extra),
            vacuum_l2_error=abs(extra),
            norm_order=order,
        )
        square = _biproduct(actual, actual)
        density = {key: value/square[0, 0] for key, value in square.items()}
        mu = {key: value for key, value in density.items() if key[1] == 0}
        mv = {key: value for key, value in density.items() if key[0] == 0}
        own_product = _biproduct(mu, mv)
        connected = _binorm(_difference(density, own_product), order)
        pu = {key: value for key, value in p.items() if key[1] == 0}
        pv = {key: value for key, value in p.items() if key[0] == 0}
        reference_square = _biproduct(_biproduct(pu, pv), _biproduct(pu, pv))
        reference_density = {
            key: value/reference_square[0, 0] for key, value in reference_square.items()
        }
        assert _binorm(_difference(density, reference_density), order) <= bound[
            "actual_joint_to_product_reference_error"
        ]
        assert connected <= bound["actual_joint_to_own_marginals_error"]
        assert all(type(value) is Q for value in bound.values())


def test_exact_factorized_reference_has_zero_error() -> None:
    vacuum = su2_theta_product_characters({0: 1, 1: Q(1, 20)}, {0: 1, 2: Q(-1, 30)})
    result = su2_theta_product_reference_bound(vacuum, vacuum_norm_error=0, vacuum_l2_error=0)
    assert result["amplitude_difference_polynomial_norm"] == 0
    assert result["normalization_error_vs_reference"] == 0
    assert result["actual_joint_to_own_marginals_error"] == 0
    assert result["actual_joint_to_product_reference_error"] == 0


def test_known_fundamental_connected_example() -> None:
    eta = Q(1, 1000)
    result = su2_theta_product_reference_bound(
        {(0, 0, 0): 1, (1, 1, 0): eta, (1, 1, 2): 3*eta},
        vacuum_norm_error=0, vacuum_l2_error=0,
    )
    assert result["amplitude_difference_polynomial_norm"] == 36*eta
    assert result["vacuum_polynomial_l2_norm_squared"] == 1+eta*eta
    assert result["reference_amplitude_l2_norm_squared"] == 1
    assert result["actual_joint_to_product_reference_error"] == 72*eta+1297*eta*eta
    exact_connected_norm = 72*eta/(1+eta*eta)+225*eta*eta/(1+eta*eta)**2
    assert exact_connected_norm <= result["actual_joint_to_own_marginals_error"]
    assert result["actual_joint_to_own_marginals_error"] == result[
        "actual_joint_to_product_reference_error"
    ]


def test_sector_budget_endpoint_bound_independently() -> None:
    rng = Random(510)
    for _ in range(200):
        delta, bu, bv = (Q(rng.randrange(101), 20) for _ in range(3))
        x = delta*Q(rng.randrange(101), 100)
        y = (delta-x)*Q(rng.randrange(101), 100)
        z = delta-x-y
        bound = max(delta, max(bu, bv)*delta+delta*delta/4)
        assert z+bv*x+bu*y+x*y <= bound


@pytest.mark.parametrize(
    "state",
    [(1, 0, 0), (1, 1, 1), (-1, 0, 1), (True, 1, 0), (0, 0), ("0", 0, 0)],
)
def test_invalid_theta_labels(state: Any) -> None:
    with pytest.raises(ValueError):
        su2_theta_norm({state: Q(0)})


@pytest.mark.parametrize("value", [0.1, True, "1/2"])
def test_nonexact_coefficients_rejected(value: Any) -> None:
    with pytest.raises(TypeError):
        su2_theta_norm({(0, 0, 0): value})
    with pytest.raises(TypeError):
        su2_theta_product_characters({0: value}, {0: 1})


@pytest.mark.parametrize("order", [-1, True, 1.5])
def test_invalid_norm_orders(order: Any) -> None:
    with pytest.raises(ValueError):
        su2_theta_norm({}, norm_order=order)
    with pytest.raises(ValueError):
        su2_theta_product_reference_bound({(0, 0, 0): 1}, vacuum_norm_error=0,
                                          vacuum_l2_error=0, norm_order=order)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ratio": -1}, {"ratio": 1}, {"ratio": Q(3, 2)}, {"circle_norm": -1},
        {"first_omitted": -1}, {"first_omitted": True}, {"norm_order": 3},
    ],
)
def test_invalid_tail_arguments(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {"ratio": Q(1, 7), "first_omitted": 4, "circle_norm": 3}
    values.update(kwargs)
    with pytest.raises(ValueError):
        su2_theta_vacuum_tail(**values)


@pytest.mark.parametrize(
    "vacuum,error,l2",
    [({}, 0, 0), ({(0, 0, 0): Q(2)}, 0, 0), ({(0, 0, 0): Q(1)}, -1, 0),
     ({(0, 0, 0): Q(1)}, 0, -1)],
)
def test_invalid_reference_premise_inputs(
    vacuum: dict[ThetaState, Q], error: int, l2: int
) -> None:
    with pytest.raises(ValueError):
        su2_theta_product_reference_bound(vacuum, vacuum_norm_error=error, vacuum_l2_error=l2)
