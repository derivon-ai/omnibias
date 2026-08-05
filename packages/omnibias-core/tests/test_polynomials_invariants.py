# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Analytic-invariant property tests for the polynomial coefficient
generators in :mod:`omnibias.core.polynomials`.

These tests are deliberately backend-free: they only use :mod:`math`
and :mod:`fractions` so they run on the bare ``omnibias-core`` install
and serve as the mathematical contract that the torch / jax fast paths
must satisfy.

For every recurrence we lock down:

1. **Closed-form value at z = 0** matches the analytic n-th derivative
   of the base activation, computed from a peer-reviewable identity:

   * sigmoid: ``sigma^(n)(0)`` reduces to ``P_n(1/2)`` and is given by
     a closed-form series in the Genocchi / Eulerian numbers; we
     hand-pin the first eight values.
   * tanh: ``tanh^(n)(0)`` is zero for even ``n`` and is the ``n``-th
     coefficient of the Taylor series of ``tanh`` (related to Bernoulli
     numbers) for odd ``n``; we hand-pin the first eight values.
   * gaussian: ``g^(n)(0) = (-1)^n He_n(0)`` where ``He_{2k}(0) =
     (-1)^k (2k)! / (2^k k!)`` and odd-index values vanish.

2. **Asymptotic invariants**:

   * sigmoid: ``sigma(z) -> 1`` as ``z -> +inf`` and all derivatives go
     to zero, so ``P_n(1) = 0`` for ``n >= 1``.
   * tanh: ``tanh(z) -> 1`` as ``z -> +inf`` and derivatives go to
     zero, so ``T_n(1) = 0`` for ``n >= 1`` and ``T_n(-1) = 0`` for
     ``n >= 1``.

3. **Integer-coefficient invariant**: every coefficient in
   ``sigmoid_polynomial_coeffs`` and ``tanh_polynomial_coeffs`` is an
   integer (stored as a float with no fractional part).

4. **Caching invariant**: repeated calls return the same tuple object
   (``functools.cache`` semantics), so consumers may use ``is``-identity
   for memoised dispatch.

5. **Riccati identity** at order 1: ``sigmoid'(z) = s (1 - s)`` is
   exactly ``P_1(s) = s - s^2``; ``tanh'(z) = 1 - t^2`` is exactly
   ``T_1(t) = 1 - t^2``.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.polynomials import (
    hermite_coeffs,
    sech_polynomial_coeffs,
    sigmoid_polynomial_coeffs,
    tanh_polynomial_coeffs,
)


def _eval_poly(coeffs: tuple[float, ...], x: float) -> float:
    """Horner evaluation of ``sum_k c_k x^k``."""
    out = 0.0
    for c in reversed(coeffs):
        out = out * x + c
    return out


# ---------------------------------------------------------------------------
# Closed-form n-th derivative values at z = 0.
# Hand-derived from the Taylor series; cross-checked symbolically.
# ---------------------------------------------------------------------------

# sigmoid^(n)(0) for n = 0..7. Computed from the Taylor expansion
#   sigma(z) = 1/2 + z/4 - z^3/48 + z^5/480 - 17 z^7 / 80640 + ...
# so even-order derivatives at zero vanish from order 2 onward and the
# odd-order values come from sigma^(n)(0) = n! * a_n.
_SIGMA_AT_ZERO = (
    0.5,                   # n=0   sigma(0)
    0.25,                  # n=1   1!  *  1/4   =  1/4
    0.0,                   # n=2
    -1.0 / 8.0,            # n=3   3!  * -1/48  = -1/8
    0.0,                   # n=4
    1.0 / 4.0,             # n=5   5!  *  1/480 =  1/4
    0.0,                   # n=6
    -17.0 / 16.0,          # n=7   7! * -17/80640 = -17/16
)


# tanh^(n)(0) for n = 0..7. Even-order derivatives vanish; odd-order
# derivatives at zero are given by the Bernoulli/tangent-number series.
_TANH_AT_ZERO = (
    0.0,                   # n=0
    1.0,                   # n=1   tanh'(0) = 1
    0.0,                   # n=2
    -2.0,                  # n=3
    0.0,                   # n=4
    16.0,                  # n=5
    0.0,                   # n=6
    -272.0,                # n=7
)


# sech^(n)(0) = Q_n(0) for n = 0..7. These constant terms are the Euler
# (secant) numbers E_n: odd indices vanish, even indices are 1, -1, 5, -61.
_SECH_AT_ZERO = (
    1.0,                   # n=0   E_0
    0.0,                   # n=1
    -1.0,                  # n=2   E_2
    0.0,                   # n=3
    5.0,                   # n=4   E_4
    0.0,                   # n=5
    -61.0,                 # n=6   E_6
    0.0,                   # n=7
)


# He_n(0) for n = 0..8. Odd indices vanish; even indices are
# (-1)^k (2k)! / (2^k k!).
_HE_AT_ZERO = (
    1.0,                   # n=0
    0.0,                   # n=1
    -1.0,                  # n=2
    0.0,                   # n=3
    3.0,                   # n=4
    0.0,                   # n=5
    -15.0,                 # n=6
    0.0,                   # n=7
    105.0,                 # n=8
)


# ---------------------------------------------------------------------------
# 1. Closed-form value at z = 0 matches analytic constant.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,expected", list(enumerate(_SIGMA_AT_ZERO)))
def test_sigmoid_polynomial_at_z_zero(n: int, expected: float) -> None:
    """``P_n(sigmoid(0)) == sigma^(n)(0)`` for the first eight orders."""
    coeffs = sigmoid_polynomial_coeffs(n)
    val = _eval_poly(coeffs, 0.5)
    assert math.isclose(val, expected, rel_tol=0.0, abs_tol=1e-12), (
        f"P_{n}(0.5)={val!r} expected {expected!r}"
    )


@pytest.mark.parametrize("n,expected", list(enumerate(_TANH_AT_ZERO)))
def test_tanh_polynomial_at_z_zero(n: int, expected: float) -> None:
    """``T_n(tanh(0)) == tanh^(n)(0)`` for the first eight orders."""
    coeffs = tanh_polynomial_coeffs(n)
    val = _eval_poly(coeffs, 0.0)
    assert math.isclose(val, expected, rel_tol=0.0, abs_tol=1e-12), (
        f"T_{n}(0)={val!r} expected {expected!r}"
    )


@pytest.mark.parametrize("n,expected", list(enumerate(_HE_AT_ZERO)))
def test_hermite_at_z_zero(n: int, expected: float) -> None:
    """``He_n(0)`` matches the closed-form for the first nine orders."""
    coeffs = hermite_coeffs(n)
    val = _eval_poly(coeffs, 0.0)
    assert math.isclose(val, expected, rel_tol=0.0, abs_tol=1e-12), (
        f"He_{n}(0)={val!r} expected {expected!r}"
    )


@pytest.mark.parametrize("n,expected", list(enumerate(_SECH_AT_ZERO)))
def test_sech_polynomial_at_z_zero(n: int, expected: float) -> None:
    """``Q_n(tanh(0)) == sech^(n)(0)`` is the Euler number ``E_n`` (first eight orders)."""
    coeffs = sech_polynomial_coeffs(n)
    val = _eval_poly(coeffs, 0.0)
    assert math.isclose(val, expected, rel_tol=0.0, abs_tol=1e-12), (
        f"Q_{n}(0)={val!r} expected E_{n}={expected!r}"
    )


# ---------------------------------------------------------------------------
# 2. Asymptotic invariants.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 8))
def test_sigmoid_polynomial_vanishes_at_one(n: int) -> None:
    """``P_n(1) = 0`` for all ``n >= 1`` (sigma -> 1, derivatives -> 0)."""
    coeffs = sigmoid_polynomial_coeffs(n)
    val = _eval_poly(coeffs, 1.0)
    assert math.isclose(val, 0.0, rel_tol=0.0, abs_tol=1e-12), (
        f"P_{n}(1)={val!r} should be 0"
    )


@pytest.mark.parametrize("n", range(1, 8))
def test_sigmoid_polynomial_vanishes_at_zero(n: int) -> None:
    """``P_n(0) = 0`` for all ``n >= 1`` (sigma -> 0, derivatives -> 0)."""
    coeffs = sigmoid_polynomial_coeffs(n)
    assert math.isclose(coeffs[0], 0.0, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize("n", range(1, 8))
def test_tanh_polynomial_vanishes_at_pm_one(n: int) -> None:
    """``T_n(+/-1) = 0`` for all ``n >= 1``."""
    coeffs = tanh_polynomial_coeffs(n)
    for x in (-1.0, 1.0):
        val = _eval_poly(coeffs, x)
        assert math.isclose(val, 0.0, rel_tol=0.0, abs_tol=1e-12), (
            f"T_{n}({x})={val!r} should be 0"
        )


# ---------------------------------------------------------------------------
# 3. Integer-coefficient invariant.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(8))
def test_sigmoid_coeffs_are_integers(n: int) -> None:
    coeffs = sigmoid_polynomial_coeffs(n)
    for c in coeffs:
        assert c == int(c), f"Non-integer sigmoid coeff at n={n}: {c!r}"


@pytest.mark.parametrize("n", range(8))
def test_tanh_coeffs_are_integers(n: int) -> None:
    coeffs = tanh_polynomial_coeffs(n)
    for c in coeffs:
        assert c == int(c), f"Non-integer tanh coeff at n={n}: {c!r}"


@pytest.mark.parametrize("n", range(8))
def test_hermite_coeffs_are_integers(n: int) -> None:
    coeffs = hermite_coeffs(n)
    for c in coeffs:
        assert c == int(c), f"Non-integer hermite coeff at n={n}: {c!r}"


@pytest.mark.parametrize("n", range(8))
def test_sech_coeffs_are_integers(n: int) -> None:
    coeffs = sech_polynomial_coeffs(n)
    for c in coeffs:
        assert c == int(c), f"Non-integer sech coeff at n={n}: {c!r}"


# ---------------------------------------------------------------------------
# 4. Caching invariant.
# ---------------------------------------------------------------------------


def test_sigmoid_polynomial_is_cached() -> None:
    a = sigmoid_polynomial_coeffs(5)
    b = sigmoid_polynomial_coeffs(5)
    assert a is b, "functools.cache must return the same tuple object"


def test_tanh_polynomial_is_cached() -> None:
    a = tanh_polynomial_coeffs(5)
    b = tanh_polynomial_coeffs(5)
    assert a is b


def test_hermite_coeffs_is_cached() -> None:
    a = hermite_coeffs(5)
    b = hermite_coeffs(5)
    assert a is b


def test_sech_polynomial_is_cached() -> None:
    a = sech_polynomial_coeffs(5)
    b = sech_polynomial_coeffs(5)
    assert a is b


# ---------------------------------------------------------------------------
# 5. Length and degree contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(8))
def test_sigmoid_polynomial_length(n: int) -> None:
    """``P_n`` has degree ``n + 1`` so the tuple has length ``n + 2``."""
    assert len(sigmoid_polynomial_coeffs(n)) == n + 2


@pytest.mark.parametrize("n", range(8))
def test_tanh_polynomial_length(n: int) -> None:
    """``T_n`` has degree ``n + 1`` so the tuple has length ``n + 2``."""
    assert len(tanh_polynomial_coeffs(n)) == n + 2


@pytest.mark.parametrize("n", range(8))
def test_hermite_coeffs_length(n: int) -> None:
    """``He_n`` has degree ``n`` so the tuple has length ``n + 1``."""
    assert len(hermite_coeffs(n)) == n + 1


@pytest.mark.parametrize("n", range(8))
def test_sech_polynomial_length(n: int) -> None:
    """``Q_n`` has degree ``n`` so the tuple has length ``n + 1``."""
    assert len(sech_polynomial_coeffs(n)) == n + 1


# ---------------------------------------------------------------------------
# 6. Numerical consistency: closed-form polynomial agrees with a tight
#    finite-difference estimate of the next-lower-order derivative.
# ---------------------------------------------------------------------------


def _sigma(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _tanh(z: float) -> float:
    return math.tanh(z)


def _gauss(z: float) -> float:
    return math.exp(-0.5 * z * z)


def _sech(z: float) -> float:
    return 1.0 / math.cosh(z)


def _nth_derivative_numeric(f, z: float, n: int, h: float = 5e-3) -> float:
    """Centered ``n``-th finite difference (loose tolerance, used only for
    smoke checks)."""
    if n == 0:
        return f(z)
    # Use the Romberg-extended central-difference table for first
    # n derivatives by repeated symmetric differencing; cheap for n <= 3.
    if n == 1:
        return (f(z + h) - f(z - h)) / (2.0 * h)
    if n == 2:
        return (f(z + h) - 2.0 * f(z) + f(z - h)) / (h * h)
    if n == 3:
        return (f(z + 2 * h) - 2.0 * f(z + h) + 2.0 * f(z - h) - f(z - 2 * h)) / (
            2.0 * h ** 3
        )
    raise NotImplementedError


@pytest.mark.parametrize("z", [-1.5, -0.3, 0.0, 0.3, 1.5])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_sigmoid_closed_form_matches_finite_difference(z: float, n: int) -> None:
    s = _sigma(z)
    closed = _eval_poly(sigmoid_polynomial_coeffs(n), s)
    fd = _nth_derivative_numeric(_sigma, z, n)
    # Loose tolerance because central differences lose ~h^2 precision,
    # but this still pins the sign and magnitude of every closed-form
    # value against an independent oracle.
    assert math.isclose(closed, fd, rel_tol=2e-3, abs_tol=2e-4), (
        f"closed sigma^({n})({z})={closed!r}, FD={fd!r}"
    )


@pytest.mark.parametrize("z", [-1.5, -0.3, 0.0, 0.3, 1.5])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_tanh_closed_form_matches_finite_difference(z: float, n: int) -> None:
    t = _tanh(z)
    closed = _eval_poly(tanh_polynomial_coeffs(n), t)
    fd = _nth_derivative_numeric(_tanh, z, n)
    assert math.isclose(closed, fd, rel_tol=5e-3, abs_tol=5e-4), (
        f"closed tanh^({n})({z})={closed!r}, FD={fd!r}"
    )


@pytest.mark.parametrize("z", [-1.5, -0.3, 0.0, 0.3, 1.5])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_gaussian_closed_form_matches_finite_difference(z: float, n: int) -> None:
    """``g^(n)(z) = (-1)^n He_n(z) g(z)`` matches a centered FD."""
    he = _eval_poly(hermite_coeffs(n), z)
    closed = ((-1) ** n) * he * _gauss(z)
    fd = _nth_derivative_numeric(_gauss, z, n)
    assert math.isclose(closed, fd, rel_tol=5e-3, abs_tol=5e-4), (
        f"closed g^({n})({z})={closed!r}, FD={fd!r}"
    )


@pytest.mark.parametrize("z", [-1.5, -0.3, 0.0, 0.3, 1.5])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_sech_closed_form_matches_finite_difference(z: float, n: int) -> None:
    """``sech^(n)(z) = Q_n(tanh z) sech(z)`` matches a centered FD."""
    t = _tanh(z)
    closed = _eval_poly(sech_polynomial_coeffs(n), t) * _sech(z)
    fd = _nth_derivative_numeric(_sech, z, n)
    assert math.isclose(closed, fd, rel_tol=5e-3, abs_tol=5e-4), (
        f"closed sech^({n})({z})={closed!r}, FD={fd!r}"
    )


# --------------------------------------------------------------------------- #
# Exactness and resource bounds (audit items: float accumulation, unbounded
# caches, unbounded recursion).
# --------------------------------------------------------------------------- #
def test_float_coefficients_are_correctly_rounded_at_every_representable_order() -> None:
    """The float tower must equal the exact integer tower, rounded once.

    Accumulating the recurrence in `float` compounds rounding at every step and
    starts disagreeing with the correctly-rounded coefficient from order 19
    (sigmoid), 20 (sech), 22 (tanh) and 30 (Hermite) -- the orders where the
    coefficients outgrow 2**53. Computing in `int` and narrowing once fixes it.
    """
    from omnibias.core.polynomials import (
        hermite_coeffs,
        sech_polynomial_coeffs,
        sigmoid_polynomial_coeffs,
        tanh_polynomial_coeffs,
    )
    from omnibias.core.verified.coeffs import (
        hermite_coeffs_exact,
        sech_poly_coeffs_exact,
        sigmoid_poly_coeffs_exact,
        tanh_poly_coeffs_exact,
    )

    # `top` is the last order whose coefficients still fit in a float64.
    families = [
        ("sigmoid", sigmoid_polynomial_coeffs, sigmoid_poly_coeffs_exact, 159),
        ("tanh", tanh_polynomial_coeffs, tanh_poly_coeffs_exact, 163),
        ("sech", sech_polynomial_coeffs, sech_poly_coeffs_exact, 163),
        ("hermite", hermite_coeffs, hermite_coeffs_exact, 296),
    ]
    for name, approx, exact, top in families:
        for n in (0, 1, 2, 3, 17, 19, 22, 30, 64, top):
            want = tuple(float(c) for c in exact(n))
            assert approx(n) == want, f"{name} order {n} is not correctly rounded"


def test_representability_ceiling_is_reported_clearly() -> None:
    """Past float64 range the answer cannot be a float; say so, don't overflow."""
    import pytest
    from omnibias.core.polynomials import (
        hermite_coeffs,
        sech_polynomial_coeffs,
        sigmoid_polynomial_coeffs,
        tanh_polynomial_coeffs,
    )

    for fn, first_unrepresentable in (
        (sigmoid_polynomial_coeffs, 160),
        (tanh_polynomial_coeffs, 164),
        (sech_polynomial_coeffs, 164),
        (hermite_coeffs, 297),
    ):
        assert fn(first_unrepresentable - 1)  # the last representable order works
        with pytest.raises(ValueError, match="exceed the float64 range"):
            fn(first_unrepresentable)


def test_order_is_bounded_in_both_directions() -> None:
    import pytest
    from omnibias.core.polynomials import (
        MAX_ORDER,
        hermite_coeffs,
        mish_inner_coeffs,
        sech_polynomial_coeffs,
        sigmoid_polynomial_coeffs,
        tanh_polynomial_coeffs,
    )

    generators = (
        sigmoid_polynomial_coeffs,
        tanh_polynomial_coeffs,
        sech_polynomial_coeffs,
        hermite_coeffs,
        mish_inner_coeffs,
    )
    for fn in generators:
        with pytest.raises(ValueError, match="must be >= 0"):
            fn(-1)
        with pytest.raises(ValueError, match="MAX_ORDER"):
            fn(MAX_ORDER + 1)


def test_coefficient_caches_are_bounded() -> None:
    """An unbounded memo keyed on a caller-supplied order is a DoS vector."""
    from omnibias.core import bell, multi_index, polynomials
    from omnibias.core.verified import coeffs

    cached = [
        polynomials.sigmoid_polynomial_coeffs,
        polynomials.tanh_polynomial_coeffs,
        polynomials.sech_polynomial_coeffs,
        polynomials.hermite_coeffs,
        polynomials.mish_inner_coeffs,
        bell.bell_number,
        multi_index._multi_indices,
        multi_index._multiply_table,
        coeffs.sigmoid_poly_coeffs_exact,
        coeffs.tanh_poly_coeffs_exact,
        coeffs.hermite_coeffs_exact,
        coeffs.sech_poly_coeffs_exact,
    ]
    for fn in cached:
        info = fn.cache_info()
        assert info.maxsize is not None, f"{fn.__name__} has an unbounded cache"
        assert info.maxsize > 0


def test_tall_towers_do_not_exhaust_the_interpreter_stack() -> None:
    """These recurrences used to be self-recursive and died around order 1000."""
    from omnibias.core.verified.coeffs import (
        hermite_coeffs_exact,
        sech_poly_coeffs_exact,
        sigmoid_poly_coeffs_exact,
        tanh_poly_coeffs_exact,
    )

    for fn in (
        sigmoid_poly_coeffs_exact,
        tanh_poly_coeffs_exact,
        sech_poly_coeffs_exact,
        hermite_coeffs_exact,
    ):
        fn.cache_clear()  # a cold call is the recursive worst case
        assert len(fn(1200)) >= 1200


def test_bell_and_multi_index_orders_are_bounded() -> None:
    import pytest
    from omnibias.core.bell import (
        MAX_BELL_NUMBER_ORDER,
        MAX_BELL_ORDER,
        bell_complete,
        bell_number,
        bell_partial,
    )
    from omnibias.core.multi_index import MAX_MULTI_INDICES, multi_indices

    # The partition enumerations are exponential in n, so they cap much lower
    # than the quadratic Bell triangle.
    with pytest.raises(ValueError, match="MAX_BELL_ORDER"):
        bell_complete(MAX_BELL_ORDER + 1)
    with pytest.raises(ValueError, match="MAX_BELL_ORDER"):
        bell_partial(MAX_BELL_ORDER + 1, 2)
    assert bell_number(500) > 0  # the documented in-tree usage still works
    with pytest.raises(ValueError, match="<="):
        bell_number(MAX_BELL_NUMBER_ORDER + 1)

    # One joint bound on the result size, since the count grows in both args.
    with pytest.raises(ValueError, match="MAX_MULTI_INDICES"):
        multi_indices(6, 60)
    assert len(multi_indices(3, 4)) == 35
    assert MAX_MULTI_INDICES > 0


def test_wide_low_order_multi_index_does_not_recurse() -> None:
    """`_generate` used to recurse once per dimension."""
    from omnibias.core.multi_index import multi_indices

    wide = multi_indices(2000, 1)
    assert len(wide) == 2001
