# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pure-Python regression tests for the Bell / Faà di Bruno combinatorics.

These run with no torch / jax / numpy dependency; they validate only the
mathematical contracts of the recurrences (exact integer arithmetic). The
backend jet kernels that *consume* these coefficients are tested in
``omnibias-jax`` / ``omnibias-torch``.
"""

from __future__ import annotations

import pytest
from omnibias.core.bell import (
    bell_complete,
    bell_number,
    bell_partial,
    faa_di_bruno_terms,
)

# Known Bell numbers Bell(0..8).
_BELL = [1, 1, 2, 5, 15, 52, 203, 877, 4140]


# ----- edge cases -----


def test_bell_partial_0_0() -> None:
    assert bell_partial(0, 0) == {(): 1}


def test_bell_partial_zero_when_k_gt_n() -> None:
    assert bell_partial(3, 4) == {}


def test_bell_partial_zero_when_only_one_is_zero() -> None:
    assert bell_partial(3, 0) == {}
    assert bell_partial(0, 2) == {}


def test_bell_partial_negative_raises() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        bell_partial(-1, 0)
    with pytest.raises(ValueError, match="must be >= 0"):
        bell_partial(2, -1)


# ----- B_{n,1} = x_n and B_{n,n} = x_1^n -----


@pytest.mark.parametrize("n", range(1, 7))
def test_bell_partial_k1_is_xn(n: int) -> None:
    expected_key = tuple(1 if i == n - 1 else 0 for i in range(n))
    assert bell_partial(n, 1) == {expected_key: 1}


@pytest.mark.parametrize("n", range(1, 7))
def test_bell_partial_kn_is_x1_pow_n(n: int) -> None:
    expected_key = tuple(n if i == 0 else 0 for i in range(n))
    assert bell_partial(n, n) == {expected_key: 1}


# ----- small hand-checked partials -----


def test_bell_partial_3_2() -> None:
    # B_{3,2} = 3 x_1 x_2.
    assert bell_partial(3, 2) == {(1, 1, 0): 3}


def test_bell_partial_4_2() -> None:
    # B_{4,2} = 4 x_1 x_3 + 3 x_2^2.
    assert bell_partial(4, 2) == {(1, 0, 1, 0): 4, (0, 2, 0, 0): 3}


def test_bell_partial_4_3() -> None:
    # B_{4,3} = 6 x_1^2 x_2.
    assert bell_partial(4, 3) == {(2, 1, 0, 0): 6}


# ----- complete Bell polynomial / Bell numbers -----


@pytest.mark.parametrize("n", range(len(_BELL)))
def test_bell_number_matches_known(n: int) -> None:
    assert bell_number(n) == _BELL[n]


@pytest.mark.parametrize("n", range(len(_BELL)))
def test_complete_at_ones_is_bell_number(n: int) -> None:
    total = sum(bell_complete(n).values())
    assert total == _BELL[n]


def test_bell_number_matches_partition_sum() -> None:
    # The fast Bell-triangle path must return the identical integer as summing
    # the complete Bell polynomial's p(n) partition terms, for every n it is
    # feasible to enumerate (regression for the O(n^2) rewrite of bell_number).
    for n in range(31):
        assert bell_number(n) == sum(bell_complete(n).values())


def test_bell_number_large_n_oeis() -> None:
    # OEIS A000110 -- values well past the partition-enumeration horizon; the
    # old implementation could not reach n=50 in 30 s.
    assert bell_number(20) == 51724158235372
    assert bell_number(25) == 4638590332229999353
    assert bell_number(30) == 846749014511809332450147
    assert len(str(bell_number(500))) == 844


def test_bell_number_negative_raises() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        bell_number(-1)


def test_complete_is_sum_of_partials() -> None:
    for n in range(7):
        merged: dict[tuple[int, ...], int] = {}
        for k in range(n + 1):
            for key, coeff in bell_partial(n, k).items():
                merged[key] = merged.get(key, 0) + coeff
        assert merged == bell_complete(n)


# ----- Faà di Bruno against the textbook closed forms -----


def _faa_eval(sigma_d: list[float], u_d: list[float], n: int) -> float:
    """Evaluate (sigma o u)^(n) from the term list.

    ``sigma_d[k]`` is sigma^(k)(u0); ``u_d[i]`` is u^(i) (1-indexed via u_d[0]
    unused). Used as an independent re-summation oracle.
    """
    total = 0.0
    for k, exps, coeff in faa_di_bruno_terms(n):
        prod = 1.0
        for i, e in enumerate(exps, start=1):
            if e:
                prod *= u_d[i] ** e
        total += coeff * sigma_d[k] * prod
    return total


def test_faa_di_bruno_textbook_n2() -> None:
    s = [0.0, 2.0, 3.0]          # sigma, sigma', sigma''
    u = [0.0, 5.0, 7.0]          # u(0) unused, u', u''
    # sigma'' u'^2 + sigma' u''.
    expected = 3.0 * 5.0**2 + 2.0 * 7.0
    assert _faa_eval(s, u, 2) == expected


def test_faa_di_bruno_textbook_n3() -> None:
    s = [0.0, 2.0, 3.0, 4.0]
    u = [0.0, 5.0, 7.0, 11.0]
    # sigma''' u'^3 + 3 sigma'' u' u'' + sigma' u'''.
    expected = 4.0 * 5.0**3 + 3.0 * 3.0 * 5.0 * 7.0 + 2.0 * 11.0
    assert _faa_eval(s, u, 3) == expected


def test_faa_di_bruno_textbook_n4() -> None:
    s = [0.0, 2.0, 3.0, 4.0, 6.0]
    u = [0.0, 5.0, 7.0, 11.0, 13.0]
    # sigma'''' u'^4 + 6 sigma''' u'^2 u'' + 3 sigma'' u''^2
    #   + 4 sigma'' u' u''' + sigma' u''''.
    expected = (
        6.0 * 5.0**4
        + 6.0 * 4.0 * 5.0**2 * 7.0
        + 3.0 * 3.0 * 7.0**2
        + 4.0 * 3.0 * 5.0 * 11.0
        + 2.0 * 13.0
    )
    assert _faa_eval(s, u, 4) == expected


def test_faa_di_bruno_reduces_to_complete_bell_for_exp() -> None:
    # For sigma = exp, all sigma^(k) = exp(u0); factoring it out, the sum over
    # Faà di Bruno terms equals the complete Bell polynomial of u's derivatives.
    u = [0.0, 1.5, -2.0, 0.7, 3.1, -1.1]
    for n in range(1, 6):
        sigma_d = [1.0] * (n + 1)
        faa = _faa_eval(sigma_d, u, n)
        bell = 0.0
        for key, coeff in bell_complete(n).items():
            prod = 1.0
            for i, e in enumerate(key, start=1):
                if e:
                    prod *= u[i] ** e
            bell += coeff * prod
        assert faa == pytest.approx(bell, rel=1e-12, abs=1e-12)
