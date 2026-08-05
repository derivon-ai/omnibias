# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Randomized soundness sweep for the SOS / Positivstellensatz certificate.

:func:`~omnibias.sos.certify_sos` makes one claim when it returns ``"proved"``:
the polynomial is a sum of squares, hence **nonnegative everywhere**. That is a
statement about uncountably many points, so it cannot be confirmed by sampling
-- but it can be *refuted* by a single point, which is what these sweeps hunt
for. A ``"proved"`` verdict on a polynomial that dips below zero anywhere is the
failure mode that matters, because downstream packages (``qubo``, ``submodular``,
``nphard``) turn it into a certified optimality gap.

``"inconclusive"`` is not a failure. The interval ``LDL^T`` check demands a
*strictly* positive-definite Gram matrix, so a polynomial that is PSD-but-singular
(``(x - y)^2``, say) is honestly refused rather than approximated. Incompleteness
is the correct direction for a rigorous checker to err in; the tests below pin
down that it errs only in that direction.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.sos import Polynomial, certify_sos, is_sos

_PROVED = "proved"


def _random_sos_polynomial(rng: np.random.Generator, n_vars: int) -> Polynomial:
    r"""``p(x) = sum_k (a_k . x + b_k)^2 + c``, nonnegative by construction."""
    n_squares = int(rng.integers(2, 4))
    coeffs: dict[tuple[int, ...], float] = {}

    def add(exponent: tuple[int, ...], value: float) -> None:
        coeffs[exponent] = coeffs.get(exponent, 0.0) + value

    for _ in range(n_squares):
        a = rng.standard_normal(n_vars)
        b = float(rng.standard_normal())
        for i in range(n_vars):
            for j in range(n_vars):
                exponent = [0] * n_vars
                exponent[i] += 1
                exponent[j] += 1
                add(tuple(exponent), float(a[i] * a[j]))
            add(tuple(1 if k == i else 0 for k in range(n_vars)), 2.0 * b * float(a[i]))
        add((0,) * n_vars, b * b)
    add((0,) * n_vars, float(rng.uniform(0.25, 1.5)))  # keep the Gram strictly PD
    return Polynomial(n_vars, coeffs)


def _random_indefinite_polynomial(rng: np.random.Generator, n_vars: int) -> Polynomial:
    """A quadratic form with a deliberately negative direction."""
    coeffs: dict[tuple[int, ...], float] = {(0,) * n_vars: float(rng.uniform(-1.0, 1.0))}
    for i in range(n_vars):
        exponent = [0] * n_vars
        exponent[i] = 2
        sign = -1.0 if i % 2 else 1.0
        coeffs[tuple(exponent)] = sign * float(rng.uniform(0.5, 2.0))
    return Polynomial(n_vars, coeffs)


def _min_over_samples(poly: Polynomial, rng: np.random.Generator, n_vars: int) -> float:
    points = rng.uniform(-6.0, 6.0, size=(400, n_vars))
    return min(float(poly.evaluate(list(point))) for point in points)


def test_a_proved_certificate_is_never_contradicted_by_a_sample() -> None:
    """The load-bearing sweep: ``proved`` must imply nonnegative at every probe."""
    for seed in range(150):
        rng = np.random.default_rng(seed)
        n_vars = int(rng.integers(2, 4))
        poly = _random_sos_polynomial(rng, n_vars)
        cert = certify_sos(poly)
        if cert.status != _PROVED:
            continue  # honest incompleteness, not a soundness failure
        worst = _min_over_samples(poly, rng, n_vars)
        assert worst >= 0.0, (
            f"seed={seed}: certify_sos returned {_PROVED!r} (pd_margin="
            f"{cert.pd_margin!r}) for a polynomial that evaluates to {worst!r} < 0 -- "
            f"the sum-of-squares claim is false"
        )


def test_indefinite_polynomials_are_never_proved() -> None:
    """A polynomial with a negative direction must never earn a certificate."""
    for seed in range(150):
        rng = np.random.default_rng(seed + 10_000)
        n_vars = int(rng.integers(2, 4))
        poly = _random_indefinite_polynomial(rng, n_vars)
        worst = _min_over_samples(poly, rng, n_vars)
        if worst >= 0.0:
            continue  # the sample failed to expose the negative direction
        assert certify_sos(poly).status != _PROVED, (
            f"seed={seed}: certify_sos proved a polynomial that reaches {worst!r} < 0"
        )
        assert not is_sos(poly)


def test_the_checker_is_conservative_not_optimistic() -> None:
    """A PSD-but-singular Gram is refused, which is the safe direction to err in."""
    psd_singular = Polynomial(2, {(2, 0): 1.0, (1, 1): -2.0, (0, 2): 1.0})  # (x - y)^2
    assert certify_sos(psd_singular).status != _PROVED
    strictly_positive = Polynomial(2, {(2, 0): 1.0, (0, 2): 1.0, (0, 0): 0.5})
    assert certify_sos(strictly_positive).status == _PROVED


def test_a_proved_certificate_reports_a_positive_margin() -> None:
    """``pd_margin`` is what the interval ``LDL^T`` actually established."""
    for seed in range(40):
        rng = np.random.default_rng(seed + 20_000)
        poly = _random_sos_polynomial(rng, 2)
        cert = certify_sos(poly)
        if cert.status == _PROVED:
            assert cert.pd_margin > 0.0, f"seed={seed}: proved with no margin"


@pytest.mark.slow
def test_proved_certificates_survive_a_wider_probe() -> None:
    """The same claim over a much larger sample and a wider box."""
    for seed in range(60):
        rng = np.random.default_rng(seed + 30_000)
        n_vars = int(rng.integers(2, 5))
        poly = _random_sos_polynomial(rng, n_vars)
        if certify_sos(poly).status != _PROVED:
            continue
        points = rng.uniform(-50.0, 50.0, size=(4000, n_vars))
        worst = min(float(poly.evaluate(list(point))) for point in points)
        assert worst >= 0.0, f"seed={seed}: proved polynomial reaches {worst!r} < 0"
