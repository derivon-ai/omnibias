# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Distinct-root, center-fiber, uniform derivative, and coverage regressions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from random import Random

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.cyclicity import (
    ConfluentExponentialPolynomial,
    certify_exponential_cyclicity,
    certify_planar_return_cyclicity,
    certify_polynomial_cyclicity,
    identity_coefficients,
    polynomial_range,
    verify_cyclicity_certificate,
    verify_exponential_cyclicity,
    verify_return_cyclicity,
)
from omnibias.dynamics.hilbert16 import (
    CyclicityLeaf,
    CyclicitySplit,
    certify_polynomial_cover,
    verify_polynomial_cover,
)
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    certify_stopped_event,
)


def test_sturm_counts_distinct_including_multiple_endpoint_roots() -> None:
    h = P.variable(1, 0)
    p = (h + 2) ** 3 * h**2 * (h - 1) * (h - 3) ** 2
    cert = certify_polynomial_cyclicity(p, ((-2, 3),))
    assert cert.exact_count == cert.upper_bound == 4
    assert cert.method == "sturm"
    assert verify_cyclicity_certificate(p, cert)
    assert certify_polynomial_cyclicity(p, ((-1, Q(1, 2)),)).exact_count == 1


def test_identity_is_not_an_infinite_isolated_root_count() -> None:
    h, a = (P.variable(2, i) for i in range(2))
    p = a * (h**3 - h)
    center = certify_polynomial_cyclicity(p, ((-2, 2), (0, 0)))
    assert center.identity and center.exact_count == 0 and center.method == "identity"
    equations = identity_coefficients(p)
    assert all(q.evaluate((0,)) == 0 for q in equations)
    assert any(q.evaluate((1,)) != 0 for q in equations)
    family = certify_polynomial_cyclicity(p, ((-2, 2), (-1, 1)))
    assert family.upper_bound == 3 and family.identity is None
    assert family.method == "polynomial_degree"
    assert certify_polynomial_cyclicity(P(2), ((0, 1), (-1, 1))).exact_count == 0


def test_uniform_rolle_over_parameter_continuum() -> None:
    h, a = (P.variable(2, i) for i in range(2))
    # Has either zero, one double, or two real roots on the height interval.
    p = h**2 + a
    cert = certify_polynomial_cyclicity(p, ((-2, 2), (-1, 1)))
    assert cert.method == "rolle" and cert.upper_bound == 2
    assert cert.derivative_order == 2 and cert.derivative_range == (2, 2)
    assert verify_cyclicity_certificate(p, cert)
    empty = certify_polynomial_cyclicity(h**2 + a + 3, ((-2, 2), (-1, 1)))
    assert empty.upper_bound == 0 and empty.derivative_order == 0


def test_exact_range_dense_and_random_rational_points() -> None:
    x, y = (P.variable(2, i) for i in range(2))
    p = x**4 - 3 * x * y + y**3 + Q(1, 7)
    box = ((Q(-3, 2), Q(1, 2)), (Q(-2, 3), Q(4, 3)))
    lo, hi = polynomial_range(p, box)
    rng = Random(1626)
    grid = [Q(i, 20) for i in range(21)]
    probes = [(a, b) for a in grid for b in grid]
    probes += [(Q(rng.randrange(1001), 1000), Q(rng.randrange(1001), 1000)) for _ in range(100)]
    for s, t in probes:
        point = tuple(a + u * (b - a) for (a, b), u in zip(box, (s, t), strict=True))
        assert lo <= p.evaluate(point) <= hi


def test_constructed_random_factored_polynomials_independent_count() -> None:
    rng, h = Random(16), P.variable(1, 0)
    for _ in range(30):
        roots = [rng.randrange(-5, 6) for _ in range(8)]
        p = P.constant(1, 1)
        for root in roots:
            p = p * (h - root)
        # Nonreal factors must not change the real count.
        p = p * (h**2 + 1)
        cert = certify_polynomial_cyclicity(p, ((-3, 3),))
        assert cert.exact_count == len({r for r in roots if -3 <= r <= 3})


def test_tamper_and_source_substitution_rejected_even_after_reseal() -> None:
    h = P.variable(1, 0)
    p = h**2 - 1
    cert = certify_polynomial_cyclicity(p, ((-2, 2),))
    assert not verify_cyclicity_certificate(h**2 + 1, cert)
    assert not verify_cyclicity_certificate(p, replace(cert, upper_bound=0))
    body = deepcopy(cert.seal)
    body["payload"]["upper_bound"] = 0
    sealed = seal_certificate(body)
    assert not verify_cyclicity_certificate(p, replace(cert, upper_bound=0, seal=sealed))
    assert not verify_cyclicity_certificate(p, cert, expected_domain=((-1, 1),))


def test_parameter_cover_handles_identity_face_and_rejects_gap() -> None:
    h, a = (P.variable(2, i) for i in range(2))
    p = a * (h**2 - 1)
    left = CyclicityLeaf(certify_polynomial_cyclicity(p, ((-2, 2), (-1, 0))))
    right = CyclicityLeaf(certify_polynomial_cyclicity(p, ((-2, 2), (0, 1))))
    tree = CyclicitySplit(1, 0, left, right)
    cert = certify_polynomial_cover(p, ((-2, 2), (-1, 1)), tree)
    assert cert.upper_bound == 2 and cert.leaves == 2
    assert verify_polynomial_cover(p, cert)
    gap_right = CyclicityLeaf(certify_polynomial_cyclicity(p, ((-2, 2), (Q(1, 100), 1))))
    with pytest.raises(ValueError, match="domain"):
        certify_polynomial_cover(p, ((-2, 2), (-1, 1)), replace(tree, right=gap_right))
    with pytest.raises(ValueError, match="inside"):
        certify_polynomial_cover(p, ((-2, 2), (-1, 1)), replace(tree, cut=1))


def test_height_cover_does_not_lose_split_boundary_root() -> None:
    h = P.variable(1, 0)
    p = h * (h - 1)
    tree = CyclicitySplit(
        0, 0,
        CyclicityLeaf(certify_polynomial_cyclicity(p, ((-2, 0),))),
        CyclicityLeaf(certify_polynomial_cyclicity(p, ((0, 2),))),
    )
    cert = certify_polynomial_cover(p, ((-2, 2),), tree)
    assert cert.upper_bound == 2  # Sum=3 is sharpened by root's independent count.
    assert verify_polynomial_cover(p, cert)
    assert not verify_polynomial_cover(p + 1, cert)
    assert not verify_polynomial_cover(p, replace(cert, upper_bound=0))


@pytest.mark.parametrize("box", [((1, 1),), ((1, 0),), ((0.0, 1.0),), ()])
def test_invalid_or_inexact_domains_rejected(box: tuple[tuple[float, float], ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify_polynomial_cyclicity(P.variable(1, 0), box)  # type: ignore[arg-type]


def test_exponential_division_handles_confluence_and_cancellation() -> None:
    p = ConfluentExponentialPolynomial([(0, (1, 0, 3)), (-1, (2,)), (1, (4, 5))])
    cert = certify_exponential_cyclicity(p)
    assert [f.dimension for f in cert.chain] == [6, 5, 4, 3, 2, 1]
    assert cert.upper_bound == 5 and verify_exponential_cyclicity(p, cert)
    zero = ConfluentExponentialPolynomial([(1, (1, 2)), (1, (-1, -2))])
    assert certify_exponential_cyclicity(zero).identity
    assert certify_exponential_cyclicity(zero).upper_bound == 0
    tiny = ConfluentExponentialPolynomial([(0, (Q(1, 10**100),)), (1, (1,))])
    assert certify_exponential_cyclicity(tiny).upper_bound == 1
    assert not verify_exponential_cyclicity(p, replace(cert, upper_bound=0))


def test_exponential_known_double_root_and_independent_derivative() -> None:
    # (exp(kappa)-1)^2 has a double zero; no positivity of coefficients assumed.
    p = ConfluentExponentialPolynomial([(0, (1,)), (1, (-2,)), (2, (1,))])
    cert = certify_exponential_cyclicity(p)
    assert cert.upper_bound == 2
    assert cert.chain[1] == ConfluentExponentialPolynomial([(1, (-2,)), (2, (2,))])
    # At kappa=0, all exponential factors are exactly one. Compare the exact
    # derivative value by the product rule independently of the recurrence.
    q = ConfluentExponentialPolynomial([(Q(-3, 2), (2, 3, 4)), (2, (-1, 5))])
    d = q.weighted_derivative(Q(1, 3))
    direct = sum((a - Q(1, 3)) * p[0] + (p[1] if len(p) > 1 else 0) for a, p in q.terms)
    assert sum(p[0] for _, p in d.terms) == direct
    with pytest.raises(ValueError, match="budget"):
        certify_exponential_cyclicity(q, max_dimension=2)
    with pytest.raises(TypeError, match="exact"):
        ConfluentExponentialPolynomial([(1.0, (1,))])  # type: ignore[list-item]


@pytest.fixture(scope="module")
def damped_return():
    x, y, h = P.variable(4, 0), P.variable(4, 1), P.variable(1, 0)
    request = StoppedEventRequest(
        flow=PolynomialFlow((y - Q(1, 10) * x, -x - Q(1, 10) * y), 1),
        initial=(P.constant(1, 0), h), parameters=(Interval(1, 1.0001),),
        target=PolynomialEvent(x, direction=1, guard=y),
        step=0.0625, max_steps=110, order=8,
    )
    result = certify_stopped_event(request)
    assert result.certified, result.reason
    return result


def test_actual_return_excludes_damped_spiral_cycles_and_replays(damped_return) -> None:
    count = certify_planar_return_cyclicity(damped_return)
    assert count.upper_bound == 0 and count.method == "value_exclusion"
    assert count.displacement_range.hi < 0
    assert verify_return_cyclicity(count, expected_source_fingerprint=damped_return.request.fingerprint)
    assert not verify_return_cyclicity(count, expected_source_fingerprint="other field")
    assert not verify_return_cyclicity(replace(count, upper_bound=1),
                                       expected_source_fingerprint=damped_return.request.fingerprint)


def test_actual_return_identity_center_stays_unresolved_by_interval_rolle(damped_return) -> None:
    x, y = P.variable(4, 0), P.variable(4, 1)
    event = certify_stopped_event(replace(damped_return.request, flow=PolynomialFlow((y, -x), 1)))
    assert event.certified, event.reason
    count = certify_planar_return_cyclicity(event)
    assert count.upper_bound is None and count.method == "unresolved_zero_count"
    assert count.displacement_range.contains_zero()


def test_actual_return_rejects_wrong_parameter_time_source_or_embedding(damped_return) -> None:
    p, t = P.variable(4, 2), P.variable(4, 3)
    request = damped_return.request
    for correction in (p, t):
        altered = replace(request, flow=PolynomialFlow((request.flow.components[0] + correction,
                                                       request.flow.components[1]), 1))
        with pytest.raises(ValueError, match="autonomous"):
            certify_planar_return_cyclicity(replace(damped_return, request=altered))
    bad = replace(request, initial=(P.constant(1, 1), request.initial[1]))
    with pytest.raises(ValueError, match="embedding"):
        certify_planar_return_cyclicity(replace(damped_return, request=bad))
    with pytest.raises(ValueError, match="replayed"):
        certify_planar_return_cyclicity(replace(damped_return, source_fingerprint="altered"))
    with pytest.raises(ValueError, match="designated"):
        certify_planar_return_cyclicity(damped_return, height_parameter=1)


def test_actual_hopf_return_has_at_most_one_cycle_near_the_unit_circle(damped_return) -> None:
    x, y = P.variable(4, 0), P.variable(4, 1)
    radial = Q(1, 10) * (P.constant(4, 1) - x*x - y*y)
    field = PolynomialFlow((y + radial*x, -x + radial*y), 1)
    request = replace(damped_return.request, flow=field, parameters=(Interval(0.99999, 1.00001),),
                      step=0.03125, max_steps=210)
    event = certify_stopped_event(request)
    assert event.certified, event.reason
    cert = certify_planar_return_cyclicity(event)
    assert cert.upper_bound == 1 and cert.method == "actual_return_rolle"
    assert cert.displacement_range.contains_zero()
    assert cert.first_derivative is not None and cert.first_derivative.hi < 0
    # The field's unit-circle restriction is exactly (y,-x), establishing a
    # known orbit separately from the upper bound; no numerical fit is used.
    assert not (x*field.components[0] + y*field.components[1] - radial*(x*x + y*y)).terms
