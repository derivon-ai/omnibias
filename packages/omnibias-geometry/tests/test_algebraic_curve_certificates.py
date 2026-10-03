# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Actual coefficient, complex singularity, and complete-edge regressions."""

from dataclasses import replace
from fractions import Fraction as Q
from math import comb
from random import Random

import pytest
from omnibias.core.proof.lean_check import generate_obligation, lean_check_available
from omnibias.core.realization.polynomial import AlgebraBudgetExceeded, SparsePolynomial
from omnibias.geometry.algebraic import (
    ChartBezoutIdentity,
    HomogeneousPlaneCurve,
    PolygonalAnnulus,
    ProjectiveSmoothnessWitness,
    RationalPolygon,
    RationalRectangle,
    RectangularAnnulus,
    affine_chart,
    certify_curve,
    find_smoothness_witness,
    replay_curve_certificate,
    segment_bernstein_coefficients,
    verify_curve_certificate_formally,
)


def _variables():
    return tuple(SparsePolynomial.variable(3, i) for i in range(3))


def _square(cx, cy, radius):
    return RationalRectangle(cx - radius, cx + radius, cy - radius, cy + radius)


def _square_polygon(cx, cy, radius):
    return RationalPolygon((
        (cx - radius, cy - radius),
        (cx + radius, cy - radius),
        (cx + radius, cy + radius),
        (cx - radius, cy + radius),
    ))


def test_conic_actual_polynomial_and_source_bound_replay():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve(x**2 + y**2 - z**2)
    witness = find_smoothness_witness(curve, max_multiplier_degree=0)
    assert witness is not None and witness.verifies(curve)
    annulus = RectangularAnnulus(_square(0, 0, Q(1, 2)), _square(0, 0, 2))
    cert = certify_curve(curve, witness, [annulus])
    assert cert.component_lower_bound == cert.harnack_upper_bound == 1
    assert cert.complete_real_scheme and cert.barrier_parents == (-1,)
    assert cert.boundary_signs == ((-1, 1),)
    assert replay_curve_certificate(curve, cert)
    for forged in (
        replace(cert, component_lower_bound=2),
        replace(cert, complete_real_scheme=False),
        replace(cert, barrier_parents=(0,)),
        replace(cert, boundary_margins=((Q(100), Q(100)),)),
        replace(cert, fixed_axis=0),
    ):
        assert not replay_curve_certificate(curve, forged)
    altered = HomogeneousPlaneCurve(x**2 + y**2 - 2 * z**2)
    assert not replay_curve_certificate(altered, cert)
    assert not witness.verifies(altered)


def test_four_separated_quartic_ovals_attain_harnack():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve((x**2 - z**2)**2 + (y**2 - z**2)**2 - Q(1, 16) * z**4)
    witness = find_smoothness_witness(curve)
    assert witness is not None
    annuli = [RectangularAnnulus(_square(a, b, Q(1, 32)), _square(a, b, Q(1, 4)))
              for a in (-1, 1) for b in (-1, 1)]
    cert = certify_curve(curve, witness, annuli)
    assert cert.complete_real_scheme
    assert cert.component_lower_bound == cert.harnack_upper_bound == 4
    assert cert.barrier_parents == (-1, -1, -1, -1)
    assert replay_curve_certificate(curve, cert)


def test_nested_quartic_annuli_do_not_claim_maximal_classification():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve((x**2 + y**2 - z**2) * (x**2 + y**2 - 4 * z**2)
                                 + Q(1, 100) * x**4)
    witness = find_smoothness_witness(curve)
    assert witness is not None
    inner = RectangularAnnulus(_square(0, 0, Q(1, 2)), _square(0, 0, Q(5, 4)))
    outer = RectangularAnnulus(_square(0, 0, Q(4, 3)), _square(0, 0, Q(5, 2)))
    cert = certify_curve(curve, witness, [inner, outer])
    assert cert.barrier_parents == (1, -1)
    assert cert.component_lower_bound == 2 and cert.harnack_upper_bound == 4
    assert not cert.complete_real_scheme
    assert replay_curve_certificate(curve, cert)


def test_octic_baseline_is_not_the_open_maximal_scheme():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve(x**8 + y**8 - z**8)
    witness = find_smoothness_witness(curve, max_multiplier_degree=0)
    assert witness is not None
    annulus = RectangularAnnulus(_square(0, 0, Q(1, 2)), _square(0, 0, 2))
    cert = certify_curve(curve, witness, [annulus])
    assert (cert.component_lower_bound, cert.harnack_upper_bound) == (1, 22)
    assert not cert.complete_real_scheme
    assert replay_curve_certificate(curve, cert)


def test_sixteen_oval_octic_with_exact_complex_smoothness_identities():
    x, y, z = _variables()
    px = (x**2 - z**2) * (x**2 - 9 * z**2)
    py = (y**2 - z**2) * (y**2 - 9 * z**2)
    curve = HomogeneousPlaneCurve(px**2 + py**2 - Q(1, 16) * z**8)
    witness = find_smoothness_witness(curve, max_multiplier_degree=12)
    assert witness is not None and witness.verifies(curve)
    annuli = [RectangularAnnulus(_square(a, b, Q(1, 1024)), _square(a, b, Q(1, 32)))
              for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)]
    cert = certify_curve(curve, witness, annuli)
    assert (cert.component_lower_bound, cert.harnack_upper_bound) == (16, 22)
    assert cert.barrier_parents == (-1,) * 16
    assert not cert.complete_real_scheme
    assert replay_curve_certificate(curve, cert)
    obligation = generate_obligation(cert.formal_seal)
    assert obligation is not None and "polynomialIdentity" in obligation
    formal = verify_curve_certificate_formally(cert)
    if lean_check_available():  # pragma: no cover - Lean-equipped environment only
        assert formal.theorem_prover_verified
    else:
        assert not formal.finite_obligation.available


def test_sixteen_oval_octic_recertifies_through_polygonal_annuli():
    x, y, z = _variables()
    px = (x**2 - z**2) * (x**2 - 9 * z**2)
    py = (y**2 - z**2) * (y**2 - 9 * z**2)
    curve = HomogeneousPlaneCurve(px**2 + py**2 - Q(1, 16) * z**8)
    witness = find_smoothness_witness(curve, max_multiplier_degree=12)
    assert witness is not None
    annuli = [
        PolygonalAnnulus(
            _square_polygon(a, b, Q(1, 1024)),
            _square_polygon(a, b, Q(1, 32)),
        )
        for a in (-3, -1, 1, 3)
        for b in (-3, -1, 1, 3)
    ]
    cert = certify_curve(curve, witness, annuli)
    assert (cert.component_lower_bound, cert.harnack_upper_bound) == (16, 22)
    assert cert.barrier_parents == (-1,) * 16
    assert cert.boundary_signs == ((-1, 1),) * 16
    assert not cert.complete_real_scheme
    assert replay_curve_certificate(curve, cert)


def test_nonrectangular_polygonal_annulus_certifies_whole_edges():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve(x**2 + y**2 - z**2)
    witness = find_smoothness_witness(curve, max_multiplier_degree=0)
    assert witness is not None
    def diamond(radius):
        return RationalPolygon((
            (radius, 0),
            (0, radius),
            (-radius, 0),
            (0, -radius),
        ))

    annulus = PolygonalAnnulus(diamond(Q(1, 2)), diamond(2))
    cert = certify_curve(curve, witness, [annulus])
    assert cert.complete_real_scheme
    assert cert.boundary_signs == ((-1, 1),)
    assert replay_curve_certificate(curve, cert)


def test_polygon_geometry_is_checked_exactly():
    with pytest.raises(ValueError, match="simple"):
        RationalPolygon(((0, 0), (3, 2), (0, 3), (2, 0)))
    with pytest.raises(TypeError, match="not floats"):
        RationalPolygon(((0.0, 0), (1, 0), (0, 1)))
    inner = _square_polygon(0, 0, 1)
    shifted = _square_polygon(2, 0, 2)
    with pytest.raises(ValueError, match="strictly contain"):
        PolygonalAnnulus(inner, shifted)


def test_annuli_in_a_declared_nondefault_projective_chart():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve(-x**2 + y**2 + z**2)
    witness = find_smoothness_witness(curve, max_multiplier_degree=0)
    assert witness is not None
    annulus = RectangularAnnulus(_square(0, 0, Q(1, 2)), _square(0, 0, 2))
    cert = certify_curve(curve, witness, [annulus], fixed_axis=0)
    assert cert.complete_real_scheme and replay_curve_certificate(curve, cert)


def test_complex_only_singularities_are_not_mistaken_for_smoothness():
    x, y, z = _variables()
    # Real locus is a smooth circle, but [1:i:0] and [1:-i:0] are singular.
    curve = HomogeneousPlaneCurve((x**2 + y**2)**2 - z**4)
    assert find_smoothness_witness(curve) is None
    one, zero = SparsePolynomial.constant(2, 1), SparsePolynomial.constant(2, 0)
    affine_witness = ChartBezoutIdentity(2, (zero, zero, -Q(1, 4) * one))
    assert affine_witness.verifies(curve)
    fake = ProjectiveSmoothnessWitness(curve.digest, (affine_witness,) * 3)
    with pytest.raises(ValueError, match="complex projective"):
        certify_curve(curve, fake, [])


def test_affine_smooth_curve_can_be_singular_at_real_infinity():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve(x * z**2 + y**3 + z**3)
    assert affine_chart(curve.polynomial.derivative(0), 2).terms == (((0, 0), Q(1)),)
    assert all(curve.polynomial.derivative(axis).evaluate((1, 0, 0)) == 0 for axis in range(3))
    assert find_smoothness_witness(curve) is None


def test_identity_tampering_is_rejected_after_rebinding_source():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve(x**2 + y**2 - z**2)
    witness = find_smoothness_witness(curve, max_multiplier_degree=0)
    assert witness is not None
    chart = witness.charts[0]
    forged = replace(chart, multipliers=(chart.multipliers[0] + 1, *chart.multipliers[1:]))
    assert not replace(witness, charts=(forged, *witness.charts[1:])).verifies(curve)


def test_whole_edges_reject_misleading_corner_samples():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve(x**2 + y**2 - z**2)
    witness = find_smoothness_witness(curve, max_multiplier_degree=0)
    assert witness is not None
    # Every inner vertex is positive, but the horizontal edge midpoint is negative.
    inner = RationalRectangle(-2, 2, -Q(1, 10), Q(1, 10))
    assert all(affine_chart(curve.polynomial, 2).evaluate(v) > 0 for v in inner.vertices)
    with pytest.raises(ValueError, match="whole-edge"):
        certify_curve(curve, witness, [RectangularAnnulus(inner, _square(0, 0, 3))],
                      subdivision_depth=2)


def test_bernstein_exact_identity_and_enclosure_on_grid_and_random_points():
    p = SparsePolynomial(2, {(4, 0): Q(2, 3), (1, 3): -2, (0, 1): 5, (0, 0): Q(1, 7)})
    start, end = (Q(-2), Q(1, 3)), (Q(3, 2), Q(-1))
    coefficients = segment_bernstein_coefficients(p, start, end)
    rng = Random(1616)
    points = [Q(i, 100) for i in range(101)] + [Q(rng.randrange(10001), 10000) for _ in range(100)]
    degree = len(coefficients) - 1
    for t in points:
        exact = p.evaluate(tuple(a + (b - a) * t for a, b in zip(start, end, strict=True)))
        reconstructed = sum(coefficients[k] * comb(degree, k) * t**k * (1 - t)**(degree - k)
                            for k in range(degree + 1))
        assert reconstructed == exact
        assert min(coefficients) <= exact <= max(coefficients)


def test_invalid_geometry_and_unknown_budget():
    x, y, z = _variables()
    curve = HomogeneousPlaneCurve(x**2 + y**2 - z**2)
    with pytest.raises(TypeError, match="not floats"):
        RationalRectangle(-1.0, 1, -1, 1)
    with pytest.raises(ValueError, match="homogeneous"):
        HomogeneousPlaneCurve(x**2 + y)
    with pytest.raises(AlgebraBudgetExceeded):
        find_smoothness_witness(curve, max_multiplier_degree=10, max_unknowns=10)
    witness = find_smoothness_witness(curve, max_multiplier_degree=0)
    assert witness is not None
    annulus = RectangularAnnulus(_square(0, 0, Q(1, 2)), _square(0, 0, 2))
    with pytest.raises(ValueError, match="separated or strictly nested"):
        certify_curve(curve, witness, [annulus, annulus])
    with pytest.raises(ValueError, match="opposite"):
        certify_curve(curve, witness,
                      [RectangularAnnulus(_square(0, 0, 2), _square(0, 0, 3))])


def test_odd_degree_pseudoline_is_derived_from_degree():
    x, _, _ = _variables()
    curve = HomogeneousPlaneCurve(x)
    witness = find_smoothness_witness(curve, max_multiplier_degree=0)
    assert witness is not None
    cert = certify_curve(curve, witness, [])
    assert cert.pseudoline_count == cert.component_lower_bound == 1
    assert cert.complete_real_scheme
