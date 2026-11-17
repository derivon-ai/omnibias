# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Critical-response laws when only projected Gaussian group means agree."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_grouped_directional import (
    certify_gaussian_grouped_directional,
    compile_gaussian_grouped_critical_face,
)
from omnibias.core.verified.gaussian_grouped_equilibrium import (
    certify_gaussian_grouped_equilibrium,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_projected_equilibrium import (
    certify_gaussian_projected_equilibrium,
)
from omnibias.core.verified.interval import Interval

ZERO = Interval.point(0)


def _equilibrium(*, normal_offset=4, cross=False, gauge=0, repeats=1):
    centers = ((-1, 0, -int(cross)), (1, 0, int(cross))) + (
        (0, -1, normal_offset-int(cross)), (0, 1, normal_offset+int(cross)),
    )*repeats
    groups = (0, 0)+(1,)*(2*repeats)
    biases = tuple(gauge+group+sum(F(value)**2 for value in row)/4
                   for group, row in zip(groups, centers, strict=True))
    source = GaussianHeatMap(centers, biases)
    base = restrict_gaussian_heat(source, ((0, 0, 1),), (0,))
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, ((1, 2, 0), (0, 1, 0)), (0, 0), (0,))
    assert face is not None
    equilibrium = certify_gaussian_projected_equilibrium(face, (1, 2, 0), (0, 0, 0))
    assert equilibrium is not None
    return equilibrium


def _compile(equilibrium, active=()):
    face = compile_gaussian_grouped_critical_face(equilibrium, active)
    assert face is not None
    return face


def _mp(mp, value):
    exact = F(value)
    return mp.mpf(exact.numerator)/exact.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def test_different_normal_centroids_bind_original_kkt_and_leave_shared_centroid_api_unchanged():
    equilibrium = _equilibrium()
    face, source = equilibrium.face, equilibrium.face.base_restriction.source_model
    assert source.contraction_bound == F(9, 4)
    assert face.base_restriction.model.contraction_bound == F(1, 2)
    assert equilibrium.strong_convexity_bound == F(1, 2)
    assert certify_gaussian_grouped_equilibrium(face, equilibrium.target, equilibrium.point) is None
    assert equilibrium.groups == ((0, 1), (2, 3))
    assert equilibrium.group_logits == (0, 1)
    assert equilibrium.group_centroids == ((0, 0, 0), (0, 0, 4))
    assert equilibrium.group_centroid_offsets == ((0, 0, 0), (0, 0, 4))
    assert equilibrium.centered_centers == ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0))
    assert equilibrium.projected_centroid == (0, 0, 0)
    assert equilibrium.group_multipliers == ((0, 1), (4, 1))
    assert equilibrium.active_indices == (0, 1)
    assert equilibrium.strong_indices == (0,) and equilibrium.weak_indices == (1,)
    assert equilibrium.slacks == (0, 0)
    assert equilibrium.inequality_multipliers == (Interval.point(1), ZERO)


@pytest.mark.parametrize("cross", [False, True])
def test_exact_group_stationarity_and_compiled_projector_identities(cross):
    equilibrium = _equilibrium(cross=cross)
    face = _compile(equilibrium)
    assert face.constraints == ((0, 0, 1), (1, 2, 0))
    assert face.projector == ((F(4, 5), F(-2, 5), 0), (F(-2, 5), F(1, 5), 0), (0, 0, 0))
    assert face.left_inverse == ((0, 0, 1), (F(1, 5), F(2, 5), 0))
    for centroid, multipliers in zip(equilibrium.group_centroids, equilibrium.group_multipliers, strict=True):
        force = tuple(v+c for v, c in zip(equilibrium.target, centroid, strict=True))
        replayed = tuple(sum(row[i]*multiplier for row, multiplier in zip(face.constraints, multipliers, strict=True))
                         for i in range(3))
        assert replayed == force
        assert tuple(sum(a*b for a, b in zip(row, centroid, strict=True)) for row in face.projector) == (0, 0, 0)
    assert face.posterior == equilibrium.posterior
    assert face.group_probabilities == equilibrium.group_probabilities
    assert face.projected_hessian[2] == (ZERO,)*3
    assert all(row[2] == ZERO for row in face.projected_hessian+face.multiplier_hessian)


@pytest.mark.parametrize("cross", [False, True])
def test_explicit_full_and_projected_hessians_include_the_correct_distinct_covariances(cross):
    mp = pytest.importorskip("mpmath")
    face = _compile(_equilibrium(cross=cross))
    with mp.workdps(110):
        e = mp.exp(1)
        p, q = 1/(1+e), e/(1+e)
        hx, hy = 1-p/2, 1-q/2
        hz = 1-int(cross)/2-8*p*q
        xz, yz = (-p/2, -q/2) if cross else (mp.mpf(0), mp.mpf(0))
        hessian = ((hx, mp.mpf(0), xz), (mp.mpf(0), hy, yz), (xz, yz, hz))
        curvature = 4*hx+hy
        php = ((4*curvature/25, -2*curvature/25, mp.mpf(0)),
               (-2*curvature/25, curvature/25, mp.mpf(0)), (mp.mpf(0),)*3)
        lhp = ((2*(2*xz-yz)/5, -(2*xz-yz)/5, mp.mpf(0)),
               (4*(hx-hy)/25, -2*(hx-hy)/25, mp.mpf(0)))
        for observed, expected in ((face.hessian, hessian), (face.projected_hessian, php), (face.multiplier_hessian, lhp)):
            for row, wanted in zip(observed, expected, strict=True):
                for interval, value in zip(row, wanted, strict=True):
                    _contains(mp, interval, value)
        assert face.hessian[2][2].hi < 0
        _contains(mp, face.equilibrium.equality_multipliers[0], 4*q)
        if cross:
            assert face.multiplier_hessian[0][0].lo > 0
            assert face.multiplier_hessian[0][1].hi < 0
        else:
            assert face.multiplier_hessian[0] == (ZERO,)*3


@pytest.mark.parametrize("cross", [False, True])
def test_normal_group_displacement_changes_full_curvature_but_not_tangent_blocks(cross):
    common = _compile(_equilibrium(normal_offset=0, cross=cross))
    displaced = _compile(_equilibrium(normal_offset=4, cross=cross))
    assert common.hessian[2][2] != displaced.hessian[2][2]
    for field in ("posterior", "group_probabilities", "projector", "left_inverse", "projected_hessian", "multiplier_hessian"):
        assert getattr(common, field) == getattr(displaced, field)
    a = certify_gaussian_grouped_directional(common, (1, 0, 0), (1.0, -0.5, 0.0))
    b = certify_gaussian_grouped_directional(displaced, (1, 0, 0), (1.0, -0.5, 0.0))
    assert a is not None and b is not None
    for field in ("value", "enclosure", "error_bound", "residual_enclosure", "equality_multipliers", "weak_slacks"):
        assert getattr(a, field) == getattr(b, field)


@pytest.mark.parametrize("active", [(), (1,)])
def test_zero_forcing_has_exact_zero_response_even_with_bad_proposal(active):
    face = _compile(_equilibrium(cross=True), active)
    result = certify_gaussian_grouped_directional(face, (0, 0, 0), (99.0, -11.0, 17.0))
    assert result is not None and result.value == (0, 0, 0)
    assert result.error_bound == result.residual_norm_bound == 0
    assert result.enclosure == result.residual_enclosure == (ZERO,)*3
    assert result.coordinate_error_bounds == (0, 0, 0)
    assert result.equality_multipliers == (ZERO,)*len(face.constraints)
    assert result.weak_slacks == result.weak_multipliers == (ZERO, ZERO)


def test_normal_directions_have_exact_zero_response_and_unrestricted_strong_multipliers():
    face = _compile(_equilibrium(cross=True))
    result = certify_gaussian_grouped_directional(face, (-3, -6, 7), (5.0, -8.0, 20.0))
    assert result is not None and result.value == (0, 0, 0)
    assert result.projected_proposal != result.value
    assert result.error_bound == 0
    assert result.equality_multipliers == (Interval.point(7), Interval.point(-3))
    assert result.weak_multipliers == result.weak_slacks == (ZERO, ZERO)


def test_full_rank_critical_face_keeps_weak_multiplier_sign_gate():
    equilibrium = _equilibrium()
    face = _compile(equilibrium, (1,))
    result = certify_gaussian_grouped_directional(face, (-1, 0, 0), (37.0, -8.0, 99.0))
    assert result is not None and result.value == (0, 0, 0)
    assert result.error_bound == 0
    assert result.equality_multipliers == (ZERO, Interval.point(-1), Interval.point(2))
    assert result.weak_multipliers == (ZERO, Interval.point(2))
    assert certify_gaussian_grouped_directional(face, (1, 0, 0), (0.0, 0.0, 0.0)) is None
    assert certify_gaussian_grouped_directional(_compile(equilibrium), (-1, 0, 0), (-1.0, 0.5, 0.0)) is None


def test_exact_projection_discards_both_original_equality_and_strong_normal_proposal_offsets():
    face = _compile(_equilibrium(cross=True))
    baseline = certify_gaussian_grouped_directional(face, (1, 0, 0), (1.0, -0.5, 0.0))
    shifted = certify_gaussian_grouped_directional(face, (1, 0, 0), (17.0, 31.5, 2.0**200))
    assert baseline is not None and shifted is not None
    assert baseline.value == shifted.value == (1, F(-1, 2), 0)
    for field in ("enclosure", "error_bound", "coordinate_error_bounds", "residual_enclosure", "residual_norm_bound",
                  "equality_multipliers", "weak_multipliers", "weak_slacks"):
        assert getattr(baseline, field) == getattr(shifted, field)
    assert baseline.error_bound > 0 and baseline.coordinate_error_bounds[2] == 0


def test_group_cardinality_affects_probabilities_original_normal_multiplier_and_tangent_response():
    mp = pytest.importorskip("mpmath")
    equilibrium = _equilibrium(repeats=2)
    assert equilibrium.groups == ((0, 1), (2, 3, 4, 5))
    face = _compile(equilibrium)
    with mp.workdps(110):
        e = mp.exp(1)
        p, q = 1/(1+2*e), 2*e/(1+2*e)
        _contains(mp, equilibrium.equality_multipliers[0], 4*q)
        _contains(mp, equilibrium.group_probabilities[0], p)
        _contains(mp, equilibrium.group_probabilities[1], q)
        wanted = (4/(4*(1-p/2)+(1-q/2)), -2/(4*(1-p/2)+(1-q/2)), mp.mpf(0))
        result = certify_gaussian_grouped_directional(face, (1, 0, 0), tuple(float(value) for value in wanted))
        assert result is not None
        for interval, value in zip(result.enclosure, wanted, strict=True):
            _contains(mp, interval, value)


def test_huge_bias_gauge_preserves_original_equilibrium_and_all_compiled_query_bounds():
    normal, shifted = _compile(_equilibrium(cross=True)), _compile(_equilibrium(cross=True, gauge=10**400))
    for field in ("group_logits", "group_centroids", "group_multipliers", "posterior", "group_probabilities",
                  "equality_multipliers", "inequality_multipliers"):
        assert getattr(normal.equilibrium, field) == getattr(shifted.equilibrium, field)
    for field in ("projector", "left_inverse", "hessian", "projected_hessian", "multiplier_hessian"):
        assert getattr(normal, field) == getattr(shifted, field)
    a = certify_gaussian_grouped_directional(normal, (1, 0, 0), (1.0, -0.5, 0.0))
    b = certify_gaussian_grouped_directional(shifted, (1, 0, 0), (1.0, -0.5, 0.0))
    assert a is not None and b is not None
    for field in ("value", "enclosure", "error_bound", "residual_enclosure", "residual_norm_bound",
                  "coordinate_error_bounds", "equality_multipliers", "weak_multipliers", "weak_slacks"):
        assert getattr(a, field) == getattr(b, field)


@pytest.mark.parametrize("epsilon", [F(1, 8), F(1, 10**1000)], ids=("finite_gap", "gap_below_binary64_range"))
def test_bias_collapse_changes_the_exact_critical_cone_and_negative_target_derivative(epsilon):
    mp = pytest.importorskip("mpmath")

    def bind(gap):
        source = GaussianHeatMap(((-1,), (1,)), (F(1, 4), F(1, 4)+gap))
        base = restrict_gaussian_heat(source, (), ())
        assert base is not None and base.model.contraction_bound == F(1, 2)
        face = compile_gaussian_polyhedral_face(base, ((1,),), (0,), (0,))
        assert face is not None
        equilibrium = certify_gaussian_projected_equilibrium(face, (0,), (0,))
        assert equilibrium is not None
        return equilibrium

    separated, collapsed = bind(epsilon), bind(F(0))
    assert separated.groups == ((0,), (1,)) and separated.group_logits == (0, epsilon)
    assert separated.group_multipliers == ((-1,), (1,))
    assert separated.strong_indices == (0,) and separated.weak_indices == ()
    assert collapsed.groups == ((0, 1),) and collapsed.group_logits == (0,)
    assert collapsed.group_multipliers == ((0,),)
    assert collapsed.strong_indices == () and collapsed.weak_indices == (0,)
    assert collapsed.inequality_multipliers == (ZERO,)
    with mp.workdps(100):
        # The original posterior mean at x=0 is tanh(epsilon/2),
        # strictly positive even when binary64 cannot represent epsilon.
        multiplier = mp.tanh(_mp(mp, epsilon)/2)
        assert multiplier > 0
        _contains(mp, separated.inequality_multipliers[0], multiplier)
    if float(epsilon) == 0:
        assert separated.inequality_multipliers[0].lo == 0

    strong_face, weak_face = _compile(separated), _compile(collapsed)
    assert strong_face.constraints == ((1,),) and strong_face.projector == ((0,),)
    assert weak_face.constraints == () and weak_face.projector == ((1,),)
    assert weak_face.hessian == ((Interval.point(0.5),),)
    before = certify_gaussian_grouped_directional(strong_face, (-1,), (-2.0,))
    after = certify_gaussian_grouped_directional(weak_face, (-1,), (-2.0,))
    assert before is not None and after is not None
    assert before.value == (0,) and before.enclosure == (ZERO,) and before.error_bound == 0
    assert before.equality_multipliers == (Interval.point(-1),)
    assert after.value == (-2,) and after.enclosure[0].contains(-2)
    assert after.weak_slacks[0].lo > 0


def test_original_scalar_kkt_secants_show_noncommuting_bias_gap_and_target_step_limits():
    mp = pytest.importorskip("mpmath")
    with mp.workdps(100):
        # Original centers +/-1 give U_gap(x)=x-tanh((x+gap)/2).
        # Its derivative is >=1/2, so its root is unique. The constrained
        # minimum on x<=0 is zero exactly when step<=tanh(gap/2).
        def original_velocity(x, gap):
            logits = (-x*x/4-x/2, gap-x*x/4+x/2)
            weights = tuple(mp.exp(value) for value in logits)
            return x-(weights[1]-weights[0])/sum(weights)

        def minimum(gap, step):
            multiplier = mp.tanh(gap/2)-step
            if multiplier >= 0:
                assert abs(original_velocity(mp.mpf(0), gap)+step+multiplier) < mp.mpf("1e-90")
                return mp.mpf(0)
            root = mp.findroot(lambda x: original_velocity(x, gap)+step, (-2*step, -step/2))
            assert root < 0 and abs(original_velocity(root, gap)+step) < mp.mpf("1e-90")
            return root

        gap = mp.mpf(1)/8
        threshold = mp.tanh(gap/2)
        for divisor in (2, 16, 256):
            step = threshold/divisor
            assert minimum(gap, step)/step == 0

        # Sending the bias gap to zero first releases the wall. Holding
        # a target step fixed approaches the collapsed source's root.
        fixed_step = mp.mpf(1)/32
        collapsed_root = minimum(mp.mpf(0), fixed_step)
        roots = [minimum(mp.mpf(1)/denominator, fixed_step) for denominator in (8, 64, 512, 4096)]
        assert roots[0] == 0 and collapsed_root < roots[-1] < roots[-2] < roots[-3] < 0
        errors = [abs(root-collapsed_root) for root in roots[1:]]
        assert errors[1] < errors[0]/7 and errors[2] < errors[1]/7

        # At exact collapse, U_0'(0)=1/2, hence the negative-target
        # one-sided derivative is -2, rather than the positive-gap value 0.
        errors = []
        for denominator in (16, 256, 4096, 65536):
            step = mp.mpf(1)/denominator
            secant = minimum(mp.mpf(0), step)/step
            errors.append(abs(secant+2))
        assert all(later < earlier/200 for earlier, later in zip(errors, errors[1:], strict=False))
        assert errors[-1] < mp.mpf("1e-8")
