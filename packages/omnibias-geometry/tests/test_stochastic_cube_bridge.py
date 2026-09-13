# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent Haar, gauge and derivative controls for the original-link cube."""
from copy import deepcopy
from fractions import Fraction as Q

import mpmath as mp
import pytest
from omnibias.core.proof.certificate import make_certificate
from omnibias.geometry.gauge.stochastic.cube_bridge import (
    conditional_cube_feedback_budget,
    cube_bridge_geometry,
    linear_character_cube_control,
    linear_character_cube_point,
    replay_cube_bridge_certificate,
)
from omnibias.geometry.gauge.stochastic.lattice import quaternion_inverse as inv
from omnibias.geometry.gauge.stochastic.lattice import quaternion_product as mul

IDENTITY = (Q(1), Q(0), Q(0), Q(0))
ROTATIONS = [IDENTITY, (Q(3, 5), Q(4, 5), Q(0), Q(0)),
             (Q(5, 13), Q(0), Q(12, 13), Q(0)), (Q(0), Q(0), Q(0), Q(1))]


def test_geometry_and_conditional_budget() -> None:
    g = cube_bridge_geometry()
    assert len(g['old_edges']) == 4 and len(g['new_edges']) == 8
    assert g['face_incidences'] == {'old': 4, 'new': 16}
    row = conditional_cube_feedback_budget()
    a = row['arithmetic']
    assert Q(a['theta_upper']) == Q(2541, 4900)
    assert Q(a['constant_upper']) == Q(39273, 1280)
    assert a['target_theta_three_quarters_passed'] and a['target_constant_31_passed']
    assert a['external_premises'] and not a['physical_form_comparison_verified']
    assert replay_cube_bridge_certificate(row['certificate'])


@pytest.mark.parametrize('x', [Q(-1), Q(-2, 5), Q(0), Q(3, 7), Q(1)])
@pytest.mark.parametrize('r', [Q(-2, 3), Q(0), Q(1, 3), Q(4, 5)])
def test_integrated_control_against_independent_haar_quadrature(x: Q, r: Q) -> None:
    a = linear_character_cube_control(x, tilt=r)['arithmetic']
    with mp.workdps(45):
        c = mp.mpf(r.numerator)/r.denominator/(1+(mp.mpf(r.numerator)/r.denominator)**2)
        chi_b = 2*mp.mpf(x.numerator)/x.denominator
        z = 1+c**5*chi_b/16
        # Angular integration of f^{*4}(X^{-1}B); Haar conditional orientation has mean zero.
        def expectation(observable):
            def integrand(u):
                chi = 2*mp.cos(u)
                return 2/mp.pi*mp.sin(u)**2*(1+c*chi)*(1+c**4*chi*chi_b/16)*observable(chi)/z
            return mp.quad(integrand, [0, mp.pi/2, mp.pi])
        assert abs(expectation(lambda chi: 1)-1) < mp.mpf('1e-40')
        checks = {'magnetic_one_face': expectation(lambda chi: 2-chi),
                  'one_face_fisher': expectation(lambda chi: c*c*(1-chi*chi/4)/(1+c*chi)**2)}
        for key, expected in checks.items():
            q = Q(a[key])
            assert abs(mp.mpf(q.numerator)/q.denominator-expected) < mp.mpf('1e-38')
    assert Q(a['new_link_fisher']) >= 0 and Q(a['old_link_fisher']) >= 0


def test_haar_profile_and_full_certificate_replay() -> None:
    row = linear_character_cube_control(Q(-1), tilt=0, kappa=Q(1, 8))
    assert row['arithmetic']['total_fisher'] == '0'
    assert row['arithmetic']['energy'] == '160'
    cert = row['certificate']
    assert replay_cube_bridge_certificate(cert)
    for field in ('payload', 'claim', 'meta'):
        bad = deepcopy(cert)
        if field == 'payload':
            bad[field]['arithmetic']['physical_heat_feedback_verified'] = True
        elif field == 'claim':
            bad[field] = 'continuum solved'
        else:
            bad[field]['transcend_backend'] = 'forged'
        resealed = make_certificate(claim=bad['claim'], payload=bad['payload'], meta=bad['meta'])
        assert not replay_cube_bridge_certificate(resealed)


def test_original_link_gauge_covariance() -> None:
    links = [ROTATIONS[i % 4] for i in range(12)]
    gauges = [ROTATIONS[(i+1) % 4] for i in range(8)]
    edges = cube_bridge_geometry()['edges']
    transformed = [mul(mul(gauges[u], q), inv(gauges[v])) for (u, v), q in zip(edges, links, strict=True)]
    a = linear_character_cube_point(links)['arithmetic']
    row = linear_character_cube_point(transformed)
    b = row['arithmetic']
    for key in ('density', 'bottom_trace', 'old_log_amplitude_gradient_squared', 'new_log_amplitude_gradient_squared'):
        assert a[key] == b[key]
    assert replay_cube_bridge_certificate(row['certificate'])


def test_all_original_link_derivatives_against_rational_group_curve() -> None:
    links = [ROTATIONS[i % 4] for i in range(12)]
    a = linear_character_cube_point(links)['arithmetic']
    rho, h = Q(a['density']), Q(1, 10**6)
    for e in range(12):
        for axis in range(3):
            values = []
            for sign in (1, -1):
                # q'(0)=2*i*sigma_axis=4*T_axis in the Casimir3/4 convention.
                rotation = [(1-h*h)/(1+h*h), *[2*sign*h/(1+h*h) if j == axis else Q(0) for j in range(3)]]
                perturbed = list(links)
                perturbed[e] = mul(rotation, links[e])
                values.append(Q(linear_character_cube_point(perturbed)['arithmetic']['density']))
            oracle = (values[0]-values[1])/(16*h*rho)
            analytic = Q(a['log_amplitude_gradient'][e][axis])
            assert abs(oracle-analytic) < 100*h*h


@pytest.mark.parametrize('x', [True, 0.5, Q(2)])
def test_invalid_exact_inputs_refused(x) -> None:
    with pytest.raises((ValueError, TypeError)):
        linear_character_cube_control(x)
    assert not replay_cube_bridge_certificate({'payload': {}})


def test_two_face_seam_retains_shared_old_edge_but_differentiates_boundary_only() -> None:
    from omnibias.geometry.gauge.stochastic.cube_bridge import (
        conditional_two_face_seam_budget,
        cube_disk_geometry,
    )
    g = cube_disk_geometry(2)
    assert len(g['old_edges']) == 7 and len(g['new_edges']) == 5
    assert len(g['boundary']) == 6 and g['new_vertices'] == 2
    assert g['face_incidences'] == {'old': 6, 'new': 10}
    assert len(set(g['old_edges'])-set(map(abs, g['boundary']))) == 1
    for count, new_vertices, new_edges, incidences in ((1, 4, 8, 16), (3, 1, 3, 6)):
        other = cube_disk_geometry(count)
        assert other['new_vertices'] == new_vertices and len(other['new_edges']) == new_edges
        assert other['face_incidences']['new'] == incidences
    row = conditional_two_face_seam_budget()
    assert Q(row['arithmetic']['theta_upper']) == Q(4235, 6272)
    assert Q(row['arithmetic']['constant_upper']) == Q(12179, 512)
    assert replay_cube_bridge_certificate(row['certificate'])


def test_two_face_global_angle_inequality_dense_grid_and_random_products() -> None:
    import math
    import random
    rng = random.Random(906)
    samples = [(math.pi*i/32, math.pi*j/32, dot)
               for i in range(33) for j in range(33) for dot in (-1, 0, 1)]
    samples += [(rng.random()*math.pi, rng.random()*math.pi, rng.uniform(-1, 1)) for _ in range(512)]
    for u, v, dot in samples:
        trace_half = math.cos(u)*math.cos(v)-dot*math.sin(u)*math.sin(v)
        angle = math.acos(min(1, max(-1, trace_half)))
        assert angle**2 <= math.pi**2/4*(4-2*math.cos(u)-2*math.cos(v))+1e-13


@pytest.mark.parametrize('constant,theta', [(0, Q(0)), (100, Q(3, 4)), (10**8, Q(999999, 1000000))])
def test_three_face_strict_budget_has_exact_violating_coupling(constant, theta) -> None:
    from omnibias.geometry.gauge.stochastic.cube_bridge import (
        three_face_strict_feedback_obstruction,
    )
    row = three_face_strict_feedback_obstruction(constant=constant, theta=theta)
    a = row['arithmetic']
    assert 0 < Q(a['counterexample_kappa']) <= Q(1, 64)
    assert Q(a['violation_margin']) > 0
    assert Q(a['energy_lower']) > Q(a['proposed_energy_upper'])
    assert a['candidate_status'] == 'DISPROVED' and not a['mass_gap_disproved']
    assert not a['optimized_ground_energy_comparison_disproved']
    assert replay_cube_bridge_certificate(row['certificate'])


def test_three_face_minimum_dense_angles_and_random_noncommuting_products() -> None:
    import math
    import random
    rng = random.Random(3887)
    samples = [(math.pi*i/24, math.pi*j/24, dot)
               for i in range(25) for j in range(25) for dot in (-1, 0, 1)]
    samples += [(rng.random()*math.pi, rng.random()*math.pi, rng.uniform(-1, 1)) for _ in range(512)]
    for u, v, dot in samples:
        trace_half = math.cos(u)*math.cos(v)-dot*math.sin(u)*math.sin(v)
        # X3=-(X1 X2)^-1 enforces the nonabelian product constraint exactly.
        action = 6-2*(math.cos(u)+math.cos(v)-trace_half)
        assert action >= 3-1e-13
    u = math.pi/3
    assert abs(6-2*(2*math.cos(u)-math.cos(2*u))-3) < 1e-14
