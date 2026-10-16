# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private optimization candidates: correctness does not imply promotion."""
import pytest
jax = pytest.importorskip("jax")
pytest.importorskip("mpmath")
jax.config.update("jax_enable_x64", True)
from dataclasses import replace
from fractions import Fraction
import hashlib
import math
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import mpmath as mp
import numpy as np
import pytest
from omnibias.jax._experimental_stage.bounds import Bound, interpret
from omnibias.jax._experimental_stage.feature_models import FeatureModel, heat_residual, heat_solution, exact_heat, train_heat_features
from omnibias.jax._experimental_stage.variable_projection import LinearResidual, certify_stationarity, _owned
from omnibias.core.operator_spec import OperatorSpec

def test_frozen_streamed_qr_preserves_primitive_metadata():
    from omnibias.jax._experimental_stage.planner import freeze_callable
    residual, hidden = _case('ridge', rows=5)
    dummy = jnp.zeros((1, 1), dtype=jnp.float64)

    def evaluate(parameters, points):
        result = residual.value_and_grad(parameters, chunk_size=2)
        return result[:2]
    frozen = freeze_callable(evaluate, (hidden, dummy))
    expected = jax.jit(evaluate)(hidden, dummy)
    actual = jax.jit(frozen)(hidden, dummy)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=1e-08, rtol=2e-09)

def _case(family, *, regularization=0.1, rows=7):
    rng = np.random.default_rng(9901)
    model = FeatureModel(family, 2, 1)
    hidden = tuple((jnp.asarray(rng.uniform(-0.45, 0.45, shape)) for shape in model.parameter_shapes))
    ops = (OperatorSpec.laplacian(0), OperatorSpec.laplacian(1)) if family == 'bottleneck' else (OperatorSpec.mixed((0, 0)), OperatorSpec.mixed((1, 0)), OperatorSpec.mixed((1, 1)))
    points = rng.uniform(-0.7, 0.7, (rows, 2))
    target = np.sin(points[:, 0]) + 0.1 * points[:, 1] ** 2
    coefficients = rng.uniform(-0.5, 0.5, (rows, len(ops)))
    coefficients[:, 0] += 1
    weights = np.linspace(0.5, 1.5, rows)
    weights[0] = 0.0
    return (LinearResidual(model, ops, points, target, weights, coefficients, regularization=regularization), hidden)

def _mp_arrays(flat, shapes):
    result, index = ([], 0)
    for shape in shapes:
        count = math.prod(shape)
        entries = list(flat[index:index + count])
        index += count
        if len(shape) == 2:
            entries = [entries[j * shape[1]:(j + 1) * shape[1]] for j in range(shape[0])]
        result.append(entries)
    return result

def _mp_design(residual, flat):
    model = residual.model
    pars = _mp_arrays(flat, model.parameter_shapes)

    def feature(feature_index, *point):
        if model.family == 'ridge':
            w, b = pars
            return mp.tanh(sum((a * x for a, x in zip(w[feature_index], point))) + b[feature_index])
        if model.family == 'separable_bank':
            w, b = pars
            return mp.tanh(sum((mp.tanh(a * x + bb) for a, x, bb in zip(w[feature_index], point, b[feature_index]))))
        a, b, w, h = pars
        latent = [mp.tanh(sum((aa * x for aa, x in zip(row, point))) + bb) for row, bb in zip(a, b)]
        return mp.tanh(sum((ww * z for ww, z in zip(w[feature_index], latent))) + h[feature_index])
    design, target = ([], [])
    for point, y, weight, coefficients in zip(np.asarray(residual.points), np.asarray(residual.target), np.asarray(residual.row_weights), np.asarray(residual.coefficients)):
        point = tuple((mp.mpf(float(v)) for v in point))
        scale = mp.sqrt(mp.mpf(float(weight)))
        row = []
        for h in range(model.features + int(model.include_offset)):
            total = mp.mpf(0)
            for op, factor in zip(residual.operators, coefficients):
                val = mp.mpf(0)
                for alpha, coefficient in op.resolve_terms(model.dimension):
                    coefficient = mp.mpf(int(coefficient.numerator)) / int(coefficient.denominator)
                    if h == model.features:
                        derivative = mp.mpf(not any(alpha))
                    else:
                        derivative = mp.diff(lambda *x: feature(h, *x), point, alpha)
                    val += coefficient * derivative
                total += mp.mpf(float(factor)) * val
            row.append(scale * total)
        design.append(row)
        target.append(scale * mp.mpf(float(y)))
    reg = mp.sqrt(mp.mpf(float(residual.regularization)))
    for h in range(residual.columns):
        design.append([reg if j == h else mp.mpf(0) for j in range(residual.columns)])
        target.append(mp.mpf(0))
    return (mp.matrix(design), mp.matrix(target))

def _mp_reference(residual, hidden):
    with mp.workdps(70):
        flat = [mp.mpf(float(v)) for value in hidden for v in np.asarray(value).ravel()]
        f, y = _mp_design(residual, flat)
        u, singular, v = mp.svd(f)
        projected = u.T * y
        c = v.T * mp.matrix([projected[i] / singular[i] for i in range(f.cols)])
        error = f * c - y
        value = mp.fdot(error, error) / 2

        def objective(z, coordinate):
            perturbed = flat.copy()
            perturbed[coordinate] = z
            f, y = _mp_design(residual, perturbed)
            r = f * c - y
            return mp.fdot(r, r) / 2
        gradients = [mp.diff(lambda z: objective(z, k), theta) for k, theta in enumerate(flat)]
        return (float(value), np.asarray(list(c), dtype=float), np.asarray(gradients, dtype=float))

@pytest.mark.parametrize('family', ('ridge', 'separable_bank', 'bottleneck'))
def test_independent_mp_reduced_value_complete_hidden_gradient(family):
    residual, hidden = _case(family)
    reference_value, reference_c, reference_gradient = _mp_reference(residual, hidden)
    for chunk in (None, 3):
        run = jax.jit(lambda p: residual.value_and_grad(p, chunk_size=chunk))
        value, gradient, c, domain = run(hidden)
        assert bool(domain)
        np.testing.assert_allclose(c, reference_c, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(value, reference_value, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(np.concatenate([np.asarray(v).ravel() for v in gradient]), reference_gradient, atol=1e-10, rtol=1e-09)
        report = residual.qualify(hidden, chunk_size=chunk)
        assert report['admissible'], report
        assert report['solve_certificate']['sigma_min_lower_bound'] > 0
        assert report['objective']['max_error_bound'] >= abs(float(value) - reference_value)
        if chunk is not None:
            assert report['emitted_graph']['primitives']['scan'] >= 1

def test_offset_mixed_and_zero_orders():
    model = FeatureModel('ridge', 2, 2)
    ops = (OperatorSpec.laplacian(0), OperatorSpec.laplacian(1), OperatorSpec.polynomial({(0, 0): 2, (1, 0): 3}))
    p = (jnp.array([[0.1, 0.2], [0.3, -0.4]]), jnp.array([0.2, -0.1]))
    result = model.build(ops)(p, jnp.ones((3, 2)))
    np.testing.assert_array_equal(result[0][:, -1], 1)
    np.testing.assert_array_equal(result[1][:, -1], 0)
    np.testing.assert_array_equal(result[2][:, -1], 2)

@pytest.mark.parametrize('family', ('ridge', 'separable_bank'))
def test_rational_operator_conversion_features_and_complete_hidden_gradient(family):
    residual, hidden = _case(family)
    operators = (OperatorSpec.polynomial({(0, 0): Fraction(1, 3), (1, 0): Fraction(2, 7)}), OperatorSpec.polynomial({(1, 1): Fraction(-5, 11)}), OperatorSpec.mixed((0, 0)))
    residual = replace(residual, operators=operators)
    feature = residual._feature
    assert feature.constant_bounds is feature.prepare.constant_bounds
    assert feature.constant_bounds is feature.prepared.constant_bounds
    assert len(feature.coefficient_conversion['nonexact_conversions']) == 3
    outputs = feature(hidden, residual.points)
    np.testing.assert_array_equal(outputs[0][:, -1], float(Fraction(1, 3)))
    traced = jax.make_jaxpr(residual.augmented_design)(hidden)
    (bounded_f, _), _ = interpret(traced, [Bound.point(v) for v in hidden], constant_bounds=feature.constant_bounds)
    actual_f, _ = residual.augmented_design(hidden)
    with mp.workdps(80):
        flat = [mp.mpf(float(v)) for value in hidden for v in np.asarray(value).ravel()]
        ideal_f, _ = _mp_design(residual, flat)
        for i in range(ideal_f.rows):
            for j in range(ideal_f.cols):
                ideal = ideal_f[i, j]
                assert mp.mpf(float(bounded_f.lo[i, j])) <= ideal <= mp.mpf(float(bounded_f.hi[i, j]))
                observed = abs(mp.mpf(float(actual_f[i, j])) - ideal)
                assert observed <= mp.mpf(float(bounded_f.error[i, j]))
    reference_value, reference_c, reference_gradient = _mp_reference(residual, hidden)
    for chunk in (None, 3):
        value, gradient, c, valid = jax.jit(lambda p: residual.value_and_grad(p, chunk_size=chunk))(hidden)
        assert bool(valid)
        np.testing.assert_allclose(value, reference_value, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(c, reference_c, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(np.concatenate([np.asarray(v).ravel() for v in gradient]), reference_gradient, atol=1e-10, rtol=1e-09)
        report = residual.qualify(hidden, chunk_size=chunk)
        assert report['admissible'], report
        assert report['coefficient_conversion'] == feature.coefficient_conversion
        assert report['coefficient_conversion']['nonexact_conversions'][0]['ideal'] == '1/3'

def test_biharmonic_features_independent_nested_spatial_ad():
    model = FeatureModel('bottleneck', 2, 2, include_offset=False)
    p = (jnp.array([[0.2, -0.3], [0.35, 0.1]]), jnp.array([0.1, -0.2]), jnp.array([[0.4, 0.2], [-0.1, 0.3]]), jnp.array([0.05, -0.08]))
    x = jnp.array([[0.1, -0.2], [-0.3, 0.4]])
    actual = model.build((OperatorSpec.laplacian(2),))(p, x)[0]
    cols = []
    for h in range(2):
        f = lambda xx: jnp.tanh(jnp.tanh(p[0] @ xx + p[1]) @ p[2][h] + p[3][h])
        lap = lambda xx: jnp.trace(jax.hessian(f)(xx))
        biharmonic = lambda xx: jnp.trace(jax.hessian(lap)(xx))
        cols.append(jax.vmap(biharmonic)(x))
    np.testing.assert_allclose(actual, jnp.stack(cols, axis=1), atol=1e-12, rtol=1e-11)

@pytest.mark.parametrize('family', ('ridge', 'bottleneck'))
def test_streamed_qr_square_feature_blocks_preserve_bias_axes(family):
    rng = np.random.default_rng(68031)
    for width in (2, 3):
        model = FeatureModel(family, 3, width)
        hidden = tuple((jnp.asarray(rng.uniform(-0.4, 0.4, shape)) for shape in model.parameter_shapes))
        points = rng.uniform(-0.7, 0.7, (2 * width - 1, 3))
        operators = (OperatorSpec.laplacian(0), OperatorSpec.laplacian(1))
        residual = LinearResidual(model, operators, points, rng.normal(size=points.shape[0]), coefficients=rng.uniform(-0.5, 0.5, (points.shape[0], 2)), regularization=0.1)
        f, y = map(np.asarray, residual.augmented_design(hidden))
        reference = np.linalg.lstsq(f, y, rcond=None)[0]
        full = np.column_stack((f, y))
        factor = np.asarray(jax.jit(lambda p: residual._factor(p, width))(hidden))
        np.testing.assert_allclose(factor.T @ factor, full.T @ full, atol=1e-12, rtol=1e-11)
        actual = residual.value_and_grad(hidden, chunk_size=width)
        np.testing.assert_allclose(actual[2], reference, atol=1e-11, rtol=1e-10)
        assert np.linalg.norm(f.T @ (f @ np.asarray(actual[2]) - y)) < 1e-11
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(residual.value_and_grad(hidden))):
            np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-10)
        assert residual.qualify(hidden, chunk_size=width)['admissible']

def test_rank_and_inexact_solution_rejection():
    f = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([1.0, 2.0, 3.0])
    good = certify_stationarity(Bound.point(f), Bound.point(y), np.array([1.0, 2.0]))
    assert good['admissible_rank'] and good['readout_error_norm_bound'] < 1e-12
    bad = certify_stationarity(Bound.point(f), Bound.point(y), np.array([1.01, 2.0]))
    assert bad['readout_error_norm_bound'] >= 0.01
    zero = certify_stationarity(Bound.point(np.zeros((3, 2))), Bound.point(y), np.zeros(2))
    assert not zero['admissible_rank']
    almost_singular = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12], [1.0, 1.0 - 1e-12]])
    assert not certify_stationarity(Bound.point(almost_singular), Bound.point(y), np.array([1.0, 0.0]))['admissible_rank']
    residual, p = _case('ridge')
    c = residual.value_and_grad(p)[2]
    injected = residual.qualify(p, solution_override=c + 0.05)
    assert not injected['admissible'] and injected['status'] == 'inconclusive_accuracy'
    degenerate = replace(residual, regularization=0.0)
    zero_hidden = tuple((jnp.zeros_like(v) for v in p))
    observed = degenerate.value_and_grad(zero_hidden, solver='svd')
    assert np.isfinite(np.asarray(observed[2])).all()
    assert degenerate.qualify(zero_hidden, solver='svd')['status'] == 'inconclusive_rank'

def test_domain_changes_and_immutable_data():
    residual, p = _case('ridge')
    boxes = tuple((Bound(np.asarray(v) - 0.01, np.asarray(v) + 0.01, 0) for v in p))
    bounded = replace(residual, hidden_domain=boxes)
    original_lower = np.asarray(bounded.hidden_domain[0].lo).copy()
    boxes[0].lo[...] = -100
    np.testing.assert_array_equal(bounded.hidden_domain[0].lo, original_lower)
    update = tuple((v + 0.001 for v in p))
    assert bounded.qualify(update)['admissible']
    assert bounded.qualify(tuple((v + 0.1 for v in p)))['status'] == 'domain_violation'
    source = np.asarray(residual.target).copy()
    snapshot = replace(residual, target=source)
    old = snapshot.source_digest
    source[:] = 100
    assert snapshot.source_digest == old
    np.testing.assert_array_equal(snapshot.target, residual.target)
    assert replace(snapshot, target=source).source_digest != old

def test_replace_batch_owns_new_data_and_preserves_original():
    residual, hidden = _case('ridge', regularization=0.0)
    old_digest = residual.source_digest
    old_points = np.asarray(residual.points).copy()
    new_points = old_points + 0.02
    new_target = np.sin(new_points[:, 0]) + 0.1 * new_points[:, 1] ** 2
    replacement = residual.replace_batch(new_points, new_target)
    assert replacement is not residual and replacement.source_digest != old_digest
    new_points[...] = 100
    new_target[...] = 100
    np.testing.assert_array_equal(residual.points, old_points)
    np.testing.assert_array_equal(replacement.points, old_points + 0.02)
    np.testing.assert_array_equal(replacement.row_weights, residual.row_weights)
    assert residual.source_digest == old_digest
    assert replacement.qualify(hidden, chunk_size=3)['admissible']
    weighted = replacement.replace_batch(replacement.points, replacement.target, weights=np.ones(replacement.target.shape))
    np.testing.assert_array_equal(weighted.row_weights, 1.0)
    with pytest.raises(ValueError, match='operator row schema'):
        residual.replace_batch(old_points[:-1], np.zeros(old_points.shape[0] - 1))

def test_per_update_certificate_and_input_domain_guard():
    residual, p = _case('ridge')
    bounded = replace(residual, input_domain=Bound(np.full((2,), -1.0), np.full((2,), 1.0), 0), hidden_domain=tuple((Bound(np.asarray(v) - 0.01, np.asarray(v) + 0.01, 0) for v in p)))
    for hidden in (p, tuple((v + 0.002 for v in p))):
        result, report = bounded.certified_value_and_grad(hidden, chunk_size=3)
        assert result is not None and report['admissible']
        assert report['scope'].startswith('pointwise')
        assert float(result[0]) > 0
    outside = tuple((v + 0.1 for v in p))
    result, report = bounded.certified_value_and_grad(outside)
    assert result is None and report['status'] == 'domain_violation'
    invalid = jax.jit(bounded.value_and_grad)(outside)
    assert not bool(invalid[3]) and np.isnan(float(invalid[0]))
    wrong_input = replace(bounded, input_domain=Bound(np.full((2,), 4.0), np.full((2,), 5.0), 0))
    result, report = wrong_input.certified_value_and_grad(p)
    assert result is None and report['status'] == 'domain_violation'

def test_exact_compiled_packet_qualification_without_another_qr(monkeypatch):
    residual, hidden = _case('ridge')
    compiled = jax.jit(lambda pp: residual.value_and_grad(pp, chunk_size=3)).lower(hidden).compile()
    packet = jax.block_until_ready(compiled(hidden))

    def forbidden_solve(*args, **kwargs):
        raise AssertionError('packet qualification must never execute another QR solve')
    monkeypatch.setattr(LinearResidual, '_solve', forbidden_solve)
    report = residual.qualify_result(hidden, packet, chunk_size=3)
    assert report['admissible'] and (not report['qualification_reexecutes_solver'])
    assert report['inspected_packet']['checked']
    metadata = report['inspected_packet']
    inspected = [packet[0], *packet[1]]
    assert metadata['output_sha256'] == [hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for v in inspected]
    assert metadata['output_shapes'] == [list(np.shape(v)) for v in inspected]
    assert metadata['output_dtypes'] == [np.asarray(v).dtype.str for v in inspected]
    assert metadata['valid'] is True
    accepted, report = residual.certified_value_and_grad(hidden, chunk_size=3, evaluator=compiled)
    assert accepted is not None and report['admissible']
    with pytest.raises(NotImplementedError):
        residual.certified_value_and_grad(hidden, evaluator=compiled, derivative_order=2)
    for actual, expected in zip(jax.tree.leaves(accepted), jax.tree.leaves(packet)):
        np.testing.assert_array_equal(actual, expected)
    bad_gradient = list(packet[1])
    bad_gradient[0] = bad_gradient[0] + 0.01
    tampered = (packet[0], tuple(bad_gradient), packet[2], packet[3])
    refused = residual.qualify_result(hidden, tampered, chunk_size=3)
    assert not refused['admissible'] and refused['status'] == 'inconclusive_packet'
    refused = residual.qualify_result(hidden, (packet[0] + 0.01, *packet[1:]), chunk_size=3)
    assert not refused['admissible'] and refused['status'] == 'inconclusive_packet'

def test_scopes_budgets_and_higher_derivative_refusal():
    residual, p = _case('ridge')
    with pytest.raises(NotImplementedError):
        replace(residual, nonlinear_in_readout=True)
    with pytest.raises(NotImplementedError):
        replace(residual, tied_hidden_readout=True)
    with pytest.raises(NotImplementedError):
        residual.value_and_grad(p, derivative_order=2)
    with pytest.raises(NotImplementedError):
        jax.grad(lambda theta: residual.value_and_grad(theta)[0])(p)
    assert residual.qualify(p, max_state_bytes=1)['status'] == 'analysis_budget'
    assert residual.qualify(p, max_equations=1)['status'] == 'inconclusive_analysis'

def test_complex_refusal_and_known_zero_offset():
    residual, p = _case('ridge', regularization=0.0)
    with pytest.raises(ValueError, match='real'):
        _owned(np.array([1.0 + 2j]))
    with pytest.raises(ValueError, match='real'):
        replace(residual, target=np.asarray(residual.target) + 1j)
    derivative_only = replace(residual, operators=(OperatorSpec.mixed((1, 0)),), coefficients=np.ones((residual.points.shape[0], 1)))
    result, report = derivative_only.certified_value_and_grad(p)
    assert result is None and 'annihilate the offset' in report['reason']
    without_offset = replace(derivative_only, model=replace(residual.model, include_offset=False))
    result, report = without_offset.certified_value_and_grad(p)
    assert result is not None and report['admissible']

@pytest.mark.parametrize('family', ('ridge', 'separable_bank', 'bottleneck'))
def test_changed_hidden_jit_vmap_and_stream_metric_adjoint(family):
    residual, p = _case(family, rows=8)
    fn = jax.jit(lambda hidden: residual.value_and_grad(hidden, chunk_size=3))
    changed = tuple((value * 1.03 + 0.002 for value in p))
    a, b = (fn(p), fn(changed))
    assert not np.allclose(a[2], b[2], atol=1e-12, rtol=1e-12)
    full = residual.value_and_grad(changed)
    for actual, expected in zip(jax.tree_util.tree_leaves(b), jax.tree_util.tree_leaves(full)):
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)
    batched = tuple((jnp.stack((aa, bb)) for aa, bb in zip(p, changed)))
    values = jax.jit(jax.vmap(lambda hidden: residual.value_and_grad(hidden)[0]))(batched)
    np.testing.assert_allclose(values, [a[0], b[0]], atol=1e-12, rtol=1e-10)

def test_weighted_unregularized_zero_residual_and_svd():
    residual, p = _case('ridge', regularization=0.0)
    f, _ = residual.design(p)
    exact_c = jnp.array([0.3, -0.2])
    unweighted = f @ exact_c / jnp.sqrt(jnp.where(residual.row_weights > 0, residual.row_weights, 1))
    exact = replace(residual, target=np.asarray(unweighted))
    for solver in ('qr', 'svd'):
        value, gradient, c, _ = exact.value_and_grad(p, solver=solver)
        np.testing.assert_allclose(c, exact_c, atol=1e-11, rtol=1e-11)
        assert float(value) < 1e-27
        assert all((np.linalg.norm(np.asarray(g)) < 1e-13 for g in gradient))
        assert exact.qualify(p, solver=solver)['admissible']
    nearly = replace(exact, target=np.asarray(exact.target) + 1e-09 * np.arange(exact.target.size))
    result, report = nearly.certified_value_and_grad(p, chunk_size=3)
    assert result is not None and report['admissible']
    assert 0 < float(result[0]) < 1e-12

def test_periodic_heat_design_against_physical_derivatives():
    model = FeatureModel('ridge', 3, 2)
    p = (jnp.array([[0.3, -0.2, 0.4], [-0.2, 0.5, 0.1]]), jnp.array([0.1, -0.1]))
    c = jnp.array([0.4, -0.3, 0.2])
    x = np.array([0.1, 1.3, 4.2])
    t = np.array([0.0, 0.2, 0.5])
    residual = heat_residual(model, x, t, 0.1, regularization=0.01)
    f, target = residual.design(p)

    def field(x, t):
        return heat_solution(model, p, c, jnp.atleast_1d(x), jnp.atleast_1d(t))[0]
    expected = jax.vmap(lambda xx, tt: jax.grad(field, 1)(xx, tt) - 0.1 * jax.grad(jax.grad(field, 0), 0)(xx, tt))(jnp.asarray(x), jnp.asarray(t))
    np.testing.assert_allclose(f @ c - target, expected, atol=1e-12, rtol=1e-12)
    initial = heat_solution(model, p, c, jnp.asarray(x), jnp.zeros_like(jnp.asarray(t)))
    np.testing.assert_allclose(initial, np.sin(x) + 0.5 * np.sin(2 * x), atol=1e-14)
    endpoint = heat_solution(model, p, c, jnp.array([0.0, 2 * np.pi]), jnp.array([0.3, 0.3]))
    np.testing.assert_allclose(endpoint[0], endpoint[1], atol=1e-14)
    assert residual.qualify(p, chunk_size=2)['admissible']
    np.testing.assert_allclose(exact_heat(x, np.zeros_like(t), 0.1), np.sin(x) + 0.5 * np.sin(2 * x))

def test_trainable_heat_point_certification_and_preserved_domain_failure():
    model = FeatureModel('ridge', 3, 1)
    hidden = (jnp.array([[0.3, -0.2, 0.4]]), jnp.array([0.1]))
    x = np.linspace(0.1, 6.0, 7)
    t = np.linspace(0.0, 0.5, 7)
    run = train_heat_features(model, hidden, x, t, x + 0.03, t, 0.1, steps=2, learning_rate=0.01, regularization=0.1, chunk_size=3)
    assert run['status'] == 'completed' and run['accepted_steps'] == 2
    assert all((row['certificate']['admissible'] for row in run['history']))
    assert np.isfinite(run['quality']['solution_relative_L2'])
    assert run['quality']['initial_error_max'] < 1e-14
    assert run['quality']['periodic_error_max'] < 1e-14
    exact_boxes = tuple((Bound.point(np.asarray(v)) for v in hidden))
    refused = train_heat_features(model, hidden, x, t, x + 0.03, t, 0.1, steps=2, learning_rate=0.01, regularization=0.1, hidden_domain=exact_boxes)
    assert refused['status'] == 'proposed_update_refused' and refused['accepted_steps'] == 0
    assert refused['refusal']['certificate']['status'] == 'domain_violation'

def test_overflow_refusal():
    residual, p = _case('ridge')
    large = tuple((jnp.full_like(v, 1e+200) for v in p))
    report = residual.qualify(large)
    assert not report['admissible']
