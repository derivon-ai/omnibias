# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private optimization candidates: correctness does not imply promotion."""
import pytest
jax = pytest.importorskip("jax")
pytest.importorskip("mpmath")
jax.config.update("jax_enable_x64", True)
import unittest
from fractions import Fraction
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
from omnibias.jax._experimental_stage.bundles import Model, build
from omnibias.jax._experimental_stage.device_training import Boxes, DOMAIN_EXIT, INVALID_INPUT, NONFINITE, OK, TrainingState, call_donated_training, copy_state_for_donation, jit_donated_training, initial_state, make_adam_step, make_training_block, recenter_boxes, resume_after_qualification
from omnibias.core.operator_spec import OperatorSpec
from omnibias.jax._experimental_stage.prefetch import DoubleBuffer, FrozenDependencies, HostPrepared, SynchronousBuffer
from omnibias.jax._experimental_stage.reduction import SpanLimits, build_reduced, build_reduced_streaming, operator_span
from omnibias.jax._experimental_stage.shared_models import SharedModel, build_shared
from omnibias.jax._experimental_stage.streaming import StatisticObjective, SumObjective, WeightedMeanObjective, build_streaming

def case(family):
    model = Model(family, 2, hidden=2)
    if family == 'ridge':
        params = (np.array([[0.3, -0.2], [-0.1, 0.4]]), np.array([0.13, -0.07]), np.array([0.4, -0.3]), np.asarray(0.04))
        ops = (OperatorSpec.laplacian(0), OperatorSpec.laplacian(1), OperatorSpec.mixed((1, 0)))
    elif family == 'separable':
        params = (np.array([0.3, -0.2]), np.array([0.13, -0.07]))
        ops = (OperatorSpec.mixed((0, 0)), OperatorSpec.mixed((1, 1)))
    else:
        params = (np.array([[0.3, -0.2], [-0.1, 0.4]]), np.array([0.13, -0.07]), np.array([[0.3, -0.2], [0.15, 0.4]]), np.array([0.05, -0.03]), np.array([0.4, -0.3]), np.asarray(0.04))
        ops = (OperatorSpec.laplacian(0), OperatorSpec.laplacian(1), OperatorSpec.laplacian(2))
    points = jnp.asarray([[0.2, -0.1], [-0.3, 0.05], [0.1, 0.3]])
    return (model, ops, jax.tree.map(jnp.asarray, params), points)

def row_loss(params, point, values):
    return sum(((v - 0.1 * point[0]) ** 2 for v in values)) + 0.01 * jnp.sum(params[0]) * point[0]

def regularizer(params):
    return 0.003 * sum((jnp.sum(p * p) for p in params))

def objectives():
    return (SumObjective(row_loss, regularizer), WeightedMeanObjective(row_loss, lambda p, x: 1 + 0.02 * jnp.sum(p[0] ** 2) + 0.05 * x[0], regularizer), StatisticObjective(lambda p, x, y: jnp.stack((row_loss(p, x, y), y[0])), lambda p, s, n: s[0] / n + 0.1 * (s[1] / n) ** 2 + regularizer(p)))

def assert_tree(test, a, b, *, atol=1e-08, rtol=2e-09):
    aa, ad = jax.tree.flatten(a)
    bb, bd = jax.tree.flatten(b)
    test.assertEqual(ad, bd)
    for x, y in zip(aa, bb):
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

def mp_reference(model, ops, params, points):
    """Differentiate the raw tanh model with mpmath, then its complete loss.

    No derivative polynomial, contracted metric, JAX differentiation or
    streaming implementation participates in this independent reference.
    """
    shapes = [p.shape for p in params]
    sizes = [p.size for p in params]
    flat = [mp.mpf(float(v)) for p in params for v in np.asarray(p).flat]
    xs = [[mp.mpf(float(v)) for v in row] for row in np.asarray(points)]

    def unpack(values):
        arrays, offset = ([], 0)
        for shape, size in zip(shapes, sizes):
            arrays.append(np.array(values[offset:offset + size], dtype=object).reshape(shape))
            offset += size
        return arrays

    def raw(p, x):
        if model.family == 'ridge':
            w, b, c, d = p
            return sum((c[h] * mp.tanh(sum((w[h, i] * x[i] for i in range(2))) + b[h]) for h in range(2))) + d.item()
        if model.family == 'separable':
            w, b = p
            return mp.tanh(sum((mp.tanh(w[i] * x[i] + b[i]) for i in range(2))))
        a, b, w, hb, c, d = p
        latent = [mp.tanh(sum((a[k, i] * x[i] for i in range(2))) + b[k]) for k in range(2)]
        return sum((c[h] * mp.tanh(sum((w[h, k] * latent[k] for k in range(2))) + hb[h]) for h in range(2))) + d.item()

    def loss(values):
        p = unpack(values)
        total = mp.mpf('0')
        for x in xs:
            ys = []
            for op in ops:
                ys.append(sum((mp.mpf(c.numerator) / c.denominator * mp.diff(lambda a, b: raw(p, (a, b)), tuple(x), alpha) for alpha, c in op.resolve_terms(2))))
            total += sum(((y - mp.mpf(float(0.1)) * x[0]) ** 2 for y in ys))
            total += mp.mpf(float(0.01)) * sum(p[0].flat) * x[0]
        return total + mp.mpf(float(0.003)) * sum((v * v for v in values))
    with mp.workdps(65):
        value = loss(flat)
        gradients = []
        for index, number in enumerate(flat):

            def varied(v):
                changed = list(flat)
                changed[index] = v
                return loss(changed)
            gradients.append(mp.diff(varied, number))
    return (float(value), tuple((np.asarray(p, dtype=float) for p in unpack(gradients))))

class StreamingChecks(unittest.TestCase):

    def test_three_families_all_objectives_ragged_jit_and_changed_weights(self):
        for family in ('ridge', 'separable', 'bottleneck'):
            model, ops, params, points = case(family)
            full = build(model, ops, 'parity')
            for objective in objectives():
                stream = build_streaming(model, ops, objective, chunk_size=2)
                expected = jax.value_and_grad(lambda p: objective.full_objective(p, points, full(p, points)))(params)
                actual = jax.jit(stream.value_and_grad)(params, points)
                assert_tree(self, actual, expected)
                assert_tree(self, jax.jit(stream.full_outputs)(params, points), full(params, points), atol=1e-09, rtol=2e-10)
                np.testing.assert_allclose(stream.value(params, points), expected[0], atol=1e-09, rtol=2e-10)
                changed = jax.tree.map(lambda p: p + 0.002, params)
                expected_changed = jax.value_and_grad(lambda p: objective.full_objective(p, points, full(p, points)))(changed)
                assert_tree(self, jax.jit(stream.value_and_grad)(changed, points), expected_changed)

    def test_complete_independent_high_precision_parameter_gradients(self):
        for family in ('ridge', 'separable', 'bottleneck'):
            model, ops, params, points = case(family)
            stream = build_streaming(model, ops, objectives()[0], chunk_size=2)
            actual = jax.jit(stream.value_and_grad)(params, points)
            reference = mp_reference(model, ops, params, points)
            np.testing.assert_allclose(actual[0], reference[0], atol=1e-09, rtol=2e-10)
            assert_tree(self, actual[1], reference[1])

    def test_shared_parameter_roles_and_vmap(self):
        model, ops, params, points = case('ridge')
        shared = SharedModel(model, (0, 1, 1, 2))
        unique = (params[0], params[1], params[3])
        full = build_shared(shared, ops)
        objective = objectives()[0]
        stream = build_streaming(shared, ops, objective, chunk_size=2)
        expected = jax.value_and_grad(lambda p: objective.full_objective(p, points, full(p, points)))(unique)
        assert_tree(self, jax.jit(stream.value_and_grad)(unique, points), expected)
        results = jax.jit(jax.vmap(stream.value_and_grad, in_axes=(None, 0)))(unique, jnp.stack((points, points * 0.9)))
        assert_tree(self, jax.tree.map(lambda a: a[0], results), expected)

    def test_zero_factors_and_one_chunk(self):
        model, ops, params, points = case('separable')
        params = (params[0].at[0].set(0.0), params[1])
        objective = objectives()[0]
        full = build(model, ops)
        stream = build_streaming(model, ops, objective, chunk_size=100)
        expected = jax.value_and_grad(lambda p: objective.full_objective(p, points, full(p, points)))(params)
        assert_tree(self, stream.value_and_grad(params, points), expected)
        self.assertTrue(all((np.isfinite(a).all() for a in jax.tree.leaves(expected))))

    def test_scan_does_not_return_collocation_auxiliaries(self):
        model, ops, params, points = case('bottleneck')
        stream = build_streaming(model, ops, objectives()[0], chunk_size=2)
        graph = jax.make_jaxpr(stream.value_and_grad)(params, points)
        scans = [equation for equation in graph.jaxpr.eqns if equation.primitive.name == 'scan']
        self.assertEqual(len(scans), 1)
        self.assertEqual(len(scans[0].outvars), scans[0].params['num_carry'])
        self.assertEqual(len(graph.jaxpr.outvars), len(params) + 1)

    def test_explicit_contract_and_shape_refusals(self):
        model, ops, params, points = case('ridge')
        with self.assertRaises(TypeError):
            build_streaming(model, ops, lambda *args: 0.0)
        with self.assertRaises(ValueError):
            build_streaming(model, ops, objectives()[0], chunk_size=0)
        with self.assertRaises(ValueError):
            build_streaming(model, ops, objectives()[0]).value(params, points[:0])

    def test_actual_emitted_sum_gradient_conditional_bounds(self):
        from omnibias.jax._experimental_stage.bounds import Bound, interpret
        for family in ('ridge', 'separable', 'bottleneck'):
            model, ops, params, points = case(family)
            program = build_streaming(model, ops, objectives()[0], chunk_size=2)
            boxes = [Bound(np.asarray(p) - 1e-08, np.asarray(p) + 1e-08, 0.0) for p in params]
            boxes.append(Bound(np.asarray(points), np.asarray(points), 0.0))
            outputs, meta = interpret(jax.make_jaxpr(program.value_and_grad)(params, points), boxes)
            self.assertTrue(outputs[0].summary(1e-09, 2e-10)['admissible'])
            self.assertTrue(all((g.summary(1e-08, 2e-09)['admissible'] for g in outputs[1:])))
            self.assertIn('scan', meta['primitives'])

class DeviceTrainingChecks(unittest.TestCase):

    def setUp(self):
        self.params = (jnp.asarray([0.3, -0.2]),)
        self.points = jnp.asarray([[0.2], [0.4]])
        self.vg = jax.value_and_grad(lambda p, x: jnp.sum((p[0] - jnp.mean(x)) ** 2))
        self.boxes = Boxes((jnp.full((2,), -1.0),), (jnp.full((2,), 1.0),), jnp.asarray([-1.0]), jnp.asarray([1.0]))

    def test_block_matches_host_loop(self):
        step = make_adam_step(self.vg)
        state = initial_state(self.params)
        expected = state
        for _ in range(5):
            expected = step(expected, self.points, self.boxes).accepted
        actual = jax.jit(make_training_block(step, 5))(state, jnp.broadcast_to(self.points, (5,) + self.points.shape), self.boxes)
        self.assertEqual(int(actual.status), OK)
        self.assertTrue(np.all(actual.evaluated))
        assert_tree(self, actual.accepted, expected, atol=1e-13, rtol=1e-13)

    def test_exit_rolls_back_moments_and_stops_block(self):
        state = initial_state(self.params)
        narrow = Boxes(jax.tree.map(lambda p: p - 1e-08, self.params), jax.tree.map(lambda p: p + 1e-08, self.params), self.boxes.input_lower, self.boxes.input_upper)
        block = jax.jit(make_training_block(make_adam_step(self.vg), 5))
        result = block(state, jnp.broadcast_to(self.points, (5,) + self.points.shape), narrow)
        self.assertEqual(int(result.status), DOMAIN_EXIT)
        self.assertEqual(int(np.sum(result.evaluated)), 1)
        assert_tree(self, result.accepted, state, atol=0, rtol=0)
        enlarged = recenter_boxes(narrow, result.proposed.params, result.delta)
        with self.assertRaises(ValueError):
            resume_after_qualification(result, enlarged, admissible=False)
        resumed = resume_after_qualification(result, enlarged, admissible=True)
        self.assertEqual(int(resumed.accepted_steps), 1)
        self.assertTrue(any((np.any(a) for a in resumed.first_moment)))

    def test_invalid_input_never_evaluated_and_nonfinite_rolls_back(self):
        state = initial_state(self.params)
        step = jax.jit(make_adam_step(self.vg))
        invalid = step(state, self.points * 100.0, self.boxes)
        self.assertEqual(int(invalid.status), INVALID_INPUT)
        self.assertFalse(bool(invalid.evaluated))

        def bad_vg(p, x):
            return (jnp.asarray(float('nan')), jax.tree.map(lambda a: a * float('nan'), p))
        result = jax.jit(make_adam_step(bad_vg))(state, self.points, self.boxes)
        self.assertEqual(int(result.status), NONFINITE)
        assert_tree(self, result.accepted, state, atol=0, rtol=0)

    def test_real_evaluation_callback_stops_at_domain_boundary(self):
        calls = []

        def observed_vg(p, x):
            jax.debug.callback(lambda value: calls.append(float(value)), p[0][0], ordered=True)
            return self.vg(p, x)
        state = initial_state(self.params)
        narrow = Boxes(jax.tree.map(lambda p: p - 1e-08, self.params), jax.tree.map(lambda p: p + 1e-08, self.params), self.boxes.input_lower, self.boxes.input_upper)
        block = jax.jit(make_training_block(make_adam_step(observed_vg), 5))
        result = block(state, jnp.broadcast_to(self.points, (5,) + self.points.shape), narrow)
        jax.block_until_ready(result)
        jax.effects_barrier()
        self.assertEqual(len(calls), 1)
        invalid = jax.jit(make_adam_step(observed_vg))(state, self.points * 100, self.boxes)
        jax.block_until_ready(invalid)
        jax.effects_barrier()
        self.assertEqual(len(calls), 1)

class DonationChecks(unittest.TestCase):

    def setUp(self):
        self.params = (jnp.asarray([0.3, -0.2]),)
        self.points = jnp.asarray([[0.2], [0.4]])
        self.vg = jax.value_and_grad(lambda p, x: jnp.sum((p[0] - jnp.mean(x)) ** 2))
        self.boxes = Boxes((jnp.full((2,), -1.0),), (jnp.full((2,), 1.0),), jnp.asarray([-1.0]), jnp.asarray([1.0]))

    def test_copy_owns_every_leaf_without_host_parameter_access(self):
        repeated = (self.params[0], self.params[0])
        state = initial_state(repeated)
        self.assertIs(state.first_moment, state.second_moment)
        owned, report = copy_state_for_donation(state)
        source_ptrs = {value.unsafe_buffer_pointer() for value in jax.tree.leaves(state)}
        pointers = [value.unsafe_buffer_pointer() for value in jax.tree.leaves(owned)]
        self.assertEqual(len(set(pointers)), len(pointers))
        self.assertFalse(source_ptrs.intersection(pointers))
        self.assertEqual(report['host_parameter_transfers'], 0)
        self.assertEqual(report['parameter_hashes'], 0)
        self.assertEqual(report['copied_bytes'], sum((v.nbytes for v in jax.tree.leaves(state))))
        self.assertGreaterEqual(report['copy_elapsed_s'], report['copy_dispatch_s'])
        assert_tree(self, state, owned, atol=0, rtol=0)
        with self.assertRaises(TypeError):
            copy_state_for_donation(TrainingState((np.ones(2),), state.first_moment, state.second_moment, state.accepted_steps))

    def test_accepted_update_second_step_reuse_and_gradient_science(self):
        step = make_adam_step(self.vg)
        ordinary, donated = (jax.jit(step), jit_donated_training(step))
        state = initial_state(self.params)
        source = jax.tree.map(lambda value: np.array(value), state)
        reference = ordinary(state, self.points, self.boxes)
        result, report = call_donated_training(donated, state, self.points, self.boxes)
        assert_tree(self, result, reference, atol=1e-13, rtol=1e-13)
        assert_tree(self, state, source, atol=0, rtol=0)
        self.assertEqual(int(result.status), OK)
        self.assertGreaterEqual(report['total_elapsed_s'], report['copy_elapsed_s'])
        preserved = jax.tree.map(lambda value: np.array(value), result)
        next_result, _ = call_donated_training(donated, result.accepted, self.points, self.boxes)
        next_reference = ordinary(reference.accepted, self.points, self.boxes)
        assert_tree(self, next_result, next_reference, atol=1e-13, rtol=1e-13)
        assert_tree(self, result, preserved, atol=0, rtol=0)
        loss, gradient = self.vg(next_result.accepted.params, self.points)
        analytic = 2 * (next_result.accepted.params[0] - jnp.mean(self.points))
        np.testing.assert_allclose(gradient[0], analytic, atol=1e-13, rtol=1e-13)
        self.assertTrue(np.isfinite(loss))

    def test_first_step_domain_exit_rolls_back_moments_in_donated_block(self):
        state = initial_state(self.params)
        narrow = Boxes(jax.tree.map(lambda p: p - 1e-08, self.params), jax.tree.map(lambda p: p + 1e-08, self.params), self.boxes.input_lower, self.boxes.input_upper)
        block = make_training_block(make_adam_step(self.vg), 3)
        batches = jnp.broadcast_to(self.points, (3,) + self.points.shape)
        result, _ = call_donated_training(jit_donated_training(block), state, batches, narrow)
        self.assertEqual(int(result.status), DOMAIN_EXIT)
        np.testing.assert_array_equal(result.evaluated, [True, False, False])
        assert_tree(self, result.accepted, state, atol=0, rtol=0)
        self.assertEqual(int(result.accepted.accepted_steps), 0)
        self.assertEqual(int(result.proposed.accepted_steps), 1)
        assert_tree(self, result, jax.jit(block)(state, batches, narrow), atol=1e-13, rtol=1e-13)

    def test_invalid_nonfinite_and_zero_gradients_keep_transactional_semantics(self):
        state = initial_state(self.params)
        invalid, _ = call_donated_training(jit_donated_training(make_adam_step(self.vg)), state, self.points * 100, self.boxes)
        self.assertEqual(int(invalid.status), INVALID_INPUT)
        self.assertFalse(bool(invalid.evaluated))
        assert_tree(self, invalid.accepted, state, atol=0, rtol=0)

        def bad(p, x):
            return (jnp.asarray(float('nan')), jax.tree.map(lambda v: v * float('nan'), p))
        nonfinite, _ = call_donated_training(jit_donated_training(make_adam_step(bad)), state, self.points, self.boxes)
        self.assertEqual(int(nonfinite.status), NONFINITE)
        self.assertTrue(bool(nonfinite.evaluated))
        assert_tree(self, nonfinite.accepted, state, atol=0, rtol=0)

        def zero(p, x):
            return (jnp.asarray(0.0, x.dtype), jax.tree.map(jnp.zeros_like, p))
        result, _ = call_donated_training(jit_donated_training(make_adam_step(zero)), state, self.points, self.boxes)
        self.assertEqual(int(result.status), OK)
        assert_tree(self, result.accepted.params, state.params, atol=0, rtol=0)
        assert_tree(self, result.accepted.first_moment, state.first_moment, atol=0, rtol=0)
        assert_tree(self, result.accepted.second_moment, state.second_moment, atol=0, rtol=0)
        self.assertEqual(int(result.accepted.accepted_steps), 1)

    def test_donated_block_stops_at_invalid_second_batch(self):
        state = initial_state(self.params)
        batches = jnp.stack((self.points, self.points * 100, self.points))
        block = make_training_block(make_adam_step(self.vg), 3)
        result, _ = call_donated_training(jit_donated_training(block), state, batches, self.boxes)
        reference = jax.jit(block)(state, batches, self.boxes)
        assert_tree(self, result, reference, atol=1e-13, rtol=1e-13)
        self.assertEqual(int(result.status), INVALID_INPUT)
        self.assertEqual(int(result.accepted.accepted_steps), 1)
        np.testing.assert_array_equal(result.evaluated, [True, False, False])

class PrefetchChecks(unittest.TestCase):

    def test_host_annotations_are_immutable_untransferred_and_budgeted(self):

        def prepare(batch, frozen):
            return HostPrepared(batch * 2, {'bound': batch * 3})
        for cls in (SynchronousBuffer, DoubleBuffer):
            with cls([np.asarray([1.0, 2.0])], prepare, FrozenDependencies({})) as stream:
                record = next(stream)
            self.assertIsInstance(record.payload, jax.Array)
            metadata = record.host_metadata.values['bound']
            self.assertIsInstance(metadata, np.ndarray)
            np.testing.assert_array_equal(metadata, [3.0, 6.0])
            with self.assertRaises(ValueError):
                metadata.setflags(write=True)
            self.assertEqual(record.timeline()['host_metadata_bytes'], 16)
            with self.assertRaises(MemoryError):
                with cls([np.asarray([1.0, 2.0])], prepare, FrozenDependencies({}), max_host_bytes=80) as stream:
                    next(stream)

    def test_ownership_order_dependency_tokens(self):
        original = np.asarray([2.0, 3.0])
        frozen = FrozenDependencies({'scale': original})
        original[:] = 99
        with self.assertRaises(ValueError):
            frozen.values['scale'].setflags(write=True)
        batches = [np.asarray([float(i), 1.0]) for i in range(3)]
        with DoubleBuffer(batches, lambda b, f: b * f['scale'], frozen) as pipeline:
            results = list(pipeline)
        self.assertEqual([r.index for r in results], [0, 1, 2])
        for index, result in enumerate(results):
            np.testing.assert_array_equal(result.require_dependencies(frozen), [2 * index, 3.0])
            self.assertGreaterEqual(result.wait()['consumer_wait_s'], 0.0)
        with self.assertRaises(ValueError):
            results[0].require_dependencies(FrozenDependencies({'scale': np.asarray([3.0, 4.0])}))

    def test_generator_and_preparation_failures_close_worker(self):

        def broken():
            yield np.asarray([1.0])
            raise RuntimeError('preserved producer failure')
        pipeline = DoubleBuffer(broken(), lambda b, f: b, FrozenDependencies({}))
        next(pipeline)
        with self.assertRaisesRegex(RuntimeError, 'preserved producer failure'):
            next(pipeline)
        self.assertTrue(pipeline._closed)
        with self.assertRaises(StopIteration):
            next(pipeline)

    def test_captured_arrays_and_memory_refused(self):
        live = np.ones(2)
        with self.assertRaises(ValueError):
            DoubleBuffer([live], lambda b, f: b + live, FrozenDependencies({}))
        with DoubleBuffer([live], lambda b, f: b, FrozenDependencies({}), max_host_bytes=8) as pipeline:
            with self.assertRaises(MemoryError):
                next(pipeline)

class ReductionChecks(unittest.TestCase):

    def test_nonbinary_basis_coefficients_join_reconstruction_bindings(self):
        from omnibias.jax._experimental_stage.bounds import Bound, interpret
        model, _, params, points = case('ridge')
        op = OperatorSpec.polynomial({(1, 0): Fraction(1, 3)})
        ops = (op, op)
        ordinary = build(model, ops)
        reduced, _ = build_reduced(model, ops)
        stream, _ = build_reduced_streaming(model, ops, objectives()[0], chunk_size=2)
        self.assertGreaterEqual(len(reduced.constant_bounds.legacy), 2)
        self.assertGreaterEqual(len(stream.constant_bounds.legacy), 2)
        assert_tree(self, ordinary(params, points), reduced(params, points))
        boxes = [Bound(np.asarray(p) - 1e-08, np.asarray(p) + 1e-08, 0.0) for p in params]
        boxes.append(Bound(np.asarray(points), np.asarray(points), 0.0))
        outputs, _ = interpret(jax.make_jaxpr(stream.value_and_grad)(params, points), boxes, constant_bounds=stream.constant_bounds)
        self.assertTrue(outputs[0].summary(1e-09, 2e-10)['admissible'])
        self.assertTrue(all((v.summary(1e-08, 2e-09)['admissible'] for v in outputs[1:])))

    def test_streamed_span_full_outputs_and_emitted_gradients_qualified(self):
        from omnibias.jax._experimental_stage.bounds import Bound, interpret
        model, _, params, points = case('ridge')
        ops = (OperatorSpec.polynomial({(1, 0): 3}), OperatorSpec.mixed((1, 0)), OperatorSpec.laplacian(0))
        ordinary = build(model, ops)
        for objective in objectives():
            stream, span = build_reduced_streaming(model, ops, objective, chunk_size=2)
            expected = jax.value_and_grad(lambda p: objective.full_objective(p, points, ordinary(p, points)))(params)
            assert_tree(self, jax.jit(stream.value_and_grad)(params, points), expected)
            assert_tree(self, stream.full_outputs(params, points), ordinary(params, points))
            boxes = [Bound(np.asarray(p) - 1e-08, np.asarray(p) + 1e-08, 0.0) for p in params]
            boxes.append(Bound(np.asarray(points), np.asarray(points), 0.0))
            outputs, _ = interpret(jax.make_jaxpr(stream.full_outputs)(params, points), boxes, constant_bounds=span.constant_bounds())
            self.assertTrue(all((v.summary(1e-09, 2e-10)['admissible'] for v in outputs)))
            joint, _ = interpret(jax.make_jaxpr(stream.value_and_grad)(params, points), boxes, constant_bounds=span.constant_bounds())
            self.assertTrue(joint[0].summary(1e-09, 2e-10)['admissible'])
            self.assertTrue(all((v.summary(1e-08, 2e-09)['admissible'] for v in joint[1:])))

    def test_exact_span_values_and_all_parameter_gradients(self):
        model, _, params, points = case('ridge')
        dx, dy = (OperatorSpec.mixed((1, 0)), OperatorSpec.mixed((0, 1)))
        combined = OperatorSpec.polynomial({(1, 0): 2, (0, 1): -3})
        ops = (dx, dy, combined, dx, OperatorSpec.laplacian(0))
        reduced, span = build_reduced(model, ops)
        self.assertEqual(len(span.basis), 3)
        self.assertEqual(span.reconstruction[2], (Fraction(2), Fraction(-3), Fraction(0)))
        ordinary = build(model, ops)
        objective = objectives()[0]
        for f in (ordinary, reduced):
            value = jax.value_and_grad(lambda p: objective.full_objective(p, points, f(p, points)))(params)
            if f is ordinary:
                expected = value
            else:
                assert_tree(self, value, expected)
        assert_tree(self, reduced(params, points), ordinary(params, points))

    def test_transpose_and_nonbinary_reconstruction_bounds(self):
        triple = OperatorSpec.polynomial({(1, 0): 3})
        single = OperatorSpec.mixed((1, 0))
        span = operator_span((triple, single), 2)
        self.assertEqual(span.reconstruction[1][0], Fraction(1, 3))
        bound = next(iter(span.constant_bounds().values()))
        self.assertGreater(bound.error[1, 0], 0.0)
        source = (jnp.asarray([0.7, -0.4]),)
        cotangent = (jnp.asarray([0.2, 0.3]), jnp.asarray([-0.2, 0.5]))
        lhs = sum((jnp.vdot(a, b) for a, b in zip(span.reconstruct(source), cotangent)))
        rhs = sum((jnp.vdot(a, b) for a, b in zip(source, span.transpose(cotangent))))
        np.testing.assert_allclose(lhs, rhs, atol=1e-15)

    def test_compact_laplacian_and_budget_refusals(self):
        lap = OperatorSpec.laplacian(4)
        span = operator_span((lap, lap), 1024, limits=SpanLimits(max_terms=4))
        self.assertEqual(len(span.basis), 1)
        self.assertEqual(span.report['compact_atoms'], 2)
        with self.assertRaises(ValueError):
            operator_span((OperatorSpec.mixed((1, 0)),), 2, limits=SpanLimits(max_work=1))
        with self.assertRaises(NotImplementedError):
            operator_span((OperatorSpec.directional(1),), 2)
if __name__ == '__main__':
    if not jax.config.x64_enabled:
        raise SystemExit('JAX_ENABLE_X64=1 is required')
    unittest.main(verbosity=2)
