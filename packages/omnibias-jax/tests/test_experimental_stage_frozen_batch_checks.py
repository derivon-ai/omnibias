# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private optimization candidates: correctness does not imply promotion."""
import pytest
jax = pytest.importorskip("jax")
pytest.importorskip("mpmath")
jax.config.update("jax_enable_x64", True)
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import omnibias.jax._experimental_stage.frozen_batch as projection
from omnibias.jax._experimental_stage.bounds import Bound, ConstantBindings
from omnibias.jax._experimental_stage.certificates import certify_low_rank, certify_low_rank_operator, low_rank_constants
from omnibias.jax._experimental_stage.low_rank_operators import build_low_rank_ridge
from omnibias.core.operator_spec import OperatorSpec
from omnibias.jax._experimental_stage.prefetch import FrozenDependencies
from omnibias.jax._experimental_stage.reuse import low_rank_objective

class FrozenBatchChecks(unittest.TestCase):

    def setUp(self):
        self.base = np.array([[0.3, -0.2], [0.1, 0.4]], dtype=np.float64)
        self.x = np.array([[0.2, -0.1], [-0.3, 0.05], [0.1, 0.3]], dtype=np.float64)
        self.frozen = FrozenDependencies({'base': self.base})
        self.factors = (jnp.asarray([[0.1], [-0.05]]), jnp.asarray([[0.04], [-0.07]]))

    def test_same_work_tokens_metric_once_and_immutable_base(self):
        metric_calls = []
        original = projection._fixed_metric

        def measured(base):
            metric_calls.append(base.shape)
            return original(base)
        with patch.object(projection, '_fixed_metric', measured):
            owner = projection.prepare_frozen_base(self.frozen)
            self.base[:] = 100
            records = []
            for asynchronous in (False, True):
                with owner.stream((self.x, self.x * 0.5), batch_shape=self.x.shape, prefetched=asynchronous) as stream:
                    records.append(list(stream))
            self.assertEqual(metric_calls, [(2, 2)])
        for a, b in zip(*records):
            self.assertEqual(a.input_token, b.input_token)
            self.assertEqual(a.dependency_token, b.dependency_token)
            self.assertEqual(a.adapter.source_digest, b.adapter.source_digest)
            self.assertIs(a.adapter.metric, owner.metric)
            self.assertIs(a.adapter.base, owner.base)
            np.testing.assert_array_equal(a.adapter.projection, b.adapter.projection)
            expected = np.asarray(a.payload) @ np.array([[0.3, 0.1], [-0.2, 0.4]])
            np.testing.assert_allclose(a.adapter.projection, expected, atol=1e-16, rtol=1e-15)
            self.assertIs(a.require_dependencies(self.frozen), a.adapter)
            with self.assertRaisesRegex(ValueError, 'different immutable base'):
                a.require_dependencies(FrozenDependencies({'base': self.base}))

    def test_complete_live_gradients_independent_mp_and_saved_bounds(self):
        owner = projection.prepare_frozen_base(self.frozen)
        with owner.stream((self.x,), batch_shape=self.x.shape, prefetched=True) as stream:
            batch = next(stream)
        fn = lambda factors: low_rank_objective(batch.adapter.geometry, factors)
        actual_value, actual_gradient = jax.jit(jax.value_and_grad(fn))(self.factors)
        with mp.workdps(70):
            base = [[mp.mpf(float(v)) for v in row] for row in np.asarray(batch.adapter.base)]
            points = [[mp.mpf(float(v)) for v in row] for row in self.x]
            flat = [mp.mpf(float(v)) for factor in self.factors for v in np.asarray(factor).flat]

            def loss(values):
                u, v = (values[:2], values[2:])
                a = [[base[h][i] + u[h] * v[i] for i in range(2)] for h in range(2)]
                z = [[sum((row[i] * a[h][i] for i in range(2))) for h in range(2)] for row in points]
                g = [[sum((a[h][i] * a[k][i] for i in range(2))) for k in range(2)] for h in range(2)]
                return sum((mp.tanh(value) ** 2 for row in z for value in row)) / 6 + mp.mpf(float(0.001)) * sum((value ** 2 for row in g for value in row))
            expected = float(loss(flat))
            gradients = []
            for index, v in enumerate(flat):

                def varied(q):
                    pp = flat.copy()
                    pp[index] = q
                    return loss(pp)
                gradients.append(float(mp.diff(varied, v)))
        np.testing.assert_allclose(actual_value, expected, atol=1e-09, rtol=2e-10)
        np.testing.assert_allclose(np.concatenate([np.asarray(v).ravel() for v in actual_gradient]), gradients, atol=1e-08, rtol=2e-09)
        self.assertGreater(max((abs(v) for v in gradients)), 100 * 1e-08)
        bindings = batch.constant_bounds()
        self.assertGreaterEqual(len(bindings.legacy), 3)
        report = certify_low_rank(batch.adapter, self.factors, radius=1e-06)
        self.assertTrue(report['admissible'], report)

    def test_operator_gradients_for_changed_and_zero_live_factors(self):
        owner = projection.prepare_frozen_base(self.frozen)
        with owner.stream((self.x,), batch_shape=self.x.shape) as stream:
            adapter = next(stream).adapter
        ops = (OperatorSpec.laplacian(0), OperatorSpec.laplacian(1))
        function = build_low_rank_ridge(adapter, ops)
        params = (*self.factors, jnp.asarray([0.03, -0.02]), jnp.asarray([0.4, -0.3]), jnp.asarray(0.02))

        def reference(p):
            u, v, b, c, offset = p
            a = adapter.base + u @ v.T
            t = jnp.tanh(adapter.points @ a.T + b)
            return (t @ c + offset, -2 * t * (1 - t * t) @ (c * jnp.sum(a * a, axis=1)))
        scalar = lambda outputs: sum((jnp.mean(y * y) for y in outputs))
        actual = jax.jit(jax.value_and_grad(lambda p: scalar(function(p))))
        expected = jax.jit(jax.value_and_grad(lambda p: scalar(reference(p))))
        for p in (params, tuple((v + 1e-06 for v in params)), (jnp.zeros_like(params[0]), *params[1:]), (params[0], jnp.zeros_like(params[1]), *params[2:])):
            a, b = (actual(p), expected(p))
            for aa, bb in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                np.testing.assert_allclose(aa, bb, atol=1e-08, rtol=2e-09)
        report = certify_low_rank_operator(adapter, ops, params, radii=1e-06, scalar_objective=lambda p, x, outputs: scalar(outputs))
        self.assertTrue(report['admissible'], report)

    def test_shape_dtype_frozen_scope_memory_and_overflow_refusal(self):
        with self.assertRaises(ValueError):
            projection.prepare_frozen_base(FrozenDependencies({'base': self.base, 'U': np.ones((2, 1))}))
        with self.assertRaises(ValueError):
            projection.prepare_frozen_base(FrozenDependencies({'base': self.base.astype(np.float32)}))
        with self.assertRaises(MemoryError):
            projection.prepare_frozen_base(self.frozen, max_host_bytes=32)
        owner = projection.prepare_frozen_base(self.frozen, max_host_bytes=1000)
        with self.assertRaises(MemoryError):
            owner.stream((np.zeros((100, 2)),), batch_shape=(100, 2))
        for data in (np.ones((2, 2)), self.x.astype(np.float32), self.x * np.nan):
            with self.assertRaises(ValueError):
                with owner.stream((data,), batch_shape=self.x.shape, prefetched=True) as stream:
                    next(stream)
        large = projection.prepare_frozen_base(FrozenDependencies({'base': np.array([[4.0]], dtype=np.float64)}))
        with self.assertRaises(ArithmeticError):
            with large.stream((np.array([[1e+308]]),), batch_shape=(1, 1), prefetched=True) as stream:
                next(stream)

    def test_enclosures_are_produced_ahead_and_binding_does_not_read_device(self):
        from omnibias.jax._experimental_stage.qualification_session import _freeze_bindings
        calls = []
        original_product = projection.matrix_product

        def observed_product(a, b):
            calls.append((a.shape, b.shape))
            return original_product(a, b)
        with patch.object(projection, 'matrix_product', observed_product):
            owner = projection.prepare_frozen_base(self.frozen)
            self.assertEqual(calls, [((2, 2), (2, 2))])
            with owner.stream((self.x, self.x * 0.5), batch_shape=self.x.shape, prefetched=True) as stream:
                batches = list(stream)
            self.assertEqual(len(calls), 3)
            references = [low_rank_constants(batch.adapter) for batch in batches]
            asarray, contiguous = (np.asarray, np.ascontiguousarray)

            def host_only(original):

                def checked(value, *args, **kwargs):
                    if isinstance(value, jax.Array):
                        raise AssertionError('device readback during prefetched bound assembly')
                    return original(value, *args, **kwargs)
                return checked
            with patch.object(np, 'asarray', host_only(asarray)), patch.object(np, 'ascontiguousarray', host_only(contiguous)):
                for batch, reference in zip(batches, references):
                    bindings = batch.constant_bounds()
                    copied = ConstantBindings().extend(bindings)
                    frozen = _freeze_bindings(bindings, None)
                    self.assertEqual(set(copied.legacy), set(reference.legacy))
                    self.assertEqual(set(frozen.legacy), set(reference.legacy))
                    for key in copied.legacy:
                        for name in ('lo', 'hi', 'error'):
                            np.testing.assert_array_equal(getattr(copied.legacy[key], name), getattr(reference.legacy[key], name))
                    for value in (batch.adapter.projection, batch.adapter.metric):
                        self.assertIsNotNone(copied.resolve(value, (), 0))
                        snapshot = copied.owned_host_snapshot(value)
                        with self.assertRaises(ValueError):
                            snapshot.setflags(write=True)
            self.assertEqual(len(calls), 3, 'consumer must not reconstruct any dot enclosure')
        self.assertEqual(batches[0].preparation.host_metadata.nbytes, 4 * self.x.shape[0] * 2 * 8)
        self.assertEqual(owner.fixed_metadata.nbytes, 4 * 2 * 2 * 8)
        self.assertNotEqual(batches[0].input_token, batches[1].input_token)
        self.assertNotEqual(batches[0].preparation.host_metadata.token, batches[1].preparation.host_metadata.token)

    def test_owned_transfer_binding_refusals_and_ordinary_collision_semantics(self):
        host = FrozenDependencies({'value': self.x}).values['value']
        value = jax.device_put(host)
        bound = Bound.point(host)
        bindings = ConstantBindings().bind_owned_transfer(value, host, bound)
        self.assertIs(bindings.owned_host_snapshot(value), host)
        for readonly in (False, True):
            mutable = np.array(host)
            if readonly:
                mutable.setflags(write=False)
            with self.assertRaises(ValueError):
                ConstantBindings().bind_owned_transfer(value, mutable, bound)
        mismatched = FrozenDependencies({'v': host.astype(np.float32)}).values['v']
        with self.assertRaises(ValueError):
            ConstantBindings().bind_owned_transfer(value, mismatched, bound)
        with self.assertRaises(ValueError):
            ConstantBindings().bind_owned_transfer(value, host.reshape(-1), Bound.point(host.reshape(-1)))
        changed = FrozenDependencies({'v': host + 1}).values['v']
        with self.assertRaises(ValueError):
            bindings.bind_owned_transfer(value, changed, Bound.point(changed))
        other = Bound(host, host, np.full(host.shape, 1e-12))
        bindings.add_legacy(host, bound).add_legacy(host, other)
        self.assertGreaterEqual(next(iter(bindings.legacy.values())).error.max(), 1e-12)
        ordinary = np.array(host)
        legacy = ConstantBindings().bind_object(ordinary, bound)
        ordinary[0, 0] += 0.1
        with self.assertRaises(ValueError):
            legacy.resolve(ordinary, (), 0)
        value.delete()
        with self.assertRaises(ValueError):
            bindings.resolve(value, (), 0)
        with self.assertRaises(ValueError):
            ConstantBindings().bind_owned_transfer(value, host, bound)
if __name__ == '__main__':
    unittest.main(verbosity=2)
