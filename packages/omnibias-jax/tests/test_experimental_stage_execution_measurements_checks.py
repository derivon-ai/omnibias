# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private optimization candidates: correctness does not imply promotion."""
import pytest
jax = pytest.importorskip("jax")
pytest.importorskip("mpmath")
jax.config.update("jax_enable_x64", True)
import threading
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from omnibias.jax._experimental_stage.device_training import Boxes, DOMAIN_EXIT, initial_state, make_adam_step, make_training_block, recenter_boxes, resume_after_qualification
from omnibias.jax._experimental_stage.execution_measurements import indexed_points, input_stream
from omnibias.jax._experimental_stage.prefetch import DoubleBuffer, FrozenDependencies, SynchronousBuffer

class InputChecks(unittest.TestCase):

    def test_equal_work_and_exact_indexing(self):
        main_thread = threading.get_ident()
        results = []
        for prefetched in (False, True):
            with input_stream((3, 2), 'fresh', 831004, 5, (-0.2, 0.2), prefetched=prefetched) as stream:
                records = list(stream)
                self.assertEqual(stream.report['maximum_future_batches'], int(prefetched))
            for index, record in enumerate(records):
                self.assertEqual(record.index, index)
                expected = np.random.default_rng(831004 + index).uniform(-0.2, 0.2, (3, 2))
                np.testing.assert_array_equal(record.payload, expected)
                self.assertEqual(record.payload.dtype, np.dtype('float64'))
                self.assertLessEqual(record.host_start_ns, record.host_end_ns)
                self.assertLessEqual(record.host_end_ns, record.transfer_start_ns)
                self.assertLessEqual(record.transfer_start_ns, record.transfer_end_ns)
                self.assertEqual(record.host_thread_id == main_thread, not prefetched)
            results.append(records)
        for sync, async_record in zip(*results):
            self.assertEqual(sync.input_token, async_record.input_token)
            self.assertEqual(sync.dependency_token, async_record.dependency_token)
            np.testing.assert_array_equal(sync.payload, async_record.payload)

    def test_fixed_resident_not_applicable(self):
        resident = np.full((2, 3), 0.1, dtype=np.float64)
        with input_stream(resident.shape, 'fixed', 4, 3, prefetched=True, resident=resident) as stream:
            resident[:] = -0.1
            first = next(stream)
            second, third = tuple(stream)
            self.assertFalse(stream.report['prefetch_applicable'])
            self.assertFalse(stream.report['prefetch_active'])
            self.assertIs(first.payload, second.payload)
            self.assertIs(first.payload, third.payload)
            self.assertEqual(second.host_preparation_s, 0.0)
            np.testing.assert_array_equal(second.payload, np.full((2, 3), 0.1))

    def test_bounds_validation_empty_and_budget(self):
        data = tuple(indexed_points((2, 3), 4, 2, (np.array([-0.2, 0.1, 0.0]), np.array([0.2, 0.1, 0.3]))))
        for batch in data:
            np.testing.assert_array_equal(batch[:, 1], [0.1, 0.1])
        for shape in ((3,), (2, 0), (True, 3), (1, 2, 3)):
            with self.assertRaises(ValueError):
                input_stream(shape, 'fresh', 4)
        for bounds in ((1.0, 0.0), (np.nan, 2.0), (0.0, np.inf)):
            with self.assertRaises(ValueError):
                input_stream((2, 3), 'fresh', 4, bounds=bounds)
        with input_stream((2, 3), 'fresh', 4, count=0, prefetched=True) as stream:
            self.assertEqual(list(stream), [])
        with self.assertRaises((MemoryError, ValueError)):
            with input_stream((2, 3), 'fresh', 4, max_host_bytes=32) as stream:
                next(stream)

    def test_producer_failure_closes_both_modes(self):
        for cls in (SynchronousBuffer, DoubleBuffer):

            def batches():
                yield np.ones((2, 3))
                raise RuntimeError('saved generation failure')
            stream = cls(batches(), lambda batch, frozen: batch, FrozenDependencies({}))
            next(stream)
            with self.assertRaisesRegex(RuntimeError, 'saved generation failure'):
                next(stream)
            self.assertTrue(stream._closed)

    def test_prefetched_suffix_retained_across_domain_exit(self):
        params = jnp.asarray([0.2])
        state = initial_state(params)
        boxes = Boxes(params - 1e-06, params + 1e-06, jnp.asarray(-0.2), jnp.asarray(0.2))
        objective = lambda p, x: jnp.sum((p[0] + x) ** 2)
        step = make_adam_step(jax.value_and_grad(objective))
        with input_stream((2, 1), 'fresh', 831004, 4, prefetched=True) as stream:
            pending = {i: next(stream) for i in range(3)}
            block = jax.jit(make_training_block(step, 3))
            result = block(state, jnp.stack([pending[i].payload for i in range(3)]), boxes)
            self.assertEqual(int(result.status), DOMAIN_EXIT)
            np.testing.assert_array_equal(result.evaluated, [True, False, False])
            newboxes = recenter_boxes(boxes, result.proposed.params, result.delta)
            resumed = resume_after_qualification(result, newboxes, admissible=True)
            del pending[0]
            pending[3] = next(stream)
            tail = block(resumed, jnp.stack([pending[i].payload for i in (1, 2, 3)]), newboxes)
            self.assertEqual(int(tail.accepted.accepted_steps), 4)
            direct = state
            wide = Boxes(jnp.asarray([-0.5]), jnp.asarray([0.5]), boxes.input_lower, boxes.input_upper)
            for x in indexed_points((2, 1), 831004, 4):
                direct = jax.jit(step)(direct, jnp.asarray(x), wide).accepted
            for actual, expected in zip(jax.tree.leaves(tail.accepted), jax.tree.leaves(direct)):
                np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=1e-15)
if __name__ == '__main__':
    unittest.main(verbosity=2)
