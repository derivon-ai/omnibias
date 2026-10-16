# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private optimization candidates: correctness does not imply promotion."""
import pytest
jax = pytest.importorskip("jax")
pytest.importorskip("mpmath")
jax.config.update("jax_enable_x64", True)
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import mpmath as mp
from omnibias.jax._experimental_stage.device_training import Boxes, DOMAIN_EXIT, INVALID_INPUT, NONFINITE, initial_state, make_adam_step, make_adam_apply_packet

class PacketOptimizerChecks(unittest.TestCase):

    def setUp(self):
        self.params = (jnp.array([0.2, -0.1]), jnp.asarray(0.03))
        self.points = jnp.array([[0.1], [0.2]])
        self.boxes = Boxes(tuple((v - 0.5 for v in self.params)), tuple((v + 0.5 for v in self.params)), jnp.asarray(-1.0), jnp.asarray(1.0))
        self.vg = jax.value_and_grad(lambda p, x: 0.5 * sum((jnp.sum(v * v) for v in p)))

    def equal(self, a, b):
        self.assertEqual(jax.tree.structure(a), jax.tree.structure(b))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_array_equal(x, y)

    def test_two_packet_updates_agree_with_independent_adam(self):
        reference = jax.jit(make_adam_step(self.vg))
        apply = jax.jit(make_adam_apply_packet())
        state = initial_state(self.params)
        for _ in range(2):
            value, gradient = self.vg(state.params, self.points)
            actual = apply(state, self.points, self.boxes, value, gradient)
            expected = reference(state, self.points, self.boxes)
            with mp.workdps(80):
                count = int(state.accepted_steps) + 1
                b1, b2, lr, epsilon = map(lambda v: mp.mpf(float(v)), (0.9, 0.999, 0.0001, 1e-08))
                ideals = [[], [], []]
                for p, m, v, g in zip(state.params, state.first_moment, state.second_moment, gradient, strict=True):
                    entries = [[], [], []]
                    for pp, mm, vv, gg in zip(np.asarray(p).flat, np.asarray(m).flat, np.asarray(v).flat, np.asarray(g).flat, strict=True):
                        pp, mm, vv, gg = map(lambda x: mp.mpf(float(x)), (pp, mm, vv, gg))
                        new_m = b1 * mm + (1 - b1) * gg
                        new_v = b2 * vv + (1 - b2) * gg * gg
                        new_p = pp - lr * (new_m / (1 - b1 ** count)) / (mp.sqrt(new_v / (1 - b2 ** count)) + epsilon)
                        for out, item in zip(entries, (new_p, new_m, new_v), strict=True):
                            out.append(float(item))
                    for out, item in zip(ideals, entries, strict=True):
                        out.append(np.array(item).reshape(p.shape))
            for result in (actual, expected):
                for computed, ideal in zip((result.accepted.params, result.accepted.first_moment, result.accepted.second_moment), ideals, strict=True):
                    for a, b in zip(computed, ideal, strict=True):
                        np.testing.assert_allclose(a, b, atol=1e-09, rtol=2e-10)
            state = actual.accepted

    def test_rejections_preserve_parameters_and_moments(self):
        apply = jax.jit(make_adam_apply_packet())
        state = initial_state(self.params)
        value, gradient = self.vg(state.params, self.points)
        tight = Boxes(tuple((v - 1e-09 for v in self.params)), tuple((v + 1e-09 for v in self.params)), self.boxes.input_lower, self.boxes.input_upper)
        cases = ((self.points, tight, value, gradient, DOMAIN_EXIT), (self.points * 100, self.boxes, value, gradient, INVALID_INPUT), (self.points, self.boxes, jnp.asarray(jnp.nan), gradient, NONFINITE))
        for points, boxes, value, gradient, status in cases:
            actual = apply(state, points, boxes, value, gradient)
            self.assertEqual(int(actual.status), status)
            self.equal(actual.accepted, state)

    def test_incomplete_gradient_is_refused(self):
        with self.assertRaises(ValueError):
            make_adam_apply_packet()(initial_state(self.params), self.points, self.boxes, jnp.asarray(0.0), (jnp.zeros_like(self.params[0]),))
if __name__ == '__main__':
    unittest.main()
