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
from omnibias.jax._experimental_stage.bounds import Bound
from omnibias.jax._experimental_stage.planner import Domain, Limits, Tolerances, refresh_environment
from omnibias.jax._experimental_stage.qualification_session import QualificationSession

class SessionChecks(unittest.TestCase):

    def setUp(self):
        self.p = (jnp.asarray([0.2]),)
        self.x = jnp.asarray([[0.1], [0.3]])
        self.scale = np.asarray([1.0])
        self.forward = lambda p, x: (p[0][0] * x[:, 0],)
        self.objective = lambda p, x, y: self.scale[0] * jnp.sum(y[0] ** 2)
        self.vg = jax.value_and_grad(lambda p, x: self.objective(p, x, self.forward(p, x)))

    def domain(self, center=0.2, radius=0.01):
        return Domain((Bound([center - radius], [center + radius], 0.0),), Bound([[-0.4]], [[0.4]], 0.0))

    def test_frozen_objective_and_owned_domain(self):
        session = QualificationSession(self.forward, self.vg, self.objective, self.p, self.x)
        domain = self.domain()
        report = session.qualify_reusing(self.p, self.x, domain)
        self.assertTrue(report['admissible'])
        expected = session.value_gradient(self.p, self.x)
        self.scale[0] = 1000000.0
        actual = jax.jit(session.value_gradient)(self.p, self.x)
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a, b)
        domain.parameters[0].lo[0] = -100
        certificate = next(iter(session._certificates.values()))
        self.assertGreater(certificate['domain'].parameters[0].lo[0], 0.1)

    def test_same_contained_and_tolerance_reuse(self):
        session = QualificationSession(self.forward, self.vg, self.objective, self.p, self.x)
        session.qualify_reusing(self.p, self.x, self.domain())
        same = session.qualify_reusing(self.p, self.x, self.domain())
        self.assertEqual(same['session']['lookup'], 'exact')
        contained = session.qualify_reusing(self.p, self.x, self.domain(radius=0.005))
        self.assertEqual(contained['session']['lookup'], 'contained_domain')
        stricter = Tolerances(forward_atol=1e-10)
        session.qualify_reusing(self.p, self.x, self.domain(), tolerances=stricter)
        self.assertEqual(session.info()['misses'], 2)
        self.assertEqual(session.info()['hits'], 2)

    def test_lru_and_byte_cap(self):
        session = QualificationSession(self.forward, self.vg, self.objective, self.p, self.x, max_entries=2)
        for center in (0.2, 0.4, 0.6):
            session.qualify_reusing((jnp.asarray([center]),), self.x, self.domain(center))
        self.assertEqual(session.info()['entries'], 2)
        self.assertEqual(session.info()['evictions'], 1)
        tiny = QualificationSession(self.forward, self.vg, self.objective, self.p, self.x, max_bytes=1)
        result = tiny.qualify_reusing(self.p, self.x, self.domain())
        self.assertTrue(result['admissible'])
        self.assertEqual(result['session']['cache_admission'], 'refused_certificate_byte_budget')
        self.assertEqual(tiny.info()['entries'], 0)

    def test_refresh_revokes_session_even_same_digest(self):
        session = QualificationSession(self.forward, self.vg, self.objective, self.p, self.x)
        session.qualify_reusing(self.p, self.x, self.domain())
        refresh_environment()
        with self.assertRaisesRegex(ValueError, 'environment refreshed'):
            session.qualify_reusing(self.p, self.x, self.domain())
        self.assertEqual(session.info()['entries'], 0)

    def test_changed_resource_limits_do_not_reuse_old_certificate(self):
        session = QualificationSession(self.forward, self.vg, self.objective, self.p, self.x)
        session.qualify_reusing(self.p, self.x, self.domain())
        with self.assertRaises(ArithmeticError):
            session.qualify_reusing(self.p, self.x, self.domain(), limits=Limits(equations=1))
        self.assertEqual(session.info()['misses'], 2)
        self.assertEqual(session.info()['hits'], 0)
if __name__ == '__main__':
    unittest.main(verbosity=2)
