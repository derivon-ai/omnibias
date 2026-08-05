# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the Hermitian-projection helpers (jax)."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.qpinn.jax.cage import hermitian_projection, hermiticity_loss


class TestHermitianProjection:
    def test_real_matrix(self):
        M = jnp.array([[1.0, 2.0], [4.0, 3.0]], dtype=jnp.float64)
        H = hermitian_projection(M)
        expected = jnp.array([[1.0, 3.0], [3.0, 3.0]], dtype=jnp.float64)
        assert jnp.allclose(H, expected)

    def test_complex_matrix(self):
        M = jnp.array(
            [[1.0 + 0j, 2.0 + 1j], [3.0 - 1j, 4.0 + 0j]],
            dtype=jnp.complex128,
        )
        H = hermitian_projection(M)
        expected = jnp.array(
            [[1.0 + 0j, 2.5 + 1j], [2.5 - 1j, 4.0 + 0j]],
            dtype=jnp.complex128,
        )
        assert jnp.allclose(H, expected)

    def test_idempotent(self):
        M = jax.random.normal(jax.random.PRNGKey(0), (4, 4), dtype=jnp.float64)
        H = hermitian_projection(M)
        H2 = hermitian_projection(H)
        assert jnp.allclose(H, H2, atol=1e-14)

    def test_rejects_non_square(self):
        M = jnp.zeros((2, 3), dtype=jnp.float64)
        with pytest.raises(ValueError, match="last two dims"):
            hermitian_projection(M)


class TestHermiticityLoss:
    def test_zero_for_hermitian(self):
        M = jnp.array([[1.0, 3.0], [3.0, 2.0]], dtype=jnp.float64)
        L = hermiticity_loss(M)
        assert float(L) < 1e-14

    def test_positive_for_asymmetric(self):
        M = jnp.array([[1.0, 2.0], [4.0, 3.0]], dtype=jnp.float64)
        L = hermiticity_loss(M)
        assert float(L) > 0
