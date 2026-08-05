# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the Hermitian-projection helpers (torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.qpinn.torch.cage import hermitian_projection, hermiticity_loss


class TestHermitianProjection:
    def test_real_matrix(self):
        M = torch.tensor([[1.0, 2.0], [4.0, 3.0]], dtype=torch.float64)
        H = hermitian_projection(M)
        expected = torch.tensor([[1.0, 3.0], [3.0, 3.0]], dtype=torch.float64)
        torch.testing.assert_close(H, expected)

    def test_complex_matrix(self):
        M = torch.tensor(
            [[1.0 + 0j, 2.0 + 1j], [3.0 - 1j, 4.0 + 0j]],
            dtype=torch.complex128,
        )
        H = hermitian_projection(M)
        # H = (M + M^H) / 2; manually:
        # diagonal: real parts unchanged ((1, 4))
        # off-diagonal upper: (2 + 1j + conj(3 - 1j)) / 2 = (2 + 1j + 3 + 1j)/2 = 2.5 + 1j
        # off-diagonal lower: (3 - 1j + conj(2 + 1j))/2 = (3 - 1j + 2 - 1j)/2 = 2.5 - 1j
        expected = torch.tensor(
            [[1.0 + 0j, 2.5 + 1j], [2.5 - 1j, 4.0 + 0j]],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(H, expected)

    def test_idempotent(self):
        torch.manual_seed(0)
        M = torch.randn(4, 4, dtype=torch.float64)
        H = hermitian_projection(M)
        H2 = hermitian_projection(H)
        torch.testing.assert_close(H, H2, atol=1e-14, rtol=1e-14)

    def test_rejects_non_square(self):
        M = torch.zeros((2, 3), dtype=torch.float64)
        with pytest.raises(ValueError, match="last two dims"):
            hermitian_projection(M)

    def test_rejects_low_dim(self):
        M = torch.zeros(5, dtype=torch.float64)
        with pytest.raises(ValueError, match="at least 2 dim"):
            hermitian_projection(M)


class TestHermiticityLoss:
    def test_zero_for_hermitian(self):
        M = torch.tensor([[1.0, 3.0], [3.0, 2.0]], dtype=torch.float64)
        L = hermiticity_loss(M)
        assert float(L) < 1e-14

    def test_positive_for_asymmetric(self):
        M = torch.tensor([[1.0, 2.0], [4.0, 3.0]], dtype=torch.float64)
        L = hermiticity_loss(M)
        assert float(L) > 0

    def test_batch(self):
        """Should work for batched matrices ``(..., N, N)``."""
        torch.manual_seed(0)
        M = torch.randn(3, 4, 4, dtype=torch.float64)
        L = hermiticity_loss(M)
        assert L.shape == ()
        assert float(L) > 0
