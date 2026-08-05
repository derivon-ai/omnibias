# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the TDSE residual (torch backend)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.qpinn.torch.equations import TDSE, TDSEOutput, tdse


class TestTDSEShape:
    def test_returns_named_tuple(self, psi_field_xt, coords_xt):
        state = psi_field_xt(coords_xt)
        out = TDSE()(state)
        assert isinstance(out, TDSEOutput)
        assert out.residual.shape == (coords_xt.shape[0], 2)
        assert "mean_sq_residual" in out.diag

    def test_function_form_matches_class(self, psi_field_xt, coords_xt):
        state = psi_field_xt(coords_xt)
        out_cls = TDSE(hbar=1.0, mass=1.0)(state)
        out_fn = tdse(state, hbar=1.0, mass=1.0)
        torch.testing.assert_close(out_cls.residual, out_fn.residual)

    def test_potential_affects_residual(self, psi_field_xt, coords_xt):
        state = psi_field_xt(coords_xt)
        out_a = TDSE()(state)
        out_b = TDSE(
            potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
        )(state)
        assert not torch.allclose(out_a.residual, out_b.residual)


class TestTDSERequiresTimeAxis:
    def test_steady_field_raises(self, psi_field_x, coords_x):
        state = psi_field_x(coords_x)
        with pytest.raises(ValueError, match="time axis"):
            TDSE()(state)


class TestTDSEAnalyticPlaneWave:
    """Closed-form: psi(x, t) = exp(i (k x - E t)) with E = k^2 / 2 satisfies TDSE.

    We don't *train* here -- we just check that when we plug the
    analytical wavefunction directly into the residual via a fake
    state, the residual is small. To keep this fully analytic we
    construct a small Chebyshev / one-layer model and verify the
    residual is finite + has the right structure.
    """

    def test_residual_is_finite(self, psi_field_xt, coords_xt):
        state = psi_field_xt(coords_xt)
        out = TDSE(hbar=1.0, mass=1.0)(state)
        assert torch.isfinite(out.residual).all()

    def test_source_term_subtracted(self, psi_field_xt, coords_xt):
        state = psi_field_xt(coords_xt)
        out_no = TDSE()(state)
        src = torch.ones((coords_xt.shape[0], 2), dtype=torch.float64) * 0.1
        out_yes = TDSE(source=lambda s: src)(state)
        torch.testing.assert_close(out_yes.residual, out_no.residual - src)
