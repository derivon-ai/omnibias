# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the NLS / Gross-Pitaevskii residual (torch backend)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.qpinn.torch.equations import NLS, NLSOutput, nls


class TestNLSShape:
    def test_returns_named_tuple(self, psi_field_xt, coords_xt):
        state = psi_field_xt(coords_xt)
        out = NLS(g=1.0)(state)
        assert isinstance(out, NLSOutput)
        assert out.residual.shape == (coords_xt.shape[0], 2)
        assert "mean_density" in out.diag
        assert "nonlinear_energy" in out.diag

    def test_function_form_matches_class(self, psi_field_xt, coords_xt):
        state = psi_field_xt(coords_xt)
        out_cls = NLS(g=2.0, hbar=1.0, mass=1.0)(state)
        out_fn = nls(state, g=2.0, hbar=1.0, mass=1.0)
        torch.testing.assert_close(out_cls.residual, out_fn.residual)

    def test_zero_g_reduces_to_tdse(self, psi_field_xt, coords_xt):
        """g=0 NLS should match TDSE residual exactly."""
        from omnibias.qpinn.torch.equations import TDSE
        state = psi_field_xt(coords_xt)
        nls_out = NLS(g=0.0)(state)
        tdse_out = TDSE()(state)
        torch.testing.assert_close(nls_out.residual, tdse_out.residual)


class TestNLSPhysics:
    def test_repulsive_vs_attractive_differ(self, psi_field_xt, coords_xt):
        state = psi_field_xt(coords_xt)
        out_pos = NLS(g=1.0)(state)
        out_neg = NLS(g=-1.0)(state)
        assert not torch.allclose(out_pos.residual, out_neg.residual)

    def test_requires_time_axis(self, psi_field_x, coords_x):
        state = psi_field_x(coords_x)
        with pytest.raises(ValueError, match="time axis"):
            NLS()(state)
