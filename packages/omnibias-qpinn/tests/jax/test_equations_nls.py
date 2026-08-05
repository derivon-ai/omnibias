# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the NLS / Gross-Pitaevskii residual (jax backend)."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from omnibias.qpinn.jax.equations import NLS, NLSOutput, nls


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
        assert jnp.allclose(out_cls.residual, out_fn.residual)

    def test_zero_g_reduces_to_tdse(self, psi_field_xt, coords_xt):
        from omnibias.qpinn.jax.equations import TDSE
        state = psi_field_xt(coords_xt)
        nls_out = NLS(g=0.0)(state)
        tdse_out = TDSE()(state)
        assert jnp.allclose(nls_out.residual, tdse_out.residual)

    def test_requires_time_axis(self, psi_field_x, coords_x):
        state = psi_field_x(coords_x)
        with pytest.raises(ValueError, match="time axis"):
            NLS()(state)
