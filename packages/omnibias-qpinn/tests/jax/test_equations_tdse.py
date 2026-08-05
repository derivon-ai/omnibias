# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the TDSE residual (jax backend)."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from omnibias.qpinn.jax.equations import TDSE, TDSEOutput, tdse


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
        assert jnp.allclose(out_cls.residual, out_fn.residual)


class TestTDSERequiresTimeAxis:
    def test_steady_field_raises(self, psi_field_x, coords_x):
        state = psi_field_x(coords_x)
        with pytest.raises(ValueError, match="time axis"):
            TDSE()(state)


class TestTDSESource:
    def test_source_term_subtracted(self, psi_field_xt, coords_xt):
        state = psi_field_xt(coords_xt)
        out_no = TDSE()(state)
        src = jnp.full((coords_xt.shape[0], 2), 0.1, dtype=jnp.float64)
        out_yes = TDSE(source=lambda s: src)(state)
        assert jnp.allclose(out_yes.residual, out_no.residual - src)
