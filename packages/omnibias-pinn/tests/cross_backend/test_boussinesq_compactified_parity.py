# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bit-parity: jax vs torch Boussinesq (q, β) envelopes."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.jax.equations import boussinesq_compactified as jbc  # noqa: E402
from omnibias.pinn.torch.equations import boussinesq_compactified as tbc  # noqa: E402


def test_compactify_and_envelope_parity() -> None:
    y1 = np.array([0.0, 0.4, -0.4, 1.2], dtype=np.float64)
    y2 = np.array([0.0, 0.2, 0.2, -0.5], dtype=np.float64)
    lam = 1.7
    qj, bj, r2j = jbc.compactify_yb_lambda(jnp.asarray(y1), jnp.asarray(y2), lam)
    qt, bt, r2t = tbc.compactify_yb_lambda(torch.tensor(y1), torch.tensor(y2), lam)
    np.testing.assert_allclose(np.asarray(qj), qt.numpy(), atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(np.asarray(bj), bt.numpy(), atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(np.asarray(r2j), r2t.numpy(), atol=1e-15, rtol=0.0)
    hj = jbc.affine_hat_jet(qj, bj, 0.4, 0.15, -0.08)
    ht = tbc.affine_hat_jet(qt, bt, 0.4, 0.15, -0.08)
    fj = jbc.compose_boussinesq_envelope_fields(
        jnp.asarray(y1),
        jnp.asarray(y2),
        lam,
        hat_omega=hj,
        hat_theta=hj,
        hat_psi=hj,
        theta_decay_power=1.0,
        psi_decay_power=0.0,
    )
    ft = tbc.compose_boussinesq_envelope_fields(
        torch.tensor(y1),
        torch.tensor(y2),
        lam,
        hat_omega=ht,
        hat_theta=ht,
        hat_psi=ht,
        theta_decay_power=1.0,
        psi_decay_power=0.0,
    )
    for key in ("omega", "omega_y1", "theta", "psi", "psi_lap"):
        np.testing.assert_allclose(
            np.asarray(fj[key]), ft[key].numpy(), atol=1e-13, rtol=0.0
        )


def test_chart_first_derivatives_match_fd() -> None:
    y1 = jnp.array([0.35, -0.2], dtype=jnp.float64)
    y2 = jnp.array([0.15, 0.4], dtype=jnp.float64)
    lam = 1.4
    jets = jbc.chart_jets(y1, y2, lam)
    eps = 1e-6
    qp, _, _ = jbc.compactify_yb_lambda(y1 + eps, y2, lam)
    qm, _, _ = jbc.compactify_yb_lambda(y1 - eps, y2, lam)
    fd = (qp - qm) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(jets["q_y1"]), np.asarray(fd), atol=1e-8)
