# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Signed-hat homotopy: C^∞ envelope, no stretch/Rung-1 forge."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.pinn.jax.discovery.ccf_hat_homotopy import (  # noqa: E402
    GAUGE,
    GAUGE_PT,
    h0_target,
    hard_gauge,
    hat_from_cheb,
    honesty_flags,
    ker_hat_init,
    node_hat_init,
    omega_from_hat,
)

LAM = 0.6057


def test_constant_hat_is_wang_envelope_after_gauge() -> None:
    y = jnp.linspace(-8.0, 8.0, 201, dtype=jnp.float64)
    coeffs = jnp.asarray([1.0])
    om, omy = omega_from_hat(y, coeffs, lam=LAM)
    om, omy = hard_gauge(y, om, omy, point=GAUGE_PT, value=GAUGE)
    assert float(jnp.interp(GAUGE_PT, y, om)) == pytest.approx(GAUGE, abs=1e-12)
    assert float(jnp.max(jnp.abs(om + om[::-1]))) < 1e-12
    assert float(jnp.max(jnp.abs(omy - omy[::-1]))) < 1e-12
    hat, _ = hat_from_cheb(y, coeffs, lam=LAM)
    assert float(jnp.max(jnp.abs(hat - hat[::-1]))) < 1e-12


def test_node_hat_is_negative_on_small_positive_y() -> None:
    y = jnp.linspace(-8.0, 8.0, 201, dtype=jnp.float64)
    coeffs = node_hat_init(y, node=0.35, lam=LAM, degree=6)
    om, omy = omega_from_hat(y, coeffs, lam=LAM)
    om, _ = hard_gauge(y, om, omy, point=GAUGE_PT, value=GAUGE)
    y_core = 0.15
    assert float(jnp.interp(y_core, y, om)) < 0.0
    assert float(jnp.interp(GAUGE_PT, y, om)) == pytest.approx(GAUGE, abs=1e-12)


def test_hard_gauge_is_scale_invariant() -> None:
    y = jnp.linspace(-8.0, 8.0, 81, dtype=jnp.float64)
    coeffs = ker_hat_init(y, eps=0.4, lam=LAM, degree=3)
    om_a, omy_a = omega_from_hat(y, coeffs, lam=LAM)
    om_b, omy_b = omega_from_hat(y, 1.4 * coeffs, lam=LAM)
    ga, _gya = hard_gauge(y, om_a, omy_a, point=GAUGE_PT, value=GAUGE)
    gb, _gyb = hard_gauge(y, om_b, omy_b, point=GAUGE_PT, value=GAUGE)
    assert float(jnp.interp(GAUGE_PT, y, ga)) == pytest.approx(GAUGE, abs=1e-12)
    assert float(jnp.interp(GAUGE_PT, y, gb)) == pytest.approx(GAUGE, abs=1e-12)
    assert float(jnp.max(jnp.abs(ga - gb))) < 1e-10


def test_train_nodes_clusters_origin_and_contains_gauge() -> None:
    from omnibias.pinn.jax.discovery.ccf_hat_homotopy import train_nodes

    y = train_nodes(97, 40.0, core=2.0)
    assert float(y[0]) < 0.0 and float(y[-1]) > 0.0
    assert bool(jnp.all(y[1:] > y[:-1]))
    assert float(jnp.min(jnp.abs(y))) == 0.0
    assert float(jnp.min(jnp.abs(y - 0.5))) < 0.05


def test_honesty_flags_never_start_rung1() -> None:
    locked = honesty_flags(dense_max_abs=1e-14, anti_ghost=True)
    assert locked["stretch_1e-13_cleared"] is True
    assert locked["rung1_1e-11_report"] is False
    assert locked["navier_stokes_proof_claim"] is False
    open_floor = honesty_flags(dense_max_abs=0.08, anti_ghost=True)
    assert open_floor["stretch_1e-13_cleared"] is False
    assert h0_target(LAM) == pytest.approx(0.5 * (2.0 + LAM))


def test_hat_homotopy_does_not_move_gates() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
    from _gates import CCF_RESIDUAL_GATE_1ST_UNSTABLE, CCF_STRETCH_RESIDUAL_GATE

    assert CCF_STRETCH_RESIDUAL_GATE == 1e-13
    assert CCF_RESIDUAL_GATE_1ST_UNSTABLE == 1e-11
