# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""IPM / Boussinesq streamfunction-formulation smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.certified.boussinesq import build_boussinesq_cap_bundle  # noqa: E402
from omnibias.pinn.certified.ipm import build_ipm_cap_bundle  # noqa: E402
from omnibias.pinn.jax.discovery import boussinesq, ipm  # noqa: E402
from omnibias.pinn.jax.equations.boussinesq_selfsimilar import (  # noqa: E402
    boussinesq_selfsimilar_residual_samples,
    infer_lambda_from_streamfunction_u1_y1,
)
from omnibias.pinn.jax.equations.ipm_selfsimilar import (
    ipm_selfsimilar_residual_samples,  # noqa: E402
)


def test_ipm_streamfunction_residual_finite() -> None:
    y1 = jnp.linspace(-1, 1, 9)
    y2 = jnp.zeros_like(y1)
    th = jnp.exp(-y1 * y1)
    ty1 = -2 * y1 * th
    ty2 = jnp.zeros_like(th)
    psi = 0.1 * y2 * th
    py1 = jnp.zeros_like(th)
    py2 = 0.1 * th
    plap = jnp.zeros_like(th)
    rt, rp = ipm_selfsimilar_residual_samples(
        y1, y2, th, ty1, ty2, psi, py1, py2, plap, 0.5
    )
    assert np.isfinite(np.asarray(rt)).all()
    assert np.isfinite(np.asarray(rp)).all()


def test_ipm_discovery_and_cap() -> None:
    pytest.importorskip("omnibias.symbolic")
    from omnibias.symbolic.ipm import verify_ipm_bundle

    out = ipm.run_ipm_discovery(ipm.IPMDiscoveryConfig(n=8, steps=20))
    bundle = build_ipm_cap_bundle(out)
    assert bundle["honesty"]["navier_stokes_proof_claim"] is False
    assert bundle["honesty"]["formulation"] == "streamfunction_poisson_residual"
    report = verify_ipm_bundle(bundle)
    assert report["residual_samples_match"] is True


def test_boussinesq_discovery_and_cap() -> None:
    pytest.importorskip("omnibias.symbolic")
    from omnibias.symbolic.boussinesq import verify_boussinesq_bundle

    out = boussinesq.run_boussinesq_discovery(boussinesq.BoussinesqDiscoveryConfig(n=8, steps=20))
    bundle = build_boussinesq_cap_bundle(out)
    assert bundle["honesty"]["lambda_n_hypothesis_is_theorem"] is False
    assert "residual_psi" in bundle
    report = verify_boussinesq_bundle(bundle)
    assert report["residual_samples_match"] is True


def test_boussinesq_lambda_inference_relation() -> None:
    assert float(infer_lambda_from_streamfunction_u1_y1(0.0)) == -3.0
    assert float(infer_lambda_from_streamfunction_u1_y1(0.5)) == -4.0


def test_boussinesq_compactified_omega_odd_and_discovery() -> None:
    from omnibias.pinn.jax.equations.boussinesq_compactified import (
        affine_hat_jet,
        compactify_yb_lambda,
        compose_boussinesq_envelope_fields,
    )

    y1 = jnp.array([0.3, -0.3, 0.5], dtype=jnp.float64)
    y2 = jnp.array([0.1, 0.1, -0.2], dtype=jnp.float64)
    lam = 1.5
    q, beta, _r2 = compactify_yb_lambda(y1, y2, lam)
    assert float(q[0]) < 1.0
    hat = affine_hat_jet(q, beta, 0.4, 0.1, -0.05)
    fields = compose_boussinesq_envelope_fields(
        y1, y2, lam, hat_omega=hat, hat_theta=hat, hat_psi=hat
    )
    # Pair (0.3, 0.1) with (-0.3, 0.1): Ω must flip sign.
    assert abs(float(fields["omega"][0] + fields["omega"][1])) < 1e-14
    out = boussinesq.run_boussinesq_discovery(
        boussinesq.BoussinesqDiscoveryConfig(n=6, steps=3, compactified=True)
    )
    assert out["honesty"]["navier_stokes_proof_claim"] is False
    assert out["honesty"]["compactified_envelope"] is True
    assert out["honesty"]["deepmind_residual_claim"] is False
    assert np.isfinite(out["max_abs_residual_omega"])
    assert float(infer_lambda_from_streamfunction_u1_y1(0.0)) == -3.0


def test_boussinesq_residual_shapes() -> None:
    y1 = jnp.linspace(0, 1, 5)
    y2 = jnp.linspace(0, 1, 5)
    z = jnp.zeros_like(y1)
    ro, rt, rp = boussinesq_selfsimilar_residual_samples(
        y1, y2, z, z, z, z, z, z, z, z, z, z, 1.2
    )
    assert ro.shape == y1.shape == rt.shape == rp.shape
