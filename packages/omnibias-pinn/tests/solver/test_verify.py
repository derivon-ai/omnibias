# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Optional certified mode: sealed a-posteriori error certificate + honesty.

Skips gracefully if either the torch backend or the verified stack is absent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omnibias.core.verified.pde_certificate")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
from omnibias.pinn.solver import verify  # noqa: E402


def _solved_poisson(source_const: float = 1.0):
    torch.set_default_dtype(torch.float64)
    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    system = pde.poisson(dom, source=source_const, boundary=0.0)
    sol = pt.solve_least_squares(
        system,
        hidden=24,
        weight_init_scale=2.0,
        seed=0,
        collocation=pde.CollocationSpec(n_interior=12, n_boundary=12),
    )
    return sol


def test_extract_layers_shapes() -> None:
    sol = _solved_poisson()
    layers = verify.extract_layers(sol.field)
    assert len(layers) == 2
    (w, beta, act), (c, b, none_act) = layers
    assert act == "tanh"
    assert none_act is None
    assert len(w) == 24 and len(w[0]) == 2  # hidden (H, D)
    assert len(beta) == 24
    assert len(c) == 1 and len(c[0]) == 24  # readout (C, H)
    assert len(b) == 1


def test_certify_poisson_seals_honest_certificate() -> None:
    from omnibias.core.proof.certificate import verify_certificate_digest
    from omnibias.core.verified.pde_certificate import (
        pinn_aposteriori_schema_errors,
        replay_pinn_aposteriori_certificate,
    )

    sol = _solved_poisson(source_const=1.0)
    result = verify.certify_poisson(
        sol,
        source=1.0,
        boundary=0.0,
        stability_interior=0.1,
        stability_boundary=1.0,
        interior_splits=2,
        boundary_splits=2,
    )
    cert = result.certificate

    # rigorous residuals are finite, non-negative sup bounds
    assert result.interior_residual >= 0.0
    assert result.boundary_residual >= 0.0
    assert result.error_bound >= 0.0

    # sealed + digest-verifiable + schema-valid + arithmetic replays
    assert verify_certificate_digest(cert)
    assert pinn_aposteriori_schema_errors(cert) == []
    assert replay_pinn_aposteriori_certificate(cert)

    # honesty: never a global-regularity / continuum claim; the enclosure IS interval-verified
    assert cert["honesty"]["unproven_claim"] is False
    assert cert["honesty"]["continuum_claim"] is False
    assert cert["honesty"]["interval_verified"] is True
    assert "theorem_prover_verified" not in cert["honesty"]

    # provenance recorded
    assert cert["payload"]["model"]["ansatz"] == "one_layer"
    assert cert["payload"]["model"]["backend"] == "torch"


def test_certify_linear_bvp_matches_convenience_wrapper() -> None:
    from omnibias.core.verified.pde_certificate import poisson as vpoisson

    sol = _solved_poisson(source_const=0.5)
    pde_op = vpoisson(sol.system.domain.ndim, 0.5)
    generic = verify.certify_linear_bvp(sol, pde_op, boundary=0.0)
    conv = verify.certify_poisson(sol, source=0.5, boundary=0.0)
    assert generic.error_bound == pytest.approx(conv.error_bound, rel=1e-12)


def test_certified_mode_rejects_time_dependent_domain() -> None:
    torch.set_default_dtype(torch.float64)
    dom = pde.Domain(("x", "t"), ((0.0, 1.0), (0.0, 0.5)))
    heat = pde.heat(dom, diffusivity=0.1, initial=0.0, boundary=0.0)
    field = pt.build_field(heat, hidden=8, seed=0)
    with pytest.raises(ValueError, match="steady"):
        verify.spatial_box(heat.domain)
    with pytest.raises(ValueError):
        verify.boundary_faces(heat.domain, 0.0)
    # extract_layers still works on any one-layer field (geometry is the guard)
    assert len(verify.extract_layers(field)) == 2
