# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Line-domain CCF discovery -> CAP -> symbolic replay integration."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from omnibias.core.proof import Conjecture  # noqa: E402
from omnibias.pinn.certified import build_default_machine  # noqa: E402
from omnibias.pinn.jax.discovery import cap, ccf_line  # noqa: E402


def test_line_discovery_cap_symbolic_replay() -> None:
    pytest.importorskip("omnibias.symbolic")
    from omnibias.symbolic import recover_ccf_scaling_law, verify_cap_bundle

    cfg = ccf_line.CCFLineDiscoveryConfig(
        n_terms=4, n_grid=40, y_max=10.0, seed=0, optimizer="adam", lam_init=0.6
    )
    result = ccf_line.run_ccf_line_discovery(cfg, steps=25, lr=5e-3)
    bundle = cap.build_cap_bundle(result, reproduces_published_lambda=None)
    assert cap.cap_schema_errors(bundle) == []
    assert bundle["domain"]["type"] == "line_compactified"
    assert bundle["honesty"]["navier_stokes_proof_claim"] is False
    assert bundle.get("hilbert_convention") == "hardy_exact" or result.extra.get(
        "hilbert_convention"
    ) == "hardy_exact"
    report = verify_cap_bundle(bundle, atol=1e-6)
    assert report["residual_samples_match"] is True
    # scaling-law recovery is best-effort on a coarse smoke profile
    law = recover_ccf_scaling_law(result.y, result.theta, result.theta_y)
    assert "lambda_recovered" in law

    machine = build_default_machine()
    verdict = machine.evaluate(
        Conjecture(
            "CCF line CAP",
            "ccf_line_compactified_cap",
            {"certificate": bundle},
        )
    )
    assert verdict.status == "PROVED"
    assert verdict.honesty_ok
