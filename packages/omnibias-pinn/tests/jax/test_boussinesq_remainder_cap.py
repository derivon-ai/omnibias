# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-16: Boussinesq CubicGN remainder leftover."""

from __future__ import annotations

import numpy as np
import pytest

from omnibias.pinn.certified.boussinesq import (
    BOUSSINESQ_REMAINDER_LEFTOVER,
    build_boussinesq_cap_bundle,
    enclose_boussinesq_grid_residual,
)
from omnibias.pinn.jax.discovery.boussinesq import (
    BoussinesqDiscoveryConfig,
    run_boussinesq_discovery,
)
from omnibias.pinn.jax.discovery.pipeline import (
    BoussinesqAdapter,
    run_singularity_pipeline,
)


def test_g1_cubic_gn_forbids_adam() -> None:
    with pytest.raises(ValueError, match="forbids Adam"):
        run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=1, method="adam"))
    with pytest.raises(ValueError, match="forbids Adam"):
        BoussinesqAdapter(n=6, steps=1).discover(seed=0, optimizer="adam")
    out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
    assert out["optimizer"] == "cubic_gn"


def test_g2_enclosure_contains_truth_sample() -> None:
    out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
    report = enclose_boussinesq_grid_residual(out)
    assert report["contains_truth_sample"] is True
    assert np.isfinite(report["lo"]) and np.isfinite(report["hi"])


def test_g3_full_stays_unproved() -> None:
    out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
    bundle = build_boussinesq_cap_bundle(out)
    assert bundle["full_boussinesq_proved"] is False


def test_g4_leftover_and_lambda_hypothesis() -> None:
    leftover = BOUSSINESQ_REMAINDER_LEFTOVER
    assert leftover["leftover_id"] == 54
    assert leftover["leftover_recorded"] is True
    out = run_boussinesq_discovery(BoussinesqDiscoveryConfig(n=6, steps=2))
    assert out["lambda_n_hypothesis"]["status"] == "empirical_hypothesis_not_theorem"


def test_g5_honesty_and_adapter() -> None:
    pipe = run_singularity_pipeline(BoussinesqAdapter(n=6, steps=2), None)
    assert pipe.discovery["optimizer"] == "cubic_gn"
    assert pipe.certificate["honesty"]["navier_stokes_proof_claim"] is False
    assert pipe.certificate["honesty"]["lambda_n_hypothesis_is_theorem"] is False
    assert pipe.certificate["remainder"]["leftover"]["leftover_id"] == 54
