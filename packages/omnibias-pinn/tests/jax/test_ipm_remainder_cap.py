# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-15: IPM CubicGN remainder leftover."""

from __future__ import annotations

import numpy as np
import pytest

from omnibias.pinn.certified.ipm import (
    IPM_REMAINDER_LEFTOVER,
    build_ipm_cap_bundle,
    enclose_ipm_grid_residual,
    ipm_banded_toy_radii,
)
from omnibias.pinn.jax.discovery.ipm import IPMDiscoveryConfig, run_ipm_discovery
from omnibias.pinn.jax.discovery.pipeline import IPMAdapter, run_singularity_pipeline


def test_g1_cubic_gn_forbids_adam() -> None:
    with pytest.raises(ValueError, match="forbids Adam"):
        run_ipm_discovery(IPMDiscoveryConfig(n=6, steps=1, method="adam"))
    with pytest.raises(ValueError, match="forbids Adam"):
        IPMAdapter(n=6, steps=1).discover(seed=0, optimizer="adam")
    out = run_ipm_discovery(IPMDiscoveryConfig(n=6, steps=2, method="cubic"))
    assert out["optimizer"] == "cubic_gn"


def test_g2_enclosure_contains_truth_sample() -> None:
    out = run_ipm_discovery(IPMDiscoveryConfig(n=6, steps=2))
    report = enclose_ipm_grid_residual(out)
    assert report["contains_truth_sample"] is True
    assert np.isfinite(report["lo"]) and np.isfinite(report["hi"])


def test_g3_full_ipm_stays_unproved() -> None:
    out = run_ipm_discovery(IPMDiscoveryConfig(n=6, steps=2))
    bundle = build_ipm_cap_bundle(out)
    assert bundle["full_ipm_proved"] is False
    assert bundle["remainder"]["full_ipm_proved"] is False


def test_g4_leftover_toy_radii_only() -> None:
    leftover = IPM_REMAINDER_LEFTOVER
    assert leftover["leftover_id"] == 53
    assert leftover["leftover_recorded"] is True
    assert leftover["named_closing_object"] == "ipm_banded_toy_radii"
    toy = ipm_banded_toy_radii()
    assert toy["full_ipm_proved"] is False


def test_g5_honesty_and_adapter() -> None:
    pipe = run_singularity_pipeline(IPMAdapter(n=6, steps=2), None)
    assert pipe.discovery["optimizer"] == "cubic_gn"
    assert pipe.certificate["honesty"]["navier_stokes_proof_claim"] is False
    assert pipe.certificate["full_ipm_proved"] is False
    assert pipe.certificate["remainder"]["leftover"]["leftover_id"] == 53
