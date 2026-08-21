# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-05: Taylor-model neuron soundness."""

from __future__ import annotations

from omnibias.core.verified.tm_neuron import (
    DISCLAIMER,
    TMNeuronSpec,
    contains_grid_and_sample,
    depth_honesty,
    honesty_payload,
    input_tm,
    source_imports_no_backend,
    tm_dense,
    worked_example,
)


def test_g1_soundness_grid_and_sample() -> None:
    spec = TMNeuronSpec(order=1, activation="sigmoid")
    tm = tm_dense(input_tm(0.0, 0.2, spec.order), 1.0, 0.0, spec)
    assert contains_grid_and_sample(tm, "sigmoid", 0.0, 0.2)


def test_g2_non_vacuous() -> None:
    ex = worked_example()
    assert float(ex["remainder_width"]) < 0.1
    assert ex["vacuous"] is False
    assert float(ex["width"]) < 0.1


def test_g3_depth_honesty_records_explosion() -> None:
    report = depth_honesty()
    assert "enclosure_exploded" in report
    if float(report["remainder_width"]) > 1e3:
        assert report["enclosure_exploded"] is True


def test_g4_no_backend_leak() -> None:
    assert source_imports_no_backend()


def test_honesty() -> None:
    payload = honesty_payload()
    assert payload["imagenet_claim"] is False
    assert payload["stretch_claim"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not a deep-net" in DISCLAIMER
