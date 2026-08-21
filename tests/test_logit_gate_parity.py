# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Numpy / torch / jax logit-gate parity on a shared feature row."""

from __future__ import annotations

import pytest
from omnibias.symbolic.conditions import observation_tanh
from omnibias.symbolic.gate import (
    LogitGate,
    logit_gate_logits_jax,
    logit_gate_logits_numpy,
    logit_gate_logits_torch,
)


def test_logit_gate_numpy_torch_jax_parity() -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)

    obs = observation_tanh()
    gate = LogitGate(sorts=("jet_monomial", "ore", "sos_template"))
    gate.train([(obs, "jet_monomial")])
    features = obs.features()
    numpy_logits = logit_gate_logits_numpy(gate.weights, gate.bias, features)
    torch_logits = logit_gate_logits_torch(gate.weights, gate.bias, features)
    jax_logits = logit_gate_logits_jax(gate.weights, gate.bias, features)
    assert numpy_logits == pytest.approx(torch_logits, abs=1e-12)
    assert numpy_logits == pytest.approx(jax_logits, abs=1e-12)
    assert gate.propose(obs)[0] == "jet_monomial"
