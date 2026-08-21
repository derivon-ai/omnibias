# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Score-flow twins for theory 09-06."""

from __future__ import annotations

import math

import torch
from omnibias.score.flow.torch.jet_flow import jet_flow_forward, jet_flow_inverse, worked_example


def test_g1_torch_matches_closed_form() -> None:
    torch.set_default_dtype(torch.float64)
    x = torch.tensor(0.3)
    scale = torch.tensor(2.0)
    y, log_det = jet_flow_forward(x, scale)
    closed = math.log(2.0 * (1.0 - math.tanh(0.6) ** 2))
    assert abs(float(log_det) - closed) < 1e-12
    z = jet_flow_inverse(y, scale)
    assert abs(float(z) - 0.3) < 1e-10
    assert worked_example()["log_det_err"] < 1e-12
