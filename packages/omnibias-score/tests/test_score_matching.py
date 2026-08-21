# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Score-package twins for theory 09-21."""

from __future__ import annotations

import torch
from omnibias.score.torch.score_matching import exact_div_neg_id, worked_example


def test_g1_torch() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert abs(ex["div"] + 1.0) < 1e-12
    assert exact_div_neg_id(1) == -1.0
    assert exact_div_neg_id(torch.tensor(0.0)) == -1.0
