# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact score matching (flow torch re-export; theory 09-21).

founding bias collapse (``delta -> 0``). Temperature collapse
(``beta -> inf``, feasibility) does not appear. do not conflate
the two. Not a new CNF ``div``.
"""

from __future__ import annotations

from omnibias.score.torch.score_matching import (
    DISCLAIMER,
    ExactSMConfig,
    exact_div_neg_id,
    exact_score_matching_loss,
    honesty_payload,
    worked_example,
)

__all__ = [
    "DISCLAIMER",
    "ExactSMConfig",
    "exact_div_neg_id",
    "exact_score_matching_loss",
    "honesty_payload",
    "worked_example",
]
