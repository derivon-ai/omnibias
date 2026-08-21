# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""World-model-as-jet (dynamics re-export; theory 09-25).

founding bias collapse (``delta -> 0``). Temperature collapse
(``beta -> inf``, feasibility) does not appear. do not conflate
the two. Not NS global regularity. Imports neither torch nor jax.
"""

from __future__ import annotations

from omnibias.core.jet_world import (
    DEFAULT_CONFIG,
    DISCLAIMER,
    JetWorldConfig,
    box_exits_safe,
    honesty_payload,
    jet_world_skill,
    lohner_oscillator_step,
    lohner_plan,
    oscillator_taylor_x,
    predict_next_jet,
    worked_example,
)

__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "JetWorldConfig",
    "box_exits_safe",
    "honesty_payload",
    "jet_world_skill",
    "lohner_oscillator_step",
    "lohner_plan",
    "oscillator_taylor_x",
    "predict_next_jet",
    "worked_example",
]
