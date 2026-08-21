# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parameter-space mixed jets (operator schemas; theory 09-27).

founding bias collapse (``delta -> 0``). Temperature collapse
(``beta -> inf``, feasibility) does not appear. do not conflate
the two. Not a ParamPINN package.
"""

from __future__ import annotations

from omnibias.core.parameter_jets import (
    DEFAULT_SPEC,
    DISCLAIMER,
    ParameterJetSpec,
    fourier_heat,
    fourier_heat_du_dmu,
    honesty_payload,
    mixed_jet,
    parameter_jet_skill,
    worked_example,
)

__all__ = [
    "DEFAULT_SPEC",
    "DISCLAIMER",
    "ParameterJetSpec",
    "fourier_heat",
    "fourier_heat_du_dmu",
    "honesty_payload",
    "mixed_jet",
    "parameter_jet_skill",
    "worked_example",
]
