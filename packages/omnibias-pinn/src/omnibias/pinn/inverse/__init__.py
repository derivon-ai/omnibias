# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Inverse problems and imaging (theory 05-01, gated).

Application submodule, not a package. Distinct from
``omnibias.pinn.solver.torch.inverse`` (PDE-coefficient recovery).
"""

from __future__ import annotations

from omnibias.pinn.inverse._core import (
    SUBMODULAR_GUARANTEE,
    BoundaryTrack,
    GuaranteeKindError,
    IdentifiabilityReport,
    InterfaceEstimate,
    InverseRegularizerError,
    LayerStack,
    SensorPlan,
    admissible_band,
    enclose_peak,
    honesty_payload,
    identifiability,
    identify_jump_order,
    invert_layered,
    level_set_front,
    locate_interface,
    location_fisher,
    merge_guarantees,
    piecewise_jump_field,
    place_sensors,
    polish_interface,
    scan_response,
    sigma_n,
    stefan_front,
    stefan_partials,
    stefan_temperature,
    track_free_boundary,
    tv_deconvolution_localize,
    worked_example,
)

__all__ = [
    "BoundaryTrack",
    "GuaranteeKindError",
    "IdentifiabilityReport",
    "InterfaceEstimate",
    "InverseRegularizerError",
    "LayerStack",
    "SUBMODULAR_GUARANTEE",
    "SensorPlan",
    "admissible_band",
    "enclose_peak",
    "honesty_payload",
    "identifiability",
    "identify_jump_order",
    "invert_layered",
    "level_set_front",
    "locate_interface",
    "location_fisher",
    "merge_guarantees",
    "piecewise_jump_field",
    "place_sensors",
    "polish_interface",
    "scan_response",
    "sigma_n",
    "stefan_front",
    "stefan_partials",
    "stefan_temperature",
    "track_free_boundary",
    "tv_deconvolution_localize",
    "worked_example",
]
