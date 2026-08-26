# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named-collapse catalogue sitting beside the founding three senses.

The founding **bias** (``delta -> 0``), **temperature** (``beta -> inf``),
and **enclosure** (``width -> 0`` of a sound enclosure) limits stay
exactly those three. This package catalogues them and hosts later named
collapses that earn a slot by minting a *different* surviving object.

Do not conflate the three founding senses. A float residual is never a
proof. ``theorem_prover_verified`` stays false unless a genuine
``lake build`` earned it.
"""

from __future__ import annotations

from omnibias.core.collapse.identity import (
    IDENTITY_SPEC,
    difference_coeffs,
    evaluate_difference,
    identity_collapse,
    remainder_collapse,
)
from omnibias.core.collapse.pairing import (
    PAIRING_SPEC,
    pairing_collapse,
    pairing_value,
)
from omnibias.core.collapse.schema import (
    FOUNDING_COLLAPSES,
    FOUNDING_NAMES,
    CollapseOutcome,
    CollapseRegister,
    CollapseRegistry,
    CollapseSpec,
    CollapseStatus,
    DistinctnessReport,
    RejectedCollapse,
    add_registry_hook,
    are_distinct,
    default_honesty,
    get_collapse,
    list_collapses,
    list_rejected_collapses,
    register_collapse,
    reject_collapse,
    require_sound_enclosure,
    reset_collapse_registry,
)
from omnibias.core.collapse.verdict import (
    VERDICT_SPEC,
    ObligationVerdict,
    VerdictStatus,
    adjudicate_residual,
    is_singleton_zero,
    search_residuals,
)
from omnibias.core.collapse.winding import (
    WINDING_SPEC,
    winding_collapse,
    winding_enclosure,
)

__all__ = [
    "CollapseOutcome",
    "CollapseRegister",
    "CollapseRegistry",
    "CollapseSpec",
    "CollapseStatus",
    "DistinctnessReport",
    "FOUNDING_COLLAPSES",
    "FOUNDING_NAMES",
    "IDENTITY_SPEC",
    "ObligationVerdict",
    "PAIRING_SPEC",
    "RejectedCollapse",
    "VERDICT_SPEC",
    "VerdictStatus",
    "WINDING_SPEC",
    "add_registry_hook",
    "adjudicate_residual",
    "are_distinct",
    "default_honesty",
    "difference_coeffs",
    "evaluate_difference",
    "get_collapse",
    "identity_collapse",
    "is_singleton_zero",
    "list_collapses",
    "list_rejected_collapses",
    "pairing_collapse",
    "pairing_value",
    "register_collapse",
    "reject_collapse",
    "remainder_collapse",
    "require_sound_enclosure",
    "reset_collapse_registry",
    "search_residuals",
    "winding_collapse",
    "winding_enclosure",
]
