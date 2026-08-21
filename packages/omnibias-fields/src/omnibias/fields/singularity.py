# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Field front end for jet-Padé singularity tracking (theory 03-10).

Re-exports the difference-package tracker. This is a diagnostic
and an estimate, not a proof of blow-up. Jets come from the
founding bias collapse (``delta -> 0``). Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate
the two.
"""

from __future__ import annotations

from omnibias.difference.singularity import (
    DISCLAIMER,
    BlowupFit,
    SingularityEstimate,
    SingularityTrack,
    honesty_payload,
    track_singularity,
)

__all__ = [
    "BlowupFit",
    "DISCLAIMER",
    "SingularityEstimate",
    "SingularityTrack",
    "honesty_payload",
    "track_singularity",
]
