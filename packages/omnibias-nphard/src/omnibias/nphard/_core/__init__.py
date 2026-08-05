# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Backend-agnostic (numpy / scipy) core of omnibias-nphard.

The named NP-hard families (:mod:`.qap`, :mod:`.gap`, :mod:`.scheduling`) each implement
the ``omnibias-discrete`` ``DiscreteProblem`` seam and a ``to_qubo()`` reduction; the
structure-preserving decoders + named classical baselines + exact exponential oracle
live in :mod:`.decode`; the predict-then-optimize metrics live in :mod:`.decision`.
These modules import numpy / scipy only -- never a tensor backend.
"""

from __future__ import annotations

from omnibias.nphard._core.bound import gilmore_lawler_bound
from omnibias.nphard._core.decision import normalized_regret, spo_plus_gradient
from omnibias.nphard._core.decode import brute_force_min, classical_optimum, decode
from omnibias.nphard._core.gap import GAPProblem, gap
from omnibias.nphard._core.qap import QAPProblem, placement_qap, qap
from omnibias.nphard._core.scheduling import SchedulingProblem, schedule

__all__ = [
    "GAPProblem",
    "QAPProblem",
    "SchedulingProblem",
    "brute_force_min",
    "classical_optimum",
    "decode",
    "gap",
    "gilmore_lawler_bound",
    "normalized_regret",
    "placement_qap",
    "qap",
    "schedule",
    "spo_plus_gradient",
]
