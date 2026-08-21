# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sliced optimal transport of activation mixtures (theory 03-04).

One-dimensional ``W_1`` is exact (founding bias collapse, ``delta -> 0``).
The directional average is sampled. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the two.
Sliced Wasserstein is not Wasserstein. Not sample-free.
"""

from __future__ import annotations

from omnibias.measure.transport._core import (
    ActivationMixture,
    SlicedResult,
    bootstrap_direction_stderr,
    honesty_payload,
    krawczyk_root_count,
    named_worked_pair,
    quantile,
    sliced_wasserstein,
    w1_exact,
    worked_w1,
    wp_quantile,
)

__all__ = [
    "ActivationMixture",
    "SlicedResult",
    "bootstrap_direction_stderr",
    "honesty_payload",
    "krawczyk_root_count",
    "named_worked_pair",
    "quantile",
    "sliced_wasserstein",
    "w1_exact",
    "worked_w1",
    "wp_quantile",
]
