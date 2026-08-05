# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Backend-agnostic (numpy) QUBO internals: conversion, decoding, oracle, bounds."""

from __future__ import annotations

from omnibias.qubo._core.bound import (
    gershgorin_min_eig_lower,
    lasserre_lower_bound,
    spectral_lower_bound,
)
from omnibias.qubo._core.convert import (
    boolean_constraints,
    ising_to_qubo,
    qubo_to_ising,
    to_polynomial,
)
from omnibias.qubo._core.decode import (
    brute_force_min,
    decode_qubo,
    energy,
    is_binary,
    one_flip_descent,
    round_relaxed,
)
from omnibias.qubo._core.frontends import max_cut, max_independent_set

__all__ = [
    "boolean_constraints",
    "brute_force_min",
    "decode_qubo",
    "energy",
    "gershgorin_min_eig_lower",
    "is_binary",
    "ising_to_qubo",
    "lasserre_lower_bound",
    "max_cut",
    "max_independent_set",
    "one_flip_descent",
    "qubo_to_ising",
    "round_relaxed",
    "spectral_lower_bound",
    "to_polynomial",
]
