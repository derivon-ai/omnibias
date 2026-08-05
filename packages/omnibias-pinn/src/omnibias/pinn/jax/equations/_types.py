# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared NamedTuple output types for the equation registry (jax twin)."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class NavierStokesOutput(NamedTuple):
    residual: Array
    continuity: Array
    diag: dict[str, float]


class BurgersOutput(NamedTuple):
    residual: Array
    diag: dict[str, float]


class HeatOutput(NamedTuple):
    residual: Array
    diag: dict[str, float]


class KSOutput(NamedTuple):
    residual: Array
    diag: dict[str, float]


class CHOutput(NamedTuple):
    residual: Array
    diag: dict[str, float]


class CCFOutput(NamedTuple):
    residual: Array
    hilbert: Array
    diag: dict[str, float]


class BiharmonicOutput(NamedTuple):
    residual: Array
    diag: dict[str, float]


class FredholmOutput(NamedTuple):
    residual: Array
    integral: Array
    diag: dict[str, Array]


class VolterraOutput(NamedTuple):
    residual: Array
    integral: Array
    diag: dict[str, Array]


__all__ = [
    "BiharmonicOutput",
    "BurgersOutput",
    "CCFOutput",
    "CHOutput",
    "FredholmOutput",
    "HeatOutput",
    "KSOutput",
    "NavierStokesOutput",
    "VolterraOutput",
]
