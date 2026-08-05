# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared NamedTuple output types for the quantum equation registry (jax)."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class TISEOutput(NamedTuple):
    """Output of :class:`omnibias.qpinn.jax.equations.TISE`.

    Attributes
    ----------
    residual
        Stacked ``(B, 2)`` residual for the real and imaginary channels
        of ``(H - E) psi``.
    energy_estimate
        Scalar variational estimate of :math:`E` from
        ``<psi | H | psi> / <psi | psi>``. ``None`` if no
        ``quadrature_weights`` provided.
    diag
        Plain-Python dict of scalar diagnostics for logging.
    """

    residual: Array
    energy_estimate: Array | None
    diag: dict[str, float]


class TDSEOutput(NamedTuple):
    """Output of :class:`omnibias.qpinn.jax.equations.TDSE`.

    Attributes
    ----------
    residual
        Stacked ``(B, 2)`` residual.
    diag
        Diagnostic dict.
    """

    residual: Array
    diag: dict[str, float]


class NLSOutput(NamedTuple):
    """Output of :class:`omnibias.qpinn.jax.equations.NLS`.

    Attributes
    ----------
    residual
        Stacked ``(B, 2)`` residual including the
        :math:`g\\,|\\psi|^2\\,\\psi` term.
    diag
        Diagnostic dict.
    """

    residual: Array
    diag: dict[str, float]


class RotatingNLSOutput(NamedTuple):
    """Output of :class:`omnibias.qpinn.jax.equations.RotatingNLS`.

    Stationary 2D Gross-Pitaevskii in the rotating frame.

    Attributes
    ----------
    residual
        Stacked ``(B, 2)`` residual.
    density
        ``|psi|^2`` at every collocation point.
    diag
        Diagnostic dict.
    """

    residual: Array
    density: Array
    diag: dict[str, float]


__all__ = [
    "NLSOutput",
    "RotatingNLSOutput",
    "TDSEOutput",
    "TISEOutput",
]
