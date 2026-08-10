# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch backend for omnibias-pinn.

Public submodules:

- :mod:`fields` -- typed PINN field types (``OneLayerVectorField``,
  ``SpectralVectorField``, ``ChebyshevVectorField``).
- :mod:`ops` -- functional operator surface (kernel for the attribute
  DSL on :class:`FieldState`).
- :mod:`cage` -- strict-conservation layer wrappers (incompressibility,
  energy, enstrophy, hard boundaries).
- :mod:`losses` -- loss-function utilities (Sobolev preconditioner,
  WP-causal, entropy, NTK-rebalance).
- :mod:`equations` -- prebuilt PDE residuals (NS, Burgers, heat, KS, CH,
  biharmonic).
- :mod:`diagnostics` -- bit-stability sweeps, forecast-horizon, spectral
  fidelity, autograd-phase check.
"""

from __future__ import annotations

from omnibias.pinn.torch import (
    cage,
    diagnostics,
    discovery,
    equations,
    fields,
    losses,
    ops,
)

__all__ = [
    "cage",
    "diagnostics",
    "discovery",
    "equations",
    "fields",
    "losses",
    "ops",
]
