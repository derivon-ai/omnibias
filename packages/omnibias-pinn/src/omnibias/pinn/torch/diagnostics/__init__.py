# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Diagnostics for the torch backend.

Three families:

* **Trajectory metrics** (numpy-only, lifted from the KS benchmark):
  :func:`relative_l2_per_time`, :func:`forecast_horizon`,
  :func:`spectral_fidelity`. These are re-exported from
  :mod:`omnibias.pinn._core.diagnostics`.

* **Bit-stability sweep** (:func:`derivative_stability`): for a typed
  field at fixed coords, sweep the polylaplacian order ``k = 1..max_k``
  and compare the closed-form result to autograd-via-Hessian
  iteration. Produces the table that drives ``docs/stability.md``.

* **Autograd phase check** (:func:`autograd_phase_check`): wallclock
  curve for autograd derivatives at orders ``2..max_order``. Used to
  substantiate the autograd-phase-transition wallclock claim
  documented in ``CHANGELOG.md`` and the project's private benchmark
  archive.
"""

from __future__ import annotations

from omnibias.pinn._core.diagnostics import (
    forecast_horizon,
    power_spectrum_per_d,
    relative_l2_per_time,
    spectral_fidelity,
)
from omnibias.pinn._core.multiscale import (
    dominant_wavenumbers,
    geometric_bands,
    suggest_frequency_bands,
)
from omnibias.pinn.torch.diagnostics.field_stability import (
    autograd_phase_check,
    derivative_stability,
)

__all__ = [
    "autograd_phase_check",
    "derivative_stability",
    "dominant_wavenumbers",
    "forecast_horizon",
    "geometric_bands",
    "power_spectrum_per_d",
    "relative_l2_per_time",
    "spectral_fidelity",
    "suggest_frequency_bands",
]
