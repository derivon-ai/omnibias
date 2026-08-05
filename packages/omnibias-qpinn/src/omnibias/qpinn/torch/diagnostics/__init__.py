# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Diagnostic scalars + soft losses for the torch backend.

Each function is a pure read of the :class:`FieldState`: no mutation.

- :mod:`norm`: :math:`\\int |\\psi|^2\\,dx` and its drift from unit.
- :mod:`energy`: variational :math:`\\langle\\hat H\\rangle` and its
  variance.
- :mod:`current`: probability current :math:`j_i = (\\hbar/m)\\,
  \\Im(\\psi^* \\partial_i \\psi)` and the continuity-equation residual.
"""

from __future__ import annotations

from omnibias.qpinn.torch.diagnostics.current import (
    continuity_residual,
    current_divergence,
    probability_current,
)
from omnibias.qpinn.torch.diagnostics.energy import (
    energy_variance,
    expectation_value,
    expected_energy,
)
from omnibias.qpinn.torch.diagnostics.norm import (
    norm_drift,
    norm_squared,
)
from omnibias.qpinn.torch.diagnostics.vortex import (
    VortexDetection,
    detect_vortices,
    detect_vortices_full,
    feynman_vortex_count,
    thomas_fermi_density_2d,
    thomas_fermi_mu_2d,
    thomas_fermi_radius_2d,
)

__all__ = [
    "VortexDetection",
    "continuity_residual",
    "current_divergence",
    "detect_vortices",
    "detect_vortices_full",
    "energy_variance",
    "expectation_value",
    "expected_energy",
    "feynman_vortex_count",
    "norm_drift",
    "norm_squared",
    "probability_current",
    "thomas_fermi_density_2d",
    "thomas_fermi_mu_2d",
    "thomas_fermi_radius_2d",
]
