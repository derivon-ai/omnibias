# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Pure-Python validated-dynamics core (depends only on ``omnibias.core.verified``).

The rigorous engines (variational / monodromy flow, Poincare enclosures,
certified Lyapunov bounds, periodic-orbit existence) are built on top of the
QR-Lohner flow and the radii-polynomial / Krawczyk machinery in
:mod:`omnibias.core.verified`.
"""

from __future__ import annotations

from omnibias.dynamics._core.fields import (
    harmonic_oscillator,
    hopf_normal_form,
    linear_system,
    radial_logistic,
)
from omnibias.dynamics._core.jet_bridge import (
    DiscretePeriodicOrbit,
    discrete_periodic_point,
    sigma_oscillator_field,
    vector_field_from_sigma_tower,
)
from omnibias.dynamics._core.lyapunov import (
    LyapunovBounds,
    certified_lyapunov_exponent,
)
from omnibias.dynamics._core.orbits import (
    PeriodicOrbitCertificate,
    prove_periodic_orbit,
)
from omnibias.dynamics._core.poincare import (
    PoincareCrossing,
    PoincareSection,
    poincare_map,
)
from omnibias.dynamics._core.variational import (
    VariationalState,
    monodromy_determinant,
    monodromy_matrix,
    monodromy_trace,
    spectral_radius_bound,
    step_transition_matrix,
    variational_flow,
    variational_step,
)

__all__ = [
    "DiscretePeriodicOrbit",
    "LyapunovBounds",
    "PeriodicOrbitCertificate",
    "PoincareCrossing",
    "PoincareSection",
    "VariationalState",
    "certified_lyapunov_exponent",
    "discrete_periodic_point",
    "harmonic_oscillator",
    "hopf_normal_form",
    "linear_system",
    "monodromy_determinant",
    "monodromy_matrix",
    "monodromy_trace",
    "poincare_map",
    "prove_periodic_orbit",
    "radial_logistic",
    "sigma_oscillator_field",
    "spectral_radius_bound",
    "step_transition_matrix",
    "variational_flow",
    "variational_step",
    "vector_field_from_sigma_tower",
]
