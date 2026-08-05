# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""omnibias-dynamics: computer-assisted dynamics on the validated tower.

Rigorous (interval / Taylor-model) tools for nonlinear dynamical systems, built
on the QR-Lohner validated flow, the Newton-Kantorovich / radii-polynomial
existence machinery, and the closed-form variational tower in
:mod:`omnibias.core.verified`:

* **variational / monodromy flow** -- propagate a state *and* its fundamental
  (variational) matrix rigorously, the basis for Floquet / stability analysis;
* **Poincare-section enclosures** -- a rigorous return map across a hyperplane;
* **certified Lyapunov-exponent bounds** -- two-sided enclosures of the leading
  exponent from the validated variational flow;
* **periodic-orbit existence** -- a radii-polynomial proof that a true periodic
  orbit lives in an explicit ball around a numerical guess.

Everything is *sound by construction*: an enclosure provably contains the true
object, and an existence claim is a proof, never a heuristic.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.dynamics._core import (
    DiscretePeriodicOrbit,
    LyapunovBounds,
    PeriodicOrbitCertificate,
    PoincareCrossing,
    PoincareSection,
    VariationalState,
    certified_lyapunov_exponent,
    discrete_periodic_point,
    harmonic_oscillator,
    hopf_normal_form,
    linear_system,
    monodromy_determinant,
    monodromy_matrix,
    monodromy_trace,
    poincare_map,
    prove_periodic_orbit,
    radial_logistic,
    sigma_oscillator_field,
    spectral_radius_bound,
    step_transition_matrix,
    variational_flow,
    variational_step,
    vector_field_from_sigma_tower,
)

try:
    __version__ = _pkg_version("omnibias-dynamics")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "DiscretePeriodicOrbit",
    "LyapunovBounds",
    "PeriodicOrbitCertificate",
    "PoincareCrossing",
    "PoincareSection",
    "VariationalState",
    "__lineage__",
    "__version__",
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
