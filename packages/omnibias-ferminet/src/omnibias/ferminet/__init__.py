# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias.ferminet: FermiNet bridge for omnibias.

This sub-package wires omnibias's closed-form n-th derivative kernels
into FermiNet-style neural variational Monte Carlo. Seven public surfaces:

* :mod:`omnibias.ferminet.folx_compat` -- folx-compatible
  ``forward_laplacian`` / ``closed_form_forward_laplacian`` /
  ``laplacian_factory`` adapters. Drop-in replacement for the folx
  Laplacian in FermiNet's ``laplacian_method`` switch.
* :mod:`omnibias.ferminet.integration` -- the envelope bridge:
  envelope value / gradient / Hessian kernels, optional one-body
  backflow, and the production
  ``make_omnibias_envelope_local_kinetic_energy`` /
  ``make_omnibias_tier2_local_kinetic_energy`` factories that the
  upstream FermiNet ``laplacian_method == 'omnibias_envelope'`` and
  ``laplacian_method == 'omnibias_tier2'`` branches consume.
* :mod:`omnibias.ferminet.restricted` -- the Tier-2 / Tier-2-full
  restricted FermiNet ansatz with closed-form Laplacian. Used for
  bringing up an end-to-end omnibias-only path before integrating
  with the upstream FermiNet checkpoint.
* :mod:`omnibias.ferminet.jastrow` -- closed-form symmetric
  Pade-Jastrow correlation factor (value / gradient / Laplacian) with
  Kato electron-electron and electron-nucleus cusps, wired into the
  Tier-2 local kinetic energy via
  ``jastrow_slater_local_kinetic_energy``.
* :mod:`omnibias.ferminet.multiblock` -- multi-block FermiNet
  primitives (per-geometry blocks for nuclear-Hessian work).
* :mod:`omnibias.ferminet.multiblock_integration` -- composition of
  multi-block primitives into a FermiNet-shaped log|psi| / local
  kinetic energy.
* :mod:`omnibias.ferminet.sampling` -- a reproducible, ansatz-agnostic
  quantum Monte-Carlo sampling substrate: Metropolis / Langevin (MALA)
  walkers over a caller-supplied ``log|psi|``, a local-energy
  estimator, autocorrelation / effective-sample-size error
  diagnostics, and a walker-level parameter log-derivative
  accumulator. See ``docs/api/qmc_sampling.md``.
* :mod:`omnibias.ferminet.stochastic_reconfiguration` -- the
  stochastic-reconfiguration (SR) VMC optimizer step: Monte-Carlo
  overlap-matrix / energy-gradient estimators built on
  ``sampling``'s accumulator, composed unmodified with
  :mod:`omnibias.curvature.natural_gradient`'s ``damped_solve`` /
  ``natural_gradient_step`` for the parameter update. Requires the
  optional ``curvature`` extra (``omnibias-ferminet[curvature]``).
  See ``docs/api/stochastic_reconfiguration.md``.

Importing :mod:`omnibias.ferminet` does **not** trigger the FermiNet
or ``ferminet`` package import; the bridge is callable without a
FermiNet checkout (mock log|psi| factories are provided), and the
real integration is exercised through the upstream FermiNet
``laplacian_method`` switch.

Theory and integration plan: see ``CHANGELOG.md`` and
``docs/roadmap.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("omnibias-ferminet")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Lazy attribute access keeps `import omnibias.ferminet` cheap; importing
# the integration / restricted modules eagerly imports JAX which may not
# be desired in some scripts.
# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "bias collapse"

__all__ = [
    "__lineage__",
    "__version__",
]
