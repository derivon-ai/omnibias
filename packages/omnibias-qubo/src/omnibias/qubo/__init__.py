# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias-qubo: differentiable + certified quadratic Boolean optimization.

Minimizing a quadratic pseudo-Boolean energy ``E(x) = x^T Q x + c^T x`` over
``x in {0, 1}^n`` (QUBO / Ising) is NP-hard, so no poly-time differentiable map yields
the *exact* global optimum (that would imply P = NP, and the exact argmin's gradient is
a.e. zero). The sound "differentiable QUBO" this package delivers is a three-part object
-- **yes, if** you accept a certified gap instead of an exactness claim:

1. a **differentiable annealed relaxation** -- a soft assignment
   ``x = sigmoid(beta z) in (0, 1)^n`` descended on the closed-form energy gradient while
   ``beta -> inf`` collapses it onto a binary vertex, *unrolled* for backprop so a model
   predicting ``Q`` / ``c`` trains *through* the optimizer
   (:mod:`omnibias.qubo.jax` / :mod:`omnibias.qubo.torch`, bit-identical twins);
2. a **heuristic decoder** -- rounding + 1-flip local search (:func:`decode_qubo`), an
   *upper* bound (:func:`brute_force_min` is the exact small-``n`` oracle);
3. a **rigorous optimality-gap certificate** (:func:`certify_qubo_gap`): a Lasserre / SOS
   bound over the Boolean hypercube (:mod:`omnibias.sos`) or a cheap spectral / box-QP
   bound (:mod:`omnibias.convex`) is a *lower* bound on the true optimum, so
   ``lower <= optimum <= energy`` is a certified gap -- never asserted zero.

The relaxation layers need a ``jax`` / ``torch`` backend; the SOS certificate needs the
``sos`` extra and the spectral seal the ``convex`` extra (each degrades gracefully).

Terminology: the relaxation's ``sigmoid(beta z)``, ``beta -> inf`` is the feasibility /
temperature sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct
from the **founding bias collapse** (the multi-bias ``delta -> 0`` limit of an ``OMBU``
to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).

.. important::

    **Bit-parity with the PyTorch twin requires 64-bit JAX** --
    ``jax.config.update("jax_enable_x64", True)`` before the first JAX array is
    created (or ``JAX_ENABLE_X64=1``). JAX otherwise truncates to ``float32``
    while PyTorch uses ``float64``, so the twins stay internally consistent but
    agree only to ``float32`` tolerance. Where a value feeds a threshold, a
    rounding step or an ``argmax``, that is enough to change the decision rather
    than just the last digits. See :mod:`omnibias.jax.precision`.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

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
from omnibias.qubo.certify import certify_qubo_gap
from omnibias.qubo.problem import (
    AnnealSchedule,
    IsingProblem,
    QUBOCertificate,
    QUBOProblem,
    QUBOSolution,
)

try:
    __version__ = _pkg_version("omnibias-qubo")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "AnnealSchedule",
    "IsingProblem",
    "QUBOCertificate",
    "QUBOProblem",
    "QUBOSolution",
    "__lineage__",
    "__version__",
    "boolean_constraints",
    "brute_force_min",
    "certify_qubo_gap",
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
