# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias-submodular: differentiable + certified monotone submodular maximization.

Maximizing a monotone submodular ``f: 2^[n] -> R>=0`` over the independent sets of a
matroid (cardinality / partition) is NP-hard, so no poly-time map yields the *exact*
optimum (that would imply P = NP). The sound "differentiable submodular maximization"
this package delivers is a three-part object -- **yes, if** you accept a certified
approximation instead of an exactness claim:

1. the **multilinear extension** ``F(p) = E_{x ~ p}[f(x)]`` replaces ``x in {0,1}^n`` by
   ``p in [0,1]^n`` (the unique multilinear polynomial interpolating ``f``), and
   **continuous greedy** -- Frank-Wolfe over the matroid polytope -- returns a fractional
   ``p*`` with ``F(p*) >= (1 - 1/e) OPT``. The exact per-coordinate gradient keeps the
   step closed-form; the differentiable soft LP oracle is *unrolled* for backprop so a
   model predicting the objective's data trains *through* it
   (:mod:`omnibias.submodular.jax` / :mod:`omnibias.submodular.torch`, bit-identical
   twins for the coverage family);
2. **pipage / swap rounding** turns ``p*`` into an integral independent set ``S`` with
   ``f(S) >= F(p*)``, hence the a-priori certified ``f(S) >= (1 - 1/e) OPT``
   (:func:`maximize`; :func:`brute_force_max` is the exact small-``n`` oracle);
3. a **rigorous gap certificate** (:func:`certify_submodular_gap`): the decoded value is
   a *lower* bound and a marginal-gain bound ``U(S)`` an *upper* bound on ``OPT``, so
   ``f(S) <= OPT <= U(S)`` is a certified gap -- never asserted zero.

Builds on the ``omnibias-discrete`` substrate: :class:`SubmodularProblem` implements the
``DiscreteProblem`` seam (``energy = -f``), so ``brute_force_min`` / ``certify_gap`` also
apply to the (for monotone ``f``, trivial) unconstrained view. The relaxation twins need
a ``jax`` / ``torch`` backend.

Scope note -- **minimization is different from maximization**. Unconstrained submodular
*minimization* ``min_S f(S)`` is a **P-class** problem, solvable **exactly** in polynomial
time (:func:`submodular_minimize` / :func:`min_norm_point`, Fujishige-Wolfe over the base
polytope; :func:`lovasz_extension` is its convex closure). Asserting this exactness is *not*
a ``P = NP`` claim: it is the NP-hard *maximization* that only ever gets the certified
``1 - 1/e`` approximation above, never an exact solver.

Terminology: the multilinear extension relaxing ``{0,1}^n -> [0,1]^n`` and the
Frank-Wolfe oracle ``sigmoid(beta (g - tau))``, ``beta -> inf``, hardening onto a ``0/1``
matroid-basis vertex is the **feasibility** / temperature sense of "collapse" (a soft
indicator becoming a step), distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit of an ``OMBU`` to the closed-form derivative ``sigma^(K-1)``; see
``docs/theory.md``).

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

from omnibias.discrete import brute_force_min, certify_gap
from omnibias.submodular._core.bound import (
    lasserre_lower_bound,
    marginal_upper_bound,
    modular_upper_bound,
    negative_coeff_lower_bound,
    total_curvature,
)
from omnibias.submodular._core.continuous import continuous_greedy
from omnibias.submodular._core.frontends import (
    budget_additive,
    facility_location,
    log_det_dpp,
    max_coverage,
)
from omnibias.submodular._core.greedy import (
    brute_force_max,
    greedy_maximize,
    lazy_greedy,
    local_search,
    p_matroid_greedy,
    stochastic_greedy,
)
from omnibias.submodular._core.nonmonotone import (
    double_greedy,
    measured_continuous_greedy,
    nonmonotone_upper_bound,
)
from omnibias.submodular._core.pipeline import maximize
from omnibias.submodular._core.rounding import pipage_round, swap_round
from omnibias.submodular._core.streaming import sieve_streaming
from omnibias.submodular.certify import (
    certify_nonmonotone_gap,
    certify_submodular_gap,
    certify_unconstrained_gap,
    verify_guarantee,
)
from omnibias.submodular.functions import (
    BudgetAdditive,
    Coverage,
    FacilityLocation,
    GraphCut,
    LogDeterminant,
    Saturated,
    Scaled,
    SubmodularFunction,
    Sum,
    indicator,
    is_monotone_submodular,
)
from omnibias.submodular.knapsack import (
    KnapsackConstraint,
    brute_force_max_knapsack,
    budgeted,
    certify_knapsack_gap,
    cost_benefit_greedy,
    knapsack_maximize,
)
from omnibias.submodular.lovasz import (
    MinimizerResult,
    lovasz_extension,
    min_norm_point,
    submodular_minimize,
)
from omnibias.submodular.matroid import (
    GraphicMatroid,
    LaminarMatroid,
    Matroid,
    MatroidIntersection,
    PartitionMatroid,
    TransversalMatroid,
    UniformMatroid,
)
from omnibias.submodular.problem import (
    ONE_MINUS_INV_E,
    ContinuousGreedySchedule,
    SubmodularCertificate,
    SubmodularProblem,
    SubmodularSolution,
)

try:
    __version__ = _pkg_version("omnibias-submodular")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "temperature collapse"

__all__ = [
    "BudgetAdditive",
    "ContinuousGreedySchedule",
    "Coverage",
    "FacilityLocation",
    "GraphCut",
    "GraphicMatroid",
    "KnapsackConstraint",
    "LaminarMatroid",
    "LogDeterminant",
    "Matroid",
    "MatroidIntersection",
    "MinimizerResult",
    "ONE_MINUS_INV_E",
    "PartitionMatroid",
    "Saturated",
    "Scaled",
    "SubmodularCertificate",
    "SubmodularFunction",
    "SubmodularProblem",
    "SubmodularSolution",
    "Sum",
    "TransversalMatroid",
    "UniformMatroid",
    "__lineage__",
    "__version__",
    "brute_force_max",
    "brute_force_max_knapsack",
    "brute_force_min",
    "budget_additive",
    "budgeted",
    "certify_gap",
    "certify_knapsack_gap",
    "certify_nonmonotone_gap",
    "certify_submodular_gap",
    "certify_unconstrained_gap",
    "continuous_greedy",
    "cost_benefit_greedy",
    "double_greedy",
    "facility_location",
    "greedy_maximize",
    "indicator",
    "is_monotone_submodular",
    "knapsack_maximize",
    "lasserre_lower_bound",
    "lazy_greedy",
    "local_search",
    "log_det_dpp",
    "lovasz_extension",
    "marginal_upper_bound",
    "max_coverage",
    "maximize",
    "measured_continuous_greedy",
    "min_norm_point",
    "modular_upper_bound",
    "negative_coeff_lower_bound",
    "nonmonotone_upper_bound",
    "p_matroid_greedy",
    "pipage_round",
    "sieve_streaming",
    "stochastic_greedy",
    "submodular_minimize",
    "swap_round",
    "total_curvature",
    "verify_guarantee",
]
