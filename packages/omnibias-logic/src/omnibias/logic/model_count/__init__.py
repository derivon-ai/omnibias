# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""(Weighted) #SAT / model counting: exact fast-paths, a sound router, and a certified sandwich.

Every entry point here is **worst-case sound**, tagged by which corner of the exact / fast /
sound triangle it occupies (exact model counting is ``#P``-hard, so there is no general
poly-time exact counter -- these are honest fragments and enclosures, never a ``P = NP``
shortcut):

* **Exact + sound.** :func:`xor_model_count` (the affine/XOR fragment -- poly-time via GF(2)
  rank, unweighted), :func:`treewidth_model_count` (bounded-treewidth DP -- poly for bounded
  width, weighted), and :func:`count_models_exact` (a component-caching #DPLL counter --
  exact but exponential worst case, weighted). ``#2-SAT`` / ``#Horn-SAT`` are ``#P``-complete
  and deliberately *not* offered as fast paths.
* **Certified enclosure + sound.** :func:`count_enclosure` -- the truncated inclusion-
  exclusion (Bonferroni) ``lower <= #models <= upper`` sandwich (:class:`CountCertificate`).
* **Router.** :func:`count` auto-dispatches to the cheapest sound method and returns a tagged
  :class:`CountResult` (``guarantee in {"exact", "certified_enclosure"}``).

:func:`exact_model_count` is the ``O(2^n)`` enumeration oracle that self-checks all of the
above on small instances. Statistical (NOT worst-case sound) estimators live separately in
:mod:`omnibias.logic.approx` and are never re-exported here.
"""

from __future__ import annotations

from omnibias.logic.model_count.certificate import CountCertificate
from omnibias.logic.model_count.enclosure import count_enclosure
from omnibias.logic.model_count.exact import CountBudgetExceeded, count_models_exact
from omnibias.logic.model_count.frontends import model_count
from omnibias.logic.model_count.problem import ModelCountProblem, exact_model_count
from omnibias.logic.model_count.proof import (
    count_prover,
    model_count_conjecture,
    prove_model_count,
    seal_count_certificate,
)
from omnibias.logic.model_count.route import CountResult, count
from omnibias.logic.model_count.treewidth import TreewidthTooLarge, treewidth_model_count
from omnibias.logic.model_count.xor import XORClause, detect_xor_system, xor_model_count

__all__ = [
    "CountBudgetExceeded",
    "CountCertificate",
    "CountResult",
    "ModelCountProblem",
    "TreewidthTooLarge",
    "XORClause",
    "count",
    "count_enclosure",
    "count_models_exact",
    "count_prover",
    "detect_xor_system",
    "exact_model_count",
    "model_count",
    "model_count_conjecture",
    "prove_model_count",
    "seal_count_certificate",
    "treewidth_model_count",
    "xor_model_count",
]
