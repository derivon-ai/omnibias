# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""omnibias.discrete.maxsat: weighted MaxSAT as a pseudo-Boolean ``DiscreteProblem``.

The substrate's first in-tree consumer. Encode a weighted CNF with :func:`max_sat` (or
build a :class:`WeightedCNF` directly) and reuse the shared pipeline:
:func:`omnibias.discrete.decode` for a min-violation assignment (upper bound),
:func:`omnibias.discrete.certify_gap` for the certified optimality gap, and the
``maxsat_relaxation`` twins in :mod:`omnibias.discrete.maxsat.jax` /
:mod:`omnibias.discrete.maxsat.torch` for the differentiable relaxation.
"""

from __future__ import annotations

from omnibias.discrete.maxsat.frontends import max_sat
from omnibias.discrete.maxsat.problem import Clause, MaxSATProblem, WeightedCNF

__all__ = ["Clause", "MaxSATProblem", "WeightedCNF", "max_sat"]
