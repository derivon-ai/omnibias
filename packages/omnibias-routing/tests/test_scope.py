# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Scope / honesty guards: routing ships a *certified gap*, never a P=NP claim.

The "yes-if" contract: every tour-producing API returns a valid tour with a
certified/reported optimality *gap*; nothing advertises a poly-time route to the
*exact* optimal tour (the one honest hard limit is kept). The only exact solver,
:func:`held_karp_dp`, is explicitly exponential (small-``n`` oracle).
"""

from __future__ import annotations

import sys

import numpy as np
import omnibias.routing as R
import pytest
from omnibias.routing import (
    HeldKarpCertificate,
    RoutingProblem,
    certify_tour_gap,
    decode_tour,
)

# names that would (falsely) advertise a poly-time exact TSP solver
_FORBIDDEN = ("solve_tsp", "optimal_tour", "exact_tour", "best_tour", "tsp_solve", "solve_tour")


def test_package_docstring_is_yes_if() -> None:
    """The scope docstring keeps the honest P=NP limit *and* the yes-if reframe."""
    doc = (R.__doc__ or "").lower()
    assert "p = np" in doc  # the one true hard limit is stated, not hidden
    assert "certified" in doc and "gap" in doc  # the yes-if deliverable
    assert "relaxation" in doc
    assert "yes" in doc and "if" in doc  # framed as yes-if, not no-because


def test_no_polytime_exact_solver_exported() -> None:
    """No public symbol claims a poly-time exact optimal tour."""
    for name in R.__all__:
        assert name.lower() not in _FORBIDDEN, f"{name} implies a poly-time exact TSP solver"


def test_exact_oracle_is_honestly_exponential() -> None:
    """held_karp_dp is documented as the exponential (small-n) oracle, not poly-time."""
    doc = (R.held_karp_dp.__doc__ or "").lower()
    assert "2^n" in doc or "exponential" in doc or "held-karp" in doc


def test_certificate_reports_gap_not_exactness() -> None:
    """HeldKarpCertificate exposes a gap; it has no field asserting exact optimality."""
    fields = set(HeldKarpCertificate.__dataclass_fields__)
    assert {"lower_bound", "tour_cost", "relaxation", "certified"} <= fields
    for forbidden in ("optimal", "is_exact", "exact", "is_optimal"):
        assert not hasattr(HeldKarpCertificate, forbidden)
    # the properties are gap-shaped, never a boolean "this is optimal"
    prob = RoutingProblem.from_coords(np.random.default_rng(0).random((6, 2)))
    tour, _ = decode_tour(prob.cost)
    cert = certify_tour_gap(prob, tour, kind="flow")
    assert cert.absolute_gap >= -1e-9 and cert.relative_gap >= -1e-9
    assert cert.is_sound


def test_tour_solution_cost_documented_as_upper_bound() -> None:
    from omnibias.routing import TourSolution

    assert "upper" in (TourSolution.__doc__ or "").lower()


def test_certificate_degrades_without_convex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the convex seal the bound stays valid but honestly reports certified=False."""
    monkeypatch.setitem(sys.modules, "omnibias.convex", None)  # force ImportError on seal
    prob = RoutingProblem.from_coords(np.random.default_rng(1).random((6, 2)))
    from omnibias.routing import held_karp_dp

    _, opt = held_karp_dp(prob.cost)
    tour, _ = decode_tour(prob.cost)
    cert = certify_tour_gap(prob, tour, kind="flow")
    assert cert.certified is False  # not interval-sealed
    assert cert.lower_bound <= opt + 1e-7  # still a valid (float) lower bound
    assert cert.is_sound
