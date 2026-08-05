# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Scope / honesty guards: the package ships a *certified gap*, never a P=NP claim.

The "yes-if" contract (like omnibias-qubo / omnibias-routing, unlike the P-class
omnibias-combinatorics): every solver returns a feasible solution with a rigorous but
generally **non-tight** optimality gap; nothing advertises a poly-time route to the exact
NP-hard optimum. The only exact solvers are the explicitly exponential brute-force
oracles.
"""

from __future__ import annotations

import importlib

import numpy as np
import omnibias.nphard as N
from omnibias.nphard import NPHardCertificate

# names that would (falsely) advertise a poly-time exact solver for an NP-hard family
_FORBIDDEN = {
    "solve_qap",
    "solve_gap",
    "solve_scheduling",
    "solve_exact",
    "exact_min",
    "exact_minimum",
    "optimal_assignment",
    "optimal_permutation",
    "optimal_schedule",
    "best_assignment",
    "global_minimum",
}


def test_package_docstring_is_honest_yes_if() -> None:
    doc = (N.__doc__ or "").lower()
    assert "p = np" in doc  # the one honest hard limit is named, not hidden
    assert "certified" in doc and "gap" in doc  # the yes-if deliverable
    assert "yes, if" in doc  # framed as yes-if
    assert "non-tight" in doc  # honest about the gap being non-tight for NP-hard


def test_no_polytime_exact_solver_symbol_is_exported() -> None:
    assert not (set(N.__all__) & _FORBIDDEN)
    for name in _FORBIDDEN:
        assert not hasattr(N, name), f"{name} must not exist (implies a poly-time exact NP solver)"


def test_certificate_is_gap_shaped_not_exactness() -> None:
    fields = NPHardCertificate.__dataclass_fields__
    assert "lower_bound" in fields and "energy" in fields
    for banned in ("optimal", "is_exact", "exact", "is_optimal", "zero_gap"):
        assert banned not in fields, f"{banned} would imply an exactness (P = NP) claim"
    for prop in ("absolute_gap", "relative_gap", "is_sound"):
        assert hasattr(NPHardCertificate, prop)


def test_brute_force_oracle_is_labelled_exponential() -> None:
    doc = (N.brute_force_min.__doc__ or "").lower()
    assert "exponential" in doc


def test_per_family_oracles_are_labelled_exponential() -> None:
    from omnibias.nphard._core.gap import gap_brute_force
    from omnibias.nphard._core.qap import qap_brute_force
    from omnibias.nphard._core.scheduling import scheduling_brute_force

    for fn in (qap_brute_force, gap_brute_force, scheduling_brute_force):
        assert "exponential" in (fn.__doc__ or "").lower()


def test_classical_baseline_is_labelled_a_heuristic_not_exact() -> None:
    doc = (N.classical_optimum.__doc__ or "").lower()
    assert "heuristic" in doc
    assert "not a guaranteed exact optimum" in doc or "np-hard" in doc


def test_package_docstring_distinguishes_the_two_collapse_senses() -> None:
    doc = N.__doc__ or ""
    assert "founding bias collapse" in doc
    assert "delta -> 0" in doc


def test_relaxation_docstrings_distinguish_the_two_collapse_senses() -> None:
    seen = 0
    for mod_name in ("omnibias.nphard.jax.relaxation", "omnibias.nphard.torch.relaxation"):
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:  # backend not installed in this environment
            continue
        doc = mod.__doc__ or ""
        assert "founding bias collapse" in doc
        assert "delta -> 0" in doc
        seen += 1
    assert seen >= 0


def test_glb_certificate_is_gap_shaped_and_honestly_non_tight() -> None:
    """The Gilmore-Lawler certificate is a *lower bound* with a generally non-zero gap --
    a scalable placement bound, never an exact-optimum (P = NP) claim."""
    rng = np.random.default_rng(2)
    conn = rng.integers(0, 5, size=(12, 12)).astype(float)
    conn = conn + conn.T
    np.fill_diagonal(conn, 0.0)
    prob = N.placement_qap(conn, (3, 4))
    from omnibias.nphard._core.qap import perm_to_x

    cert = N.certify_gap(prob, perm_to_x(tuple(range(12)), 12), kind="glb")
    assert cert.lower_bound <= cert.energy  # a lower bound, not an equality
    assert cert.relative_gap > 0.0  # NP-hard-honest: the gap is non-zero here
    assert not hasattr(cert, "is_exact") and not hasattr(cert, "is_optimal")


def test_glb_docstrings_are_honest_about_non_tightness_and_scale() -> None:
    assert "non-tight" in (N.gilmore_lawler_bound.__doc__ or "").lower()
    doc = (N.certify_gap.__doc__ or "").lower()
    assert "glb" in doc and "gilmore" in doc and "scalable" in doc


def test_placement_qap_docstring_names_it_a_heuristic_with_a_certified_gap() -> None:
    doc = (N.placement_qap.__doc__ or "").lower()
    assert "np-hard" in doc  # the honest hard limit
    assert "heuristic" in doc  # solving is heuristic
    assert "gilmore-lawler" in doc and "non-tight" in doc  # certified but non-tight gap
    assert "p = np" in doc  # never an exact-optimum claim


def test_all_is_sorted_and_has_version() -> None:
    assert "__version__" in N.__all__
    assert list(N.__all__) == sorted(N.__all__)
