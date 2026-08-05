# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Soundness regressions for the submodular gap certificate.

The load-bearing claim of a gap certificate is the sandwich ``f(S) <= OPT <= U``.
Checking only ``f(S) <= U`` is *internal consistency*, and it stays true when both
sides sit far below ``OPT`` -- so every assertion here brackets the brute-force
optimum, not just the decoded value.

``GraphCut`` is the canonical non-monotone family (``f(empty) = f(V) = 0``), which
is exactly where a bound derived under a monotonicity assumption breaks.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    Coverage,
    GraphCut,
    SubmodularProblem,
    UniformMatroid,
    brute_force_max,
    certify_submodular_gap,
    marginal_upper_bound,
    modular_upper_bound,
    nonmonotone_upper_bound,
)


def _graph_cut(seed: int) -> tuple[GraphCut, UniformMatroid, float]:
    """A random symmetric zero-diagonal cut instance plus its exact optimum."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 8))
    w = rng.random((n, n))
    w = (w + w.T) / 2.0
    np.fill_diagonal(w, 0.0)
    function = GraphCut(w)
    matroid = UniformMatroid(n, n)
    _, opt = brute_force_max(function, matroid)
    return function, matroid, float(opt)


def test_graph_cut_is_declared_non_monotone() -> None:
    """The certificate layer needs to *know* monotonicity to pick a valid bound."""
    function, _, _ = _graph_cut(0)
    assert not function.is_monotone
    assert Coverage(np.array([[1.0, 0.0], [0.0, 1.0]])).is_monotone


def test_certified_upper_bound_dominates_opt_on_non_monotone() -> None:
    """``certify_submodular_gap`` must bound the true optimum, not merely the decode."""
    for seed in range(200):
        function, matroid, opt = _graph_cut(seed)
        problem = SubmodularProblem(function, matroid)
        selection = np.ones(function.n)
        cert = certify_submodular_gap(problem, selection)
        assert cert.upper_bound >= opt - 1e-9, (
            f"seed={seed}: certified upper_bound={cert.upper_bound!r} is BELOW the true "
            f"optimum {opt!r} -- the certificate claims a bound that does not hold"
        )


def test_marginal_upper_bound_refuses_non_monotone_input() -> None:
    """The marginal bound is only valid for monotone ``f``; it must say so, loudly."""
    function, matroid, _ = _graph_cut(0)
    with pytest.raises(ValueError, match="monotone"):
        marginal_upper_bound(function, matroid, np.ones(function.n))


def test_monotone_safe_bounds_stay_sound_without_monotonicity() -> None:
    """``modular_upper_bound`` / ``nonmonotone_upper_bound` need no monotonicity."""
    for seed in range(200):
        function, matroid, opt = _graph_cut(seed)
        assert modular_upper_bound(function, matroid) >= opt - 1e-9, f"seed={seed}"
        assert nonmonotone_upper_bound(function) >= opt - 1e-9, f"seed={seed}"


def test_certificate_soundness_is_not_mere_internal_consistency() -> None:
    """The self-check must not report success when the bound misses ``OPT``."""
    for seed in range(50):
        function, matroid, opt = _graph_cut(seed)
        problem = SubmodularProblem(function, matroid)
        cert = certify_submodular_gap(problem, np.ones(function.n))
        if cert.internal_consistent:
            assert cert.value <= opt + 1e-9 and opt <= cert.upper_bound + 1e-9, (
                f"seed={seed}: internal_consistent=True but the sandwich "
                f"{cert.value!r} <= {opt!r} <= {cert.upper_bound!r} does not hold"
            )


def test_the_weak_self_check_is_no_longer_named_is_sound() -> None:
    """The old name invited ``value <= upper_bound`` to be read as ``OPT <= upper_bound``."""
    function, matroid, _ = _graph_cut(0)
    cert = certify_submodular_gap(SubmodularProblem(function, matroid), np.ones(function.n))
    assert not hasattr(cert, "is_sound")
    assert isinstance(cert.internal_consistent, bool)


def test_no_apriori_ratio_is_claimed_for_a_non_monotone_problem() -> None:
    """``1 - 1/e`` is a monotone-only theorem; claiming it elsewhere is unfounded."""
    function, matroid, _ = _graph_cut(0)
    cert = certify_submodular_gap(SubmodularProblem(function, matroid), np.ones(function.n))
    assert cert.approx_ratio == 0.0
    assert cert.method == "modular_nonmonotone"


def test_monotone_certificate_still_sound() -> None:
    """Regression guard: the monotone path must keep working after the fix."""
    for seed in range(100):
        rng = np.random.default_rng(seed + 9_000)
        n = int(rng.integers(4, 7))
        m = int(rng.integers(3, 8))
        coverage = Coverage((rng.random((m, n)) < 0.5).astype(float))
        matroid = UniformMatroid(n, n)
        _, opt = brute_force_max(coverage, matroid)
        problem = SubmodularProblem(coverage, matroid)
        cert = certify_submodular_gap(problem, np.ones(n))
        assert cert.upper_bound >= float(opt) - 1e-9, f"seed={seed}"
