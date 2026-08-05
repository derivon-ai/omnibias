# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Residual-adaptive refinement (RAR): selection rules and driver wiring.

The selection half is pure numpy and is tested directly against hand-built score
vectors, where "did it pick the right points" is decidable. The driver half is
tested for the property that actually matters to a caller: the interior set grows
by the specified schedule, stops at ``max_points``, and every optimiser path --
including the two whose iteration loop is not a plain ``for`` (L-BFGS owns its
inner loop; ``gauss_newton`` is functional over a flat vector) -- refines.

The accuracy claim (RAR beats uniform sampling on a sharp front at equal point
budget) is a separate, heavier gate in ``test_refinement_accuracy.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
from omnibias.pinn.solver._core.sampling import (  # noqa: E402
    candidate_points,
    select_refinement_points,
)
from omnibias.pinn.solver.torch.assemble import (  # noqa: E402
    interior_residual,
    to_tensor,
)

SPEC = pde.CollocationSpec(n_interior=6, n_boundary=6)
BUDGET = {"hidden": 12, "seed": 0, "collocation": SPEC, "adam_iters": 6}


def _domain() -> pde.Domain:
    return pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))


def _poisson() -> pde.System:
    def source(c):
        xp = pde.array_namespace(c)
        return -2.0 * math.pi**2 * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    return pde.poisson(_domain(), source=source, boundary=0.0)


def _spec(**kwargs: object) -> pde.RefinementSpec:
    # ``strategy`` is pinned here rather than inherited: most of these tests assert
    # *which* points were chosen, which is only decidable for the greedy rule.
    base = {
        "every": 2,
        "n_candidates": 32,
        "n_add": 4,
        "max_points": 512,
        "strategy": "greedy",
    }
    return pde.RefinementSpec(**{**base, **kwargs})  # type: ignore[arg-type]


def test_defaults_are_the_measured_configuration() -> None:
    """The benchmark in docs/benchmarks.md was run at these defaults."""
    default = pde.RefinementSpec()
    assert default.strategy == "proportional"
    assert default.power == 2.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"strategy": "adaptive"}, "strategy must be"),
        ({"every": 0}, "every must be >= 1"),
        ({"n_candidates": 0}, "n_candidates must be >= 1"),
        ({"n_add": 0}, "n_add must be >= 1"),
        ({"power": 0.0}, "power must be > 0"),
        ({"max_points": 0}, "max_points must be >= 1"),
    ],
)
def test_refinement_spec_validates(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(**kwargs)


def test_candidate_points_are_in_bounds_and_differ_per_round() -> None:
    ref = _spec()
    first = candidate_points(_domain(), SPEC, ref, round_index=1)
    second = candidate_points(_domain(), SPEC, ref, round_index=2)
    assert first.shape == (ref.n_candidates, 2)
    assert np.all((first >= 0.0) & (first <= 1.0))
    # Each round reseeds with seed + round_index, so rounds never repeat points.
    assert not np.allclose(first, second)
    assert np.allclose(first, candidate_points(_domain(), SPEC, ref, round_index=1))


def test_greedy_keeps_exactly_the_largest_scores() -> None:
    ref = _spec(n_add=3)
    candidates = np.arange(10, dtype=float).reshape(10, 1)
    scores = np.array([0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0])
    keep = select_refinement_points(candidates, scores, ref, n_existing=0)
    assert sorted(keep[:, 0].tolist()) == [1.0, 3.0, 5.0]


def test_greedy_ranks_by_magnitude_not_sign() -> None:
    ref = _spec(n_add=2)
    candidates = np.arange(4, dtype=float).reshape(4, 1)
    scores = np.array([-5.0, 0.1, 4.0, -0.2])
    keep = select_refinement_points(candidates, scores, ref, n_existing=0)
    assert sorted(keep[:, 0].tolist()) == [0.0, 2.0]


def test_non_finite_scores_are_never_preferred() -> None:
    ref = _spec(n_add=2)
    candidates = np.arange(4, dtype=float).reshape(4, 1)
    scores = np.array([np.nan, 3.0, np.inf, 2.0])
    keep = select_refinement_points(candidates, scores, ref, n_existing=0)
    assert sorted(keep[:, 0].tolist()) == [1.0, 3.0]


def test_max_points_caps_the_addition_and_then_stops_it() -> None:
    ref = _spec(n_add=10, max_points=100)
    candidates = np.random.default_rng(0).uniform(size=(32, 2))
    scores = np.linspace(0.0, 1.0, 32)
    partial = select_refinement_points(candidates, scores, ref, n_existing=95)
    assert partial.shape == (5, 2)
    full = select_refinement_points(candidates, scores, ref, n_existing=100)
    assert full.shape == (0, 2)
    assert full.dtype == candidates.dtype


def test_n_add_is_clamped_to_the_candidate_count() -> None:
    ref = _spec(n_add=50)
    candidates = np.random.default_rng(1).uniform(size=(7, 2))
    keep = select_refinement_points(
        candidates, np.ones(7), ref, n_existing=0
    )
    assert keep.shape == (7, 2)


def test_proportional_favours_high_scores_but_still_explores() -> None:
    ref = _spec(n_add=4, strategy="proportional", power=2.0)
    candidates = np.arange(40, dtype=float).reshape(40, 1)
    scores = np.zeros(40)
    scores[30:] = 1.0  # only the last ten carry residual
    picks = {
        int(v)
        for r in range(12)
        for v in select_refinement_points(
            candidates, scores, ref, n_existing=0, round_index=r
        )[:, 0]
    }
    assert picks and picks <= set(range(30, 40))
    assert len(picks) > 4, "proportional sampling should spread over the support"


def test_proportional_falls_back_to_uniform_on_a_dead_residual() -> None:
    ref = _spec(n_add=3, strategy="proportional")
    candidates = np.arange(20, dtype=float).reshape(20, 1)
    keep = select_refinement_points(candidates, np.zeros(20), ref, n_existing=0)
    assert keep.shape == (3, 1)
    assert len(set(keep[:, 0].tolist())) == 3, "sampling is without replacement"


def test_score_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="4 candidates but 3 scores"):
        select_refinement_points(
            np.zeros((4, 2)), np.zeros(3), _spec(), n_existing=0
        )


@pytest.mark.parametrize(
    "optimizer",
    ["adam", "lbfgs", "cubic_newton", "cubic_gauss_newton", "gauss_newton"],
)
def test_every_optimizer_path_refines(optimizer: str) -> None:
    ref = _spec(every=3, n_candidates=24, n_add=5)
    sol = pt.solve_optimize(
        _poisson(), optimizer=optimizer, iters=9, refinement=ref, **BUDGET
    )
    d = sol.diagnostics
    assert d["n_interior_uniform"] == SPEC.n_interior**2
    # iters=9, every=3 -> refinement rounds at 3 and 6 (never at iteration 0, and
    # never during the Adam warmup, whose residual carries no signal yet).
    assert d["n_refinement_rounds"] == 2
    assert d["n_interior_final"] == d["n_interior_uniform"] + 2 * ref.n_add
    assert math.isfinite(sol.residual_norm)


def test_refinement_is_off_by_default() -> None:
    sol = pt.solve_optimize(_poisson(), iters=3, **BUDGET)
    assert "n_interior_final" not in sol.diagnostics


def test_rounds_never_redraw_the_same_candidates() -> None:
    """The round index is monotone, so no two rounds add identical points.

    A per-phase iteration counter would reseed the draw and silently duplicate an
    earlier round's points, wasting the ``max_points`` budget on copies.
    """
    ref = _spec(every=2, n_candidates=12, n_add=6)
    sol = pt.solve_optimize(
        _poisson(), optimizer="adam", iters=8, refinement=ref, **BUDGET
    )
    rounds = sol.diagnostics["n_refinement_rounds"]
    added = sol.diagnostics["n_interior_final"] - sol.diagnostics["n_interior_uniform"]
    assert rounds == 3
    assert added == rounds * ref.n_add
    # Replay the draws the driver made: distinct round indices, distinct points.
    drawn = np.concatenate(
        [candidate_points(_domain(), SPEC, ref, round_index=r) for r in range(1, rounds + 1)]
    )
    assert np.unique(drawn, axis=0).shape[0] == drawn.shape[0]


def test_driver_respects_max_points() -> None:
    cap = SPEC.n_interior**2 + 7
    ref = _spec(every=1, n_candidates=16, n_add=5, max_points=cap)
    sol = pt.solve_optimize(
        _poisson(), optimizer="adam", iters=8, refinement=ref, **BUDGET
    )
    assert sol.diagnostics["n_interior_final"] == cap


def test_refined_points_land_where_the_residual_is_largest() -> None:
    """The added points must track the residual, not just be extra samples."""
    system = _poisson()
    field = pt.build_field(system, hidden=16, seed=0)
    ref = _spec(every=1, n_candidates=400, n_add=40, strategy="greedy")
    candidates = candidate_points(_domain(), SPEC, ref, round_index=1)
    with torch.no_grad():
        coords = to_tensor(candidates, field)
        rows = interior_residual(field, system, coords)
        scores = rows.reshape(-1, candidates.shape[0]).abs().amax(dim=0).numpy()
    keep = select_refinement_points(candidates, scores, ref, n_existing=0)
    with torch.no_grad():
        kept_rows = interior_residual(field, system, to_tensor(keep, field))
        kept = kept_rows.reshape(-1, keep.shape[0]).abs().amax(dim=0).numpy()
    assert float(kept.min()) >= float(np.median(scores))
    assert float(kept.mean()) > float(scores.mean())


def test_refinement_composes_with_grad_norm_balancing() -> None:
    sol = pt.solve_optimize(
        _poisson(),
        optimizer="cubic_newton",
        iters=6,
        refinement=_spec(every=2, n_candidates=16, n_add=3),
        loss_balancing="grad_norm",
        balance_every=2,
        condition_weight=1.0,
        **BUDGET,
    )
    assert sol.diagnostics["loss_balancing"] == "grad_norm"
    assert sol.diagnostics["n_interior_final"] > sol.diagnostics["n_interior_uniform"]
