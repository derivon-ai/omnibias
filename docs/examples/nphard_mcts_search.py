# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""A Go-like MCTS search track with a differentiable-relaxation prior -- omnibias-nphard.

Run:

    pip install "omnibias-nphard[jax,convex]"
    python docs/examples/nphard_mcts_search.py

The "AlphaGo point" made concrete on an NP-hard family: *tackle it well with search + a
learned prior, and certify the gap* -- never *solve it exactly*. We phrase parallel-machine
scheduling as a **construction MDP** (state = jobs placed so far; action = the next job's
machine; reward = -energy) and run a small, self-contained **UCT / PUCT** Monte-Carlo tree
search. The differentiable annealed relaxation heatmap becomes the AlphaZero-style action
**prior**. The demo is deterministic and CPU-tiny:

1. **Matches the oracle.** On a tiny instance the relaxation-guided search reaches the exact
   brute-force optimum.
2. **Prior beats uniform.** On a larger instance the differentiable prior reaches the
   optimum where an otherwise-identical uniform-prior search does not.
3. **Certified gap.** The heuristic MCTS solution is still handed to ``certify_gap`` for a
   sound (honestly non-tight) optimality gap ``lower <= optimum <= energy`` -- a heuristic
   search, never an optimality guarantee.

Terminology: the relaxation's ``sigmoid(beta z)``, ``beta -> inf`` is the feasibility /
temperature sense of "collapse", distinct from the **founding bias collapse** (the
multi-bias ``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from omnibias.nphard import brute_force_min, certify_gap, schedule  # noqa: E402
from omnibias.nphard.jax import relax as relax_jax  # noqa: E402
from omnibias.nphard.search import (  # noqa: E402
    mcts_search,
    mdp_for,
    random_rollout,
    relaxation_prior,
    uniform_prior,
)


def _guided(problem: object, shape: tuple[int, int], iterations: int, seed: int) -> object:
    mdp = mdp_for(problem)
    heat = np.asarray(relax_jax(problem)).reshape(shape)
    return mcts_search(
        mdp,
        prior_fn=relaxation_prior(heat, temperature=1.0),
        rollout_fn=random_rollout(mdp, np.random.default_rng(seed)),
        iterations=iterations,
        seed=seed,
    )


def _uniform(problem: object, iterations: int, seed: int) -> object:
    mdp = mdp_for(problem)
    return mcts_search(
        mdp,
        prior_fn=uniform_prior,
        rollout_fn=random_rollout(mdp, np.random.default_rng(seed)),
        iterations=iterations,
        seed=seed,
    )


def oracle_match_demo() -> None:
    print("=== 1. relaxation-guided MCTS matches the brute-force oracle (tiny instance) ===")
    rng = np.random.default_rng(0)
    prob = schedule(rng.integers(1, 10, size=6).astype(float), 2)
    _, e_opt = brute_force_min(prob)
    result = _guided(prob, (6, 2), iterations=80, seed=0)
    print(f"  J=6 machines=2: guided MCTS load-variance {result.energy:.1f}   optimum {e_opt:.1f}")
    assert abs(result.energy - e_opt) < 1e-6, "guided search should reach the optimum here"
    print("  -> guided search reaches the exact optimum.\n")


def prior_beats_uniform_demo() -> None:
    print("=== 2. the differentiable prior beats a uniform-prior search ===")
    rng = np.random.default_rng(3)
    prob = schedule(rng.integers(1, 12, size=8).astype(float), 3)
    _, e_opt = brute_force_min(prob)
    guided = _guided(prob, (8, 3), iterations=80, seed=3)
    uniform = _uniform(prob, iterations=80, seed=3)
    print(f"  J=8 machines=3:  optimum {e_opt:.1f}")
    print(f"    guided  (relaxation prior)  {guided.energy:.1f}")
    print(f"    uniform (flat prior)        {uniform.energy:.1f}")
    assert guided.energy <= uniform.energy, "the differentiable prior should help (or tie)"
    assert abs(guided.energy - e_opt) < 1e-6, "guided reaches the optimum"
    assert uniform.energy > e_opt + 1e-6, "the uniform-prior search does not, here"

    cert = certify_gap(prob, guided.assignment, kind="spectral")
    print("  certify the guided solution (honestly non-tight, NP-hard):")
    print(f"    lower_bound {cert.lower_bound:.1f} <= optimum {e_opt:.1f} <= energy {cert.energy:.1f}")
    assert cert.lower_bound <= e_opt + 1e-6 and cert.is_sound
    print("  -> the guided prior wins; the found solution carries a sound optimality gap.\n")


def main() -> None:
    oracle_match_demo()
    prior_beats_uniform_demo()
    print("OK: MCTS with a differentiable prior matches the oracle, beats uniform, certifies a gap.")


if __name__ == "__main__":
    main()
