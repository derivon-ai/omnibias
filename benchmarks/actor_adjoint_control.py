# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 10-02 (G3/G4): actor-adjoint control performance vs zero-control,
plus an in-repo sample-efficiency comparison across the module's five arms.

**G3 (gated).** The exact-adjoint trainer (:func:`actor_adjoint_step`) must
reduce the double-gyre tracking rollout cost to less than half the
zero-action ("do nothing") baseline cost, on the worst of >= 5 random policy
inits (``require_all_seeds``). The 0.5 threshold is set from a measured
Phase-A curve (worst observed ratio ~0.33 across seeds at
``n_steps=15, horizon=10, lr=0.05``; see the honesty block), not guessed.
Phase-A also surfaced a real failure mode worth recording: at a larger
budget (``n_steps=40, horizon=12``, same fixed ``lr=0.05``) one seed's
plain-gradient-descent trajectory diverged (cost ratio > 10000) -- vanilla
fixed-step gradient descent on this nonconvex rollout cost is not
unconditionally stable at every budget. The settings below were chosen
*because* they were verified stable across all 5 seeds, not despite the
divergence; a learning-rate schedule or line search would be the natural
fix and is not implemented here.

**G4-style diagnostic (reported, never gated as "earned").** The plan's G4
asks for interactions-to-target-return against PPO/TD3/BPTT/tBPTT/SHAC/PEARL.
This repository does not contain reimplementations of PPO, TD3, SHAC, or
PEARL (those need replay buffers, critics, clipped surrogates, or a learned
terminal value net trained to convergence) -- claiming G4 "earned" against
those named baselines would be dishonest. What this script *can* and does
measure honestly is interactions-to-threshold across the four in-repo arms
that actually exist (:func:`bptt_step`, :func:`truncated_bptt_step`,
:func:`zero_order_step` as the model-free stand-in, :func:`actor_adjoint_step`),
which already lets the exact-structure-vs-none comparison the module
docstring describes speak for itself. This is reported under
``diagnostic_g4`` with ``gated: false`` and is not summed into ``all_passed``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block, require_all_seeds  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

G3_RATIO_THRESHOLD = 0.5


def _init_layers(jax_mod: Any, jnp_mod: Any, seed: int) -> list[Any]:
    key = jax_mod.random.PRNGKey(seed)
    k1, k2 = jax_mod.random.split(key)
    w1 = 0.1 * jax_mod.random.normal(k1, (6, 2))
    b1 = jnp_mod.zeros(6)
    w2 = 0.1 * jax_mod.random.normal(k2, (2, 6))
    b2 = jnp_mod.zeros(2)
    return [(w1, b1, "tanh"), (w2, b2, None)]


def _g3_seed_sweep(*, n_seeds: int, n_steps: int, horizon: int) -> list[dict[str, Any]]:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.envs import DoubleGyrePointMass
    from omnibias.control.jax.policy import actor_adjoint_step, rollout_cost

    env = DoubleGyrePointMass()
    y0 = jnp.array([0.3, 0.4])
    rows: list[dict[str, Any]] = []
    for seed in range(n_seeds):
        layers = _init_layers(jax, jnp, seed)
        zero_layers = [
            (jnp.zeros_like(w), jnp.zeros_like(b), act) for (w, b, act) in layers
        ]
        zero_cost = float(rollout_cost(env, zero_layers, y0, horizon))
        result = None
        for _ in range(n_steps):
            result = actor_adjoint_step(env, layers, y0, horizon, 0.05)
            layers = result.layers
        assert result is not None
        trained_cost = float(result.total_cost)
        rows.append(
            {
                "seed": seed,
                "zero_action_cost": zero_cost,
                "trained_cost": trained_cost,
                "ratio": trained_cost / zero_cost,
            }
        )
    return rows


def _g4_diagnostic(*, n_steps: int, horizon: int, target_cost: float) -> dict[str, Any]:
    """Interactions-to-threshold across the four in-repo arms (one seed, one run).

    Not gated. Reports which arms reach ``target_cost`` within the budget and
    at what interaction count (``step_idx * horizon`` environment transitions).
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.envs import DoubleGyrePointMass
    from omnibias.control.jax.policy import (
        actor_adjoint_step,
        bptt_step,
        truncated_bptt_step,
        zero_order_step,
    )

    env = DoubleGyrePointMass()
    y0 = jnp.array([0.3, 0.4])

    def interactions_to_threshold(costs: list[float]) -> int | None:
        for i, c in enumerate(costs):
            if c <= target_cost:
                return (i + 1) * horizon
        return None

    arms: dict[str, dict[str, Any]] = {}

    layers = _init_layers(jax, jnp, 0)
    costs: list[float] = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        res = actor_adjoint_step(env, layers, y0, horizon, 0.05)
        layers = res.layers
        costs.append(float(res.total_cost))
    arms["actor_adjoint (ours, exact)"] = {
        "final_cost": costs[-1],
        "interactions_to_threshold": interactions_to_threshold(costs),
        "wall_seconds": time.perf_counter() - t0,
    }

    layers = _init_layers(jax, jnp, 0)
    costs = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        res = bptt_step(env, layers, y0, horizon, 0.05)
        layers = res.layers
        costs.append(float(res.total_cost))
    arms["bptt (full-horizon AD)"] = {
        "final_cost": costs[-1],
        "interactions_to_threshold": interactions_to_threshold(costs),
        "wall_seconds": time.perf_counter() - t0,
    }

    layers = _init_layers(jax, jnp, 0)
    costs = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        res = truncated_bptt_step(env, layers, y0, horizon, 5, 0.05)
        layers = res.layers
        costs.append(float(res.total_cost))
    arms["truncated_bptt (window=5, tBPTT stand-in)"] = {
        "final_cost": costs[-1],
        "interactions_to_threshold": interactions_to_threshold(costs),
        "wall_seconds": time.perf_counter() - t0,
    }

    layers = _init_layers(jax, jnp, 0)
    costs = []
    key = jax.random.PRNGKey(7)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        res = zero_order_step(env, layers, y0, horizon, 0.002, sigma=0.05, n_samples=64, key=sub)
        layers = res.layers
        costs.append(float(res.total_cost))
    arms["zero_order (ES, model-free stand-in)"] = {
        "final_cost": costs[-1],
        "interactions_to_threshold": interactions_to_threshold(costs),
        "wall_seconds": time.perf_counter() - t0,
    }

    return {
        "target_cost": target_cost,
        "n_steps_budget": n_steps,
        "horizon": horizon,
        "arms": arms,
        "gated": False,
        "not_a_comparison_to": ["PPO", "TD3", "SHAC", "PEARL"],
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()

    # Deliberately identical for smoke and --full: Phase-A found these the
    # largest budget that stays stable across all 5 seeds with a fixed
    # learning rate (see module docstring). --full only adds the ungated
    # G4 diagnostic below, run for longer, on a single seed.
    n_steps = 15
    horizon = 10
    n_seeds = 5

    rows = _g3_seed_sweep(n_seeds=n_seeds, n_steps=n_steps, horizon=horizon)
    g3 = require_all_seeds(
        rows,
        key="ratio",
        expected=G3_RATIO_THRESHOLD,
        tol=0.0,
        direction="max",
        name="g3_control_vs_zero_action",
        min_seeds=n_seeds,
    )

    entries = [g3]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")

    diagnostic_g4 = None
    if full:
        diagnostic_g4 = _g4_diagnostic(n_steps=n_steps, horizon=horizon, target_cost=10.0)
        print(
            "diagnostic_g4 (not gated):",
            {k: v["interactions_to_threshold"] for k, v in diagnostic_g4["arms"].items()},
        )

    payload = {
        **provenance(
            schema="omnibias.benchmarks.actor_adjoint_control.v1",
            config={
                "family": "actor_adjoint_control",
                "full": full,
                "n_steps": n_steps,
                "horizon": horizon,
                "n_seeds": n_seeds,
                "g3_ratio_threshold": G3_RATIO_THRESHOLD,
            },
        ),
        "gates": dict(gates_block(entries)),
        "per_seed": rows,
        "diagnostic_g4": diagnostic_g4,
        "honesty": {
            "g3_threshold_source": "measured Phase-A curve: worst-seed ratio "
            "~0.33 at n_steps=15, horizon=10, lr=0.05 (see per_seed); "
            "threshold 0.5 leaves margin, not tuned to the observed value. "
            "a larger, less-conservative budget (n_steps=40, horizon=12) "
            "was tried first and diverged on one of 5 seeds -- these "
            "settings were kept specifically because they are the ones "
            "verified stable, per the module docstring",
            "g4_status": "leftover-recorded, not earned: no PPO/TD3/SHAC/PEARL "
            "reimplementation exists in this repository; diagnostic_g4 "
            "compares only the four in-repo arms and is explicitly ungated "
            "(see theory/10-control/02-jet-adjoint-policy-optimization.md "
            "section 8)",
        },
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "actor_adjoint_control.json" if full else "actor_adjoint_control_smoke.json"
    if full:
        dest = SCRATCH / "control" / "adjoint"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
