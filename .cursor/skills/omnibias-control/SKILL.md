---
name: omnibias-control
description: Train policies through a declared dynamical system with exact jet-adjoint policy gradients, a certified truncation horizon, a sound policy-gradient-bias enclosure, a CBF-QP safety filter, and certified contact smoothing. Use when inventing a proof-carrying controller, a new DifferentiableEnvironment, or when the user mentions optimal control, policy gradients, CBFs, or safe RL.
---

# Exact-adjoint control with a certified layer

`omnibias.control` trains a policy through a declared
`DifferentiableEnvironment` with an **exact** adjoint (closed-form `dpi/dy`
from the jet tower), then certifies the two numbers every short-horizon
trainer otherwise guesses: how long the window needs to be, and how much bias
that window costs.

## Why nested AD fails

Truncated BPTT and finite-difference policy gradients pick a horizon by hand
and eat an unknown bias. Nested AD through a long rollout is cubic in horizon
and still cannot enclose `||grad J - grad J_h||`. Generic CBF-QPs have no
recoverable-set certificate. Contact smoothing without `sigma^(n)` is a leaky
ReLU. That combination — exact adjoint + certified horizon + sound bias box —
is the workload nested AD / PPO stacks cannot produce.

## What only this tower unlocks

`actor_adjoint_step` uses closed-form `dpi/dy`. `certified_horizon` reuses
`omnibias.dynamics.spectral_radius_bound` on the enclosed closed-loop
monodromy. `truncation_bias_bound` reuses `omnibias.verify.lipschitz_bound`.
Contact smoothing rides the exact tower plus a one-sided hardening-bias
enclosure. Two limits coexist and both are available: CBF-QP sharpens with
temperature collapse (`beta -> inf`); the adjoint's `dpi/dy` is exact at every
temperature (founding `delta -> 0`).

## Use

| You want | Import from | Key entry points |
| --- | --- | --- |
| Declare an environment | `omnibias.control.ocp` | `DifferentiableEnvironment`, `OCPSpec`, `rollout_generic` |
| Built-in environments | `omnibias.control.{jax,torch}.envs` | `DoubleGyrePointMass`, `AdvectionDiffusionGrid` |
| Train a policy | `omnibias.control.{jax,torch}.policy` | `actor_adjoint_step`, `actor_adjoint_jet_step`, `bptt_step`, `truncated_bptt_step`, `zero_order_step` |
| Certify the training window | `omnibias.control.horizon` | `certified_horizon` |
| Bound truncation cost | `omnibias.control.certified.gradient_bias` | `truncation_bias_bound`, `terminal_adjoint_error_bound` |
| Proof-carrying object | `omnibias.control.bundle` | `build_bundle`, `bundle_status`, `summary` |
| Safety filter | `omnibias.control.{jax,torch}` | CBF-QP `safe_rollout`, `certify_recoverable` |
| Contact dynamics | `omnibias.control.{jax,torch}.contact` | `ContactPointMass1D`, `contact_certificate` |
| Robotics via wrapped engine | `omnibias.control.robotics` | Brax / MuJoCo-MJX adapters (`ImportError` without the engine) |

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.control.jax.envs import DoubleGyrePointMass
from omnibias.control.jax.policy import actor_adjoint_step

env = DoubleGyrePointMass()
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
layers = [
    (0.1 * jax.random.normal(k1, (6, 2)), jnp.zeros(6), "tanh"),
    (0.1 * jax.random.normal(k2, (2, 6)), jnp.zeros(2), None),
]
y0 = jnp.array([0.3, 0.4])
for _ in range(15):
    result = actor_adjoint_step(env, layers, y0, horizon=10, lr=0.05)
    layers = result.layers
```

`omnibias.control.torch.policy.actor_adjoint_step` is the bit-identical twin.
`certified_horizon` is sound inside the declared `radius` around each `M_k`.
Get `terminal_error_bound` from `terminal_adjoint_error_bound`. Robotics
adapters raise if the engine is missing.

Cookbook: `docs/cookbook/certified-policy-gradient.md`.

## Extend

- Source: `packages/omnibias-control`. Pure-Python adjoint:
  `omnibias.core.adjoint`.
- Tests: `python -m pytest packages/omnibias-control/tests -q`.
- Research doctrine: `omnibias-control-research`. Compose with
  `omnibias-dynamics`, `omnibias-verify`, `omnibias-curvature`.
- No new package: Group 10 lands in existing modules (`theory/10-control/01-ledger.md`).

## Next invention

A MIMO plant whose certified horizon and truncation-bias enclosure are
computed from the closed-loop monodromy in one call, then consumed by
`actor_adjoint_jet_step` without a guessed window.

## Bakeoffs

`docs/benchmarks/actor_adjoint_control_smoke.json`,
`adjoint_exactness_smoke.json`, `certified_horizon_smoke.json`,
`gradient_bias_enclosure_smoke.json`.

## Further references

- API: `docs/api/adjoint_control.md`
- Theory: `theory/10-control/`
