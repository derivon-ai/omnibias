# omnibias.control adjoint (jet-adjoint policy optimization)

A discrete adjoint (costate) policy-gradient stack on top of a declared
`DifferentiableEnvironment`, plus a certified layer that derives (rather than
hand-picks) a training-window length and bounds what a short-horizon
shortcut costs (theory
[10-02](https://github.com/derivon-ai/omnibias/blob/main/theory/10-control/02-jet-adjoint-policy-optimization.md) /
[10-03](https://github.com/derivon-ai/omnibias/blob/main/theory/10-control/03-certified-horizon-gradient-bias.md) /
[10-04](https://github.com/derivon-ai/omnibias/blob/main/theory/10-control/04-certified-contact.md)).
This is a **submodule of the existing `omnibias-control` package** -- see
[omnibias-control](control.md) for the pre-existing CBF-QP safety filter and
model-relative recoverable-set certificate, which this stack composes with
rather than replaces.

## Guarantee level

| Piece | Level | Acceptance domain |
| --- | --- | --- |
| `dpi/dy` via the directional jet tower | by construction (founding bias collapse, `delta -> 0`) | any policy MLP `mlp_jet`/`layer_jet`/`compose_jet` support |
| `actor_adjoint_gradient` vs BPTT | empirical, exact | matches float64 round-off on a closed-loop rollout (gate G1) |
| Analytic LQR reference | by construction | `adjoint_recursion` reproduces `lambda_k = P_k y_k` from `discrete_riccati_sweep` to `1.78e-15` (gate G2) |
| `certified_horizon` | sound enclosure, conditional | correct only within the caller-declared per-step Jacobian ball (gate G5) |
| `truncation_bias_bound` | sound enclosure, conditional | correct only given the caller's terminal-error model (gate G6) |
| CBF-QP filter | temperature collapse (`beta -> inf`) | unchanged from [omnibias-control](control.md); a different founding limit than the adjoint above -- do not conflate |
| Contact smoothing | exact tower + one-sided bias bound | 1-D point-mass only; 2-D Coulomb is a leftover |

## Acceptance gates

See [`theory/10-control/02-jet-adjoint-policy-optimization.md`](https://github.com/derivon-ai/omnibias/blob/main/theory/10-control/02-jet-adjoint-policy-optimization.md)
section 8 for the full statement. G1/G2/G3/G5/G6/G9 are earned in this repo;
G4 (interactions-to-target vs PPO/TD3/SHAC/PEARL) and G8 (cost parity vs SHAC
on a robotics engine) are **not** -- no reimplementation of those named
baselines, and no Brax/MuJoCo-MJX installation, exists in this repository.
`benchmarks/actor_adjoint_control.py` reports an ungated in-repo diagnostic
across the arms that do exist instead of fabricating either comparison.

## Recursion algebra (pure Python)

::: omnibias.core.adjoint
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - discrete_riccati_sweep
        - adjoint_recursion
        - policy_gradient
        - closed_loop_jacobian
        - n_step_adjoint_bootstrap
        - td_lambda_mix
        - worked_example
        - adjoint_skill

## Environment seam

::: omnibias.control.ocp
    options:
      show_root_heading: false
      heading_level: 3

## Backend adjoint (JAX)

::: omnibias.control.jax.adjoint
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - policy_forward
        - policy_jacobian_dy
        - actor_adjoint_gradient

## Policy trainers (JAX)

::: omnibias.control.jax.policy
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - bptt_step
        - truncated_bptt_step
        - zero_order_step
        - actor_adjoint_step
        - actor_adjoint_jet_step
        - PSDTerminalHead
        - fit_terminal_head

## Environments (JAX)

::: omnibias.control.jax.envs
    options:
      show_root_heading: false
      heading_level: 3

## Backend twins (torch)

`omnibias.control.torch.{adjoint,policy,envs}` mirror the JAX modules above
call-for-call (`torch.Tensor` in place of `jax.Array`); both twins call the
identical `omnibias.core.adjoint` function objects, so bit-parity is
structural (gate G9), not merely measured to a tolerance.

## Certified horizon

::: omnibias.control.horizon
    options:
      show_root_heading: false
      heading_level: 3

## Gradient-bias enclosure

::: omnibias.control.certified.gradient_bias
    options:
      show_root_heading: false
      heading_level: 3

## Proof-carrying bundle

::: omnibias.control.bundle
    options:
      show_root_heading: false
      heading_level: 3

## Contact smoothing

::: omnibias.core.contact_smoothing
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.control.jax.contact
    options:
      show_root_heading: false
      heading_level: 3

## Robotics via a wrapped engine

::: omnibias.control.robotics
    options:
      show_root_heading: false
      heading_level: 3

Requires `brax` or `mujoco-mjx` to be installed; the adapters raise
`ImportError` rather than degrading silently when the engine is missing.
omnibias contributes the optimizer and certificates, not the physics engine.

## Example and benchmarks

Cookbook: [certified-policy-gradient](../cookbook/certified-policy-gradient.md).
Benchmarks: [`benchmarks/adjoint_exactness.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/adjoint_exactness.py)
(G1/G2/G9),
[`benchmarks/actor_adjoint_control.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/actor_adjoint_control.py)
(G3, plus the ungated G4 diagnostic),
[`benchmarks/certified_horizon.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/certified_horizon.py)
(G5),
[`benchmarks/gradient_bias_enclosure.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/gradient_bias_enclosure.py)
(G6). Smoke artifacts:
[`docs/benchmarks/adjoint_exactness_smoke.json`](../benchmarks/adjoint_exactness_smoke.json),
[`docs/benchmarks/actor_adjoint_control_smoke.json`](../benchmarks/actor_adjoint_control_smoke.json),
[`docs/benchmarks/certified_horizon_smoke.json`](../benchmarks/certified_horizon_smoke.json),
[`docs/benchmarks/gradient_bias_enclosure_smoke.json`](../benchmarks/gradient_bias_enclosure_smoke.json).

Status: Alpha (`0.1.0a1`), submodule of `omnibias-control`.
