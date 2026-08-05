# omnibias-control

Differentiable **and** certified safe control on the omnibias stack: a batched,
per-sample control-barrier-function (CBF-QP) **safety filter**, autodiff
Lie-derivative CBF-row builders for generic control-affine and learned-Lagrangian
dynamics, a differentiable safe rollout, and a rigorous model-relative
recoverable-set certificate.

The filter solves the state-dependent projection

\[
a^\star(s) \;=\; \arg\min_a \tfrac12\lVert a - a_{\text{nom}}\rVert^2
\quad\text{s.t.}\quad G(s)\,a \le h(s),
\]

whose rows are the CBF inequality (*brake before you hit the obstacle*) plus the
actuator box `||a||_inf <= a_max`. It is the temperature-collapse projection layer of
`omnibias-convex` specialised to per-sample constraints, so it is batched, `jit`-able,
and **differentiable** -- a policy can be trained *through* it. The recoverable-set
certificate (interval branch-and-bound from `omnibias-verify`) proves the region of
states from which an actuator-admissible action keeps the barrier non-negative; on
that region forward invariance is a theorem, and the same register gives a certified
safe speed limit.

A worked, runnable walkthrough is in the
[certified-safe-control cookbook](../cookbook/certified-safe-control.md) and
[`docs/examples/control_learned_lagrangian_cbf.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/control_learned_lagrangian_cbf.py).

## Specs & containers

::: omnibias.control.problem
    options:
      show_root_heading: false
      heading_level: 3

## Safety filter (JAX)

::: omnibias.control.jax.filter
    options:
      show_root_heading: false
      heading_level: 3

## CBF-row builders (JAX)

::: omnibias.control.jax.builders
    options:
      show_root_heading: false
      heading_level: 3

## Safe rollout (JAX)

::: omnibias.control.jax.rollout
    options:
      show_root_heading: false
      heading_level: 3

## Backend twins (torch)

Bit-identical PyTorch twins of the filter, builders, and rollout (parity `~1e-12`).

::: omnibias.control.torch.filter
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.control.torch.builders
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.control.torch.rollout
    options:
      show_root_heading: false
      heading_level: 3

## Recoverable-set certificate

::: omnibias.control.certify
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).
