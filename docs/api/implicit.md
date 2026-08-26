# Implicit / DEQ Newton (08-08)

An equilibrium `u = sigma(W u + x)` is solved to a named residual
(Newton or Banach iteration) and differentiated by the
implicit-function theorem with exact `sigma'`. One linear solve
against `I - diag(sigma'(z)) W` replaces unrolled backprop through
the fixed-point iteration.

Status is **shipped**. G1–G3 are CI-gated. IFT is the
chain rule at a fixed point, not an absence of the chain rule. Not a
global min and not CCF stretch. Bias collapse (`delta -> 0`)
supplies `sigma'`. Anderson acceleration is extra and raises.
See theory spec 08-08.

## Tracing (G4)

- **PyTorch.** `deq_solve` uses a Python `while`. Do not wrap it in
  `torch.compile`.
- **JAX.** The default Newton / Banach loop is `lax.while_loop`.
  `require_contraction=True` falls back to a Python `while` so a
  bound `>= 1` can raise instead of silently unrolling. Do not wrap
  a data-dependent Python `break` in `jax.jit`.

## Core algebra

::: omnibias.core.implicit
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.implicit
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.implicit
    options:
      show_root_heading: false
      heading_level: 3
