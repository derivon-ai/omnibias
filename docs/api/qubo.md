# omnibias-qubo

Differentiable **and** certified **quadratic Boolean optimization** (QUBO / Ising) on
the omnibias stack: an annealed sigmoid relaxation solved differentiably by the
temperature-collapse penalty, a rounding + 1-flip local-search decoder, a
brute-force oracle, and a rigorous optimality-gap certificate.

Minimizing `E(x) = xᵀ Q x + cᵀ x` over `x ∈ {0,1}ⁿ` is NP-hard, so no poly-time
differentiable map yields the *exact* global optimum (that would imply `P = NP`, and the
exact argmin's gradient is a.e. zero). The sound "differentiable QUBO" is therefore a
three-part object -- a **yes-if**, not a no-because:

\[
\underbrace{\ell}_{\text{certified lower bound}} \;\le\; \text{optimum} \;\le\;
\underbrace{E(z)}_{\text{decoded binary point (upper bound)}} .
\]

1. A **differentiable annealed relaxation** -- a soft assignment
   `x = sigmoid(beta·theta) ∈ (0,1)ⁿ` descended on the closed-form energy gradient while
   `beta → ∞` collapses it onto a binary vertex, *unrolled* for backprop so a model that
   predicts `Q` / `c` trains *through* the optimizer. Bit-identical torch + jax twins.
2. A **heuristic decoder** (rounding + 1-flip local search) that yields a valid binary
   point -- the upper bound (`brute_force_min` is the exact small-`n` oracle).
3. A **rigorous optimality-gap certificate**: a Lasserre / SOS bound over the Boolean
   hypercube (`omnibias-sos`) or a cheap spectral / box-QP bound (`omnibias-convex`) is a
   *lower* bound on the true optimum, so `lower <= optimum <= E(z)` is a certified gap --
   never asserted zero; a weaker bound only widens it.

Terminology: the relaxation's `sigmoid(beta·)`, `beta → ∞` is the **feasibility /
temperature** sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct
from the **founding bias collapse** -- the multi-bias `delta → 0` limit to the
closed-form derivative `sigma^(K-1)` (see [Theory](../theory.md)).

## Problem & certificate containers

::: omnibias.qubo.problem
    options:
      show_root_heading: false
      heading_level: 3

## Conversions & SOS encoders (numpy)

::: omnibias.qubo._core.convert
    options:
      show_root_heading: false
      heading_level: 3

## Decoder & exact oracle (numpy)

::: omnibias.qubo._core.decode
    options:
      show_root_heading: false
      heading_level: 3

## Problem constructors (numpy)

::: omnibias.qubo._core.frontends
    options:
      show_root_heading: false
      heading_level: 3

## Differentiable relaxation layer (JAX)

::: omnibias.qubo.jax.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Backend twin (torch)

Bit-identical PyTorch twin of the relaxation layer (parity `~1e-9`, float64).

::: omnibias.qubo.torch.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Lower bounds (numpy)

::: omnibias.qubo._core.bound
    options:
      show_root_heading: false
      heading_level: 3

## Optimality-gap certificate

::: omnibias.qubo.certify
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).
