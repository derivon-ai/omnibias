# omnibias-nphard

Differentiable **and** certified heuristics for named **NP-hard families** -- the
**quadratic assignment problem** (QAP), the **generalized assignment problem** (GAP), and
**parallel-machine scheduling** -- plus a from-scratch **MCTS "search track"** that uses
the differentiable relaxation as an AlphaZero-style prior. Built on
[omnibias-qubo](qubo.md): each family reduces to a quadratic pseudo-Boolean (QUBO-form)
energy, so the shared annealed relaxation, structure-preserving decoders, and
spectral / SOS certificate all apply.

These families are NP-hard, so no poly-time differentiable map yields the *exact* global
optimum (that would imply `P = NP`, and the exact argmin's gradient is a.e. zero). Like
[omnibias-qubo](qubo.md) / [omnibias-routing](routing.md), and unlike the P-class
[omnibias-combinatorics](combinatorics.md) (whose certificate is *tight*), the deliverable
is a **yes-if** -- a certified but honestly **non-tight** gap:

\[
\underbrace{\ell}_{\text{certified lower bound}} \;\le\; \text{optimum} \;\le\;
\underbrace{E(z)}_{\text{decoded feasible solution (upper bound)}} .
\]

1. A **differentiable annealed relaxation** -- a soft assignment
   `x = sigmoid(beta·theta) ∈ (0,1)ⁿ` descended while `beta → ∞` collapses it onto a
   binary vertex, *unrolled* for backprop so a model that predicts the flow / cost /
   processing-times trains *through* the solver. Bit-identical torch + jax twins.
2. A **structure-preserving decoder** (`decode`) -- Hungarian + 2-opt for QAP, a
   capacity-feasible rounding for GAP, argmax + LPT-repair for scheduling -- a strong
   heuristic *upper* bound, with named classical baselines (`classical_optimum`: scipy
   FAQ/2-opt, LPT, OR-Tools / greedy) and an exact exponential oracle (`brute_force_min`).
3. A **rigorous optimality-gap certificate** (`certify_gap`): a spectral / box-QP
   (`kind="spectral"`), Lasserre / SOS (`kind="sos"`), or -- for QAP -- the QAP-specific
   **Gilmore-Lawler** bound (`kind="glb"`) as the *lower* bound, so `lower <= optimum <=
   E(z)` is certified. The gap is **honestly non-tight** -- a weaker bound only widens it;
   it is never asserted zero.

### Certified chip placement (the QAP flagship)

Block / macro floorplanning is exactly the Koopmans-Beckmann QAP: place `N` modules on `N`
grid slots to minimise the connectivity-weighted Manhattan wirelength. `placement_qap`
builds such an instance from a netlist connectivity + a slot grid, and the **Gilmore-Lawler
bound** (`gilmore_lawler_bound` / `certify_gap(kind="glb")`) certifies it. GLB is the
category-of-one lever: **sound** at every size (exact integer arithmetic for the integer
placement default, outward-rounded intervals for floats), far **tighter** than the generic
spectral bound (single/low-double-digit gaps vs. ~100%), and -- unlike the SOS SDP, which is
intractable past `N ≈ 4` -- it stays `O(N³)` and non-trivial at realistic block counts
(`N ≈ 12-25`). See [`certified_chip_placement.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/certified_chip_placement.py).
Scope is block-level floorplanning (tens of modules) and the gap is NP-hard-honest -- never
a claim of beating industrial million-cell placers.

Terminology: the relaxation's `sigmoid(beta·)`, `beta → ∞` is the **feasibility /
temperature** sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct
from the **founding bias collapse** -- the multi-bias `delta → 0` limit to the
closed-form derivative `sigma^(K-1)` (see [Theory](../theory.md)).

## Schedule & certificate containers

::: omnibias.nphard.problem
    options:
      show_root_heading: false
      heading_level: 3

## Quadratic assignment (QAP, numpy)

::: omnibias.nphard._core.qap
    options:
      show_root_heading: false
      heading_level: 3

## Generalized assignment (GAP, numpy)

::: omnibias.nphard._core.gap
    options:
      show_root_heading: false
      heading_level: 3

## Parallel-machine scheduling (numpy)

::: omnibias.nphard._core.scheduling
    options:
      show_root_heading: false
      heading_level: 3

## Gilmore-Lawler bound (QAP, numpy)

::: omnibias.nphard._core.bound
    options:
      show_root_heading: false
      heading_level: 3

## Decoder / baseline / oracle dispatch (numpy)

::: omnibias.nphard._core.decode
    options:
      show_root_heading: false
      heading_level: 3

## Decision-focused metrics (numpy)

::: omnibias.nphard._core.decision
    options:
      show_root_heading: false
      heading_level: 3

## Differentiable relaxation layer (JAX)

::: omnibias.nphard.jax.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Backend twin (torch)

Bit-identical PyTorch twin of the relaxation layer (scheduling parity `~1e-13`, float64;
a frustrated QAP coordinate pinned near `0.5` is a chaotic-amplification regime where the
two frameworks' reduction order can diverge to `~1e-7`).

::: omnibias.nphard.torch.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Decision-focused QAP cost (JAX)

::: omnibias.nphard.jax.decision_focused
    options:
      show_root_heading: false
      heading_level: 3

## Optimality-gap certificate

::: omnibias.nphard.certify
    options:
      show_root_heading: false
      heading_level: 3

## Go-like MCTS search track

::: omnibias.nphard.search.mcts
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).
