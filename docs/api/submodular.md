# omnibias-submodular

Differentiable **and** certified **monotone submodular maximization** on the omnibias
stack: the multilinear-extension relaxation, continuous greedy (Frank-Wolfe over a
matroid polytope) with bit-identical torch + jax twins, pipage / swap rounding carrying
the a-priori `1 - 1/e` guarantee, a classical greedy baseline + a brute-force oracle, and
a rigorous optimality-gap certificate.

Maximizing a monotone submodular `f : 2^[n] → ℝ≥0` over the independent sets of a matroid
(cardinality / partition) is NP-hard, so no poly-time differentiable map yields the
*exact* optimum (that would imply `P = NP`). The sound "differentiable submodular
maximization" is therefore a three-part object -- a **yes-if**, not a no-because:

\[
\underbrace{f(S)}_{\text{decoded feasible set (lower bound)}} \;\le\; \text{OPT} \;\le\;
\underbrace{U(S)}_{\text{marginal-gain bound (upper bound)}} .
\]

1. The **multilinear extension** `F(p) = E_{x∼p}[f(x)]` replaces `x ∈ {0,1}ⁿ` by
   `p ∈ [0,1]ⁿ` (the unique multilinear polynomial interpolating `f`), and **continuous
   greedy** -- Frank-Wolfe over the matroid polytope -- returns a fractional `p*` with
   `F(p*) ≥ (1 − 1/e)·OPT`. The exact per-coordinate gradient keeps the step closed-form;
   the differentiable soft LP oracle `sigmoid(beta·(g − tau))` is *unrolled* for backprop,
   so a model predicting the objective's data trains *through* it. Bit-identical torch +
   jax twins (the coverage family).
2. **Pipage / swap rounding** turns `p*` into an integral independent set `S` with
   `f(S) ≥ F(p*)`, hence the a-priori certified `f(S) ≥ (1 − 1/e)·OPT`
   (`greedy_maximize` is the best-in-class baseline; `brute_force_max` the exact
   small-`n` oracle).
3. A **rigorous optimality-gap certificate**: the decoded value is a *lower* bound and the
   upper bound `U(S)` = min(marginal-gain, modular) bound an *upper* bound on `OPT`, so
   `f(S) ≤ OPT ≤ U(S)` is a certified gap -- never asserted zero; a looser `U` only widens
   it. A secondary unconstrained SOS bound is available via the `omnibias-discrete` seam.

Beyond the coverage headline the package spans the standard toolbox, all keeping the same
yes-if framing: **accelerated greedy** (`lazy_greedy` CELF, identical output; `stochastic_greedy`
`(1 − 1/e − ε)`), **knapsack** budgets (`knapsack_maximize`, `certify_knapsack_gap`),
more **function families** (`FacilityLocation`, `BudgetAdditive`, `LogDeterminant` DPP,
`GraphCut`) and a submodular **algebra** (`Sum` / `Scaled` / `Saturated`), **non-monotone**
maximization (`double_greedy` `1/2`, `measured_continuous_greedy` `1/e`), **more matroids**
(laminar / graphic / transversal / intersection with `p_matroid_greedy` `1/(p+1)`),
**streaming** (`sieve_streaming` one-pass `1/2 − ε`), and a **curvature**-sharpened ratio.

Scope note -- **minimization ≠ maximization.** Unconstrained submodular *minimization*
`min_S f(S)` is a **P-class** problem solved **exactly** in poly time (`submodular_minimize`
/ `min_norm_point`, Fujishige–Wolfe over the base polytope; `lovasz_extension` is its convex
closure). That exactness is *not* a `P = NP` claim: it is the NP-hard *maximization* that
only ever earns the certified `1 − 1/e` approximation, never an exact solver.

Terminology: the multilinear extension relaxing `{0,1}ⁿ → [0,1]ⁿ` and the Frank-Wolfe
oracle `sigmoid(beta·(g − tau))`, `beta → ∞`, hardening onto a `0/1` matroid-basis vertex
is the **feasibility / temperature** sense of "collapse" (a soft indicator hardening to a
step), distinct from the **founding bias collapse** -- the multi-bias `delta → 0` limit to
the closed-form derivative `sigma^(K-1)` (see [Theory](../theory.md)).

## Problem & certificate containers

::: omnibias.submodular.problem
    options:
      show_root_heading: false
      heading_level: 3

## Submodular functions & multilinear extension

::: omnibias.submodular.functions
    options:
      show_root_heading: false
      heading_level: 3

## Matroid constraints & the LP oracle

::: omnibias.submodular.matroid
    options:
      show_root_heading: false
      heading_level: 3

## Problem constructors (numpy)

::: omnibias.submodular._core.frontends
    options:
      show_root_heading: false
      heading_level: 3

## Continuous greedy (numpy)

::: omnibias.submodular._core.continuous
    options:
      show_root_heading: false
      heading_level: 3

## Rounding: pipage & swap (numpy)

::: omnibias.submodular._core.rounding
    options:
      show_root_heading: false
      heading_level: 3

## Greedy baseline, accelerated greedy & exact oracle (numpy)

Includes `greedy_maximize`, Minoux's `lazy_greedy` (CELF -- identical output, far fewer
oracle calls), Mirzasoleiman's seeded `stochastic_greedy` (`(1 − 1/e − ε)`),
`p_matroid_greedy` (`1/(p+1)` over a matroid intersection), and the exponential
`brute_force_max` oracle.

::: omnibias.submodular._core.greedy
    options:
      show_root_heading: false
      heading_level: 3

## Knapsack-constrained maximization (numpy)

A knapsack (budget) constraint is *not* a matroid; `knapsack_maximize` is Sviridenko's
partial-enumeration greedy (`1 − 1/e`) and `certify_knapsack_gap` seals it with a
fractional-knapsack upper bound.

::: omnibias.submodular.knapsack
    options:
      show_root_heading: false
      heading_level: 3

## Non-monotone maximization (numpy)

For non-monotone submodular `f` (e.g. `GraphCut`): `double_greedy` (unconstrained, `1/3`
deterministic / `1/2` randomized) and `measured_continuous_greedy` (matroid, `1/e`), with
the sound singleton `nonmonotone_upper_bound`.

::: omnibias.submodular._core.nonmonotone
    options:
      show_root_heading: false
      heading_level: 3

## Streaming (numpy)

::: omnibias.submodular._core.streaming
    options:
      show_root_heading: false
      heading_level: 3

## Lovász extension & exact submodular minimization (numpy, P-class)

The honest mirror of the NP-hard maximization: `lovasz_extension` (exact convex closure)
and `submodular_minimize` / `min_norm_point` (Fujishige–Wolfe, exact in polynomial time).

::: omnibias.submodular.lovasz
    options:
      show_root_heading: false
      heading_level: 3

## End-to-end pipeline (numpy)

::: omnibias.submodular._core.pipeline
    options:
      show_root_heading: false
      heading_level: 3

## Differentiable relaxation layer (JAX)

::: omnibias.submodular.jax.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Backend twin (torch)

Bit-identical PyTorch twin of the continuous-greedy relaxation (parity `~1e-9`, float64).

::: omnibias.submodular.torch.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## Upper bound (numpy)

::: omnibias.submodular._core.bound
    options:
      show_root_heading: false
      heading_level: 3

## Optimality-gap certificate

::: omnibias.submodular.certify
    options:
      show_root_heading: false
      heading_level: 3

Status: Alpha (`0.1.0a1`).
