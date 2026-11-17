---
name: omnibias-advisor
description: Advise a user on which omnibias derivative program, API and optimizer fit their PINN or scientific-ML problem, measure the choice on their shape, and state what the evidence does and does not support.
---

# Advising on omnibias

Use this when the question is about a user's problem ("which program should I use", "will this be faster", "what do I train
it with"), not when changing the library; the package skills own maintenance. The work is to choose, measure on the user's
shape and report with its limits. Do not quote a speedup from memory: take it from the row of
[regimes](../../../docs/regimes.md) that covers the user's operator, back end, device, dtype and `D`/`H` range, and from
`explain()` of their own call. The [capability rule](../../../.cursor/rules/omnibias.mdc) states what the library claims;
[performance evidence](../../../docs/performance.md) states how each number was measured.

## 1. Fix the quantity and its order

Before any API, write down what the residual reads: a Laplacian power, a divergence, a log-determinant, directional or
mixed partials, a residual Gram or Gauss-Newton step; the highest derivative order; input dimension `D`, width `H`, depth,
batch, dtype and device; and whether the callable is only evaluated or is also differentiated into the parameters. The
quantity decides the strategy. Activation derivatives, directional jets, mixed jets, Laplacian contractions and parameter
curvature are different comparisons, and a win for one says nothing about another. Mixed jets grow combinatorially with
`D` and order; a Laplacian contraction does not carry a Hessian. A hard discontinuity is not differentiated.

## 2. Check the structure

A dense chain (a layer list, or `nn.Sequential` of linear layers and recognised activations) is served by the structured
programs; a gated, PirateNet-style or attention model runs on the generic engines and has no stream program, so expect
what those engines cost, not a structured-program result. See [models that are not dense chains](../../../docs/operator-planner.md#models-that-are-not-dense-chains)
and the [coverage matrix](../../../docs/operator-coverage.md). When `D` is larger than `H`, the first-layer projection
programs apply; a one-layer ridge network has closed forms. If the model is not covered, say so and name the baseline.

## 3. Choose the API family

Start from the planner front end of the back end the user trains in: `compile_operator` for a derivative operator and
`compile_residual` for a residual Gram, Gauss-Newton step or NTK diagonal, from `omnibias.jax.operators` or
`omnibias.torch.operators`. They choose the program from a descriptor and report the choice. Reach for a named
experimental program only when the user needs a specific one, and say it is opt-in. Layers, fields and certified or
partition modules are different families: [capabilities](../../../docs/capabilities.md) maps them to their need.

## 4. Choose the optimizer from the evidence

Derivative cost and optimizer choice are separate questions. Parameter-curvature methods (Gauss-Newton, Levenberg-Marquardt,
ENGD, SPRING, KFAC-style and L-BFGS in `optim_pinn`) state their stopping rules and assumptions; they do not establish
convergence for a PDE. Read [curvature-aware training](../../../docs/capabilities.md#curvature-aware-training) for what was
measured, and claim a better accuracy for the training time than Adam only where a registered comparison at the user's
problem class says so. Otherwise propose the comparison: the same time budget, held-out solution error, several seeds, and
compile time reported next to steady-state time.

## 5. Measure on the user's shape

The cost model ranks; only a timing decides. For a training run, ask the planner to time every feasible program once on the
value-and-gradient the step runs, and keep the answer:

```python
from omnibias.jax.operators import compile_operator

op = compile_operator(
    model, (params, x), operator="laplacian", differentiate="params",
    mode="find", top_k=None, expected_uses=steps, cache="plans.json",
)
print(op.explain())
```

`planned_evaluator` of the training-step modules applies this preset. `top_k=2`, the default, times only the two programs
the model ranks first. Give a large problem a finite `memory_budget`, or on JAX `safe_find=True`, because a candidate whose
buffers no card holds can end the process while it compiles. A plan is reused only under the same descriptor, device, budget
and library versions, so rerun the find after an upgrade.

## 6. Read `explain()`

`selected_strategy` and `mode` say what ran and whether it was measured: a `heuristic` plan has predictions only.
Compare `measured_seconds` with the predicted `seconds` of each candidate to see how far the ranking was from the
measurement. `refused` lists every program that does not apply with its reason, including a `find:` refusal for a candidate
that crashed or exceeded the budget; `executor_failures` lists what failed while timing or tracing. `objective` and
`uses_source` say what was ranked and where the use count came from, and `versions` and `memory_budget_bytes` say what the
plan is valid for.

## 7. Report with the limits

Say which regime row the answer rests on, with the losses of the same row, not only the wins, and say when the user's shape
lies outside the covered ranges: that is unmeasured, not a tie. Where an independent baseline is faster, name it and the
cells. Read a loss against omnibias's own variant as one omnibias program beating another. Keep compile time, resource
refusals and float32 cancellation visible, since a direct formula can suffer cancellation. A stochastic
estimator needs a key and an accuracy; an exact request refuses it. Finish with the command that reproduces the user's own
timing, not a table copied from this repository.
