---
name: omnibias-dev-empirical-validation
description: Make "validated and refined from training results" an enforceable loop for omnibias packages -- a data-driven / verified / best-in-class methodology -- pick every design choice from a measured curve, back the headline with a sound certificate, beat a named classical baseline, run CPU smoke locally and heavier sweeps via optional GPU submit, calibrate across seeds to avoid overfitting, and lock results in as a validation report + CI smoke. Use when validating, benchmarking, tuning, or refining any omnibias package against baselines and oracles. For contributors modifying omnibias itself, not for consumers using it.
---

# Empirical validation & refinement (data-driven, verified, best-in-class)

A package is not "done" when tests pass; it is done when its headline claim
survives measurement against a real baseline. Treat the three words as **gates**,
not adjectives.

## The three gates

- **Data-driven.** Every knob (`AnnealSchedule` `beta0` / `beta_growth` /
  `stages` / `steps`, SOS `level`, decode restarts, step `scale`) is chosen from
  a measured curve you can point to -- sweep, plot, pick. No guessed schedules.
- **Verified.** The headline number is backed by a *sound* object: a certified
  gap that sandwiches an oracle (`certify_gap`), or the grid-and-random enclosure
  soundness test for the `delta -> 0` register -- not merely a loss that fell.
- **Best-in-class.** Before shipping, **beat or match a named classical baseline**
  on a fair benchmark (same instances, same budget). No baseline => no claim.

Refine = iterate on the training dynamics until all three pass, then lock them in.

## Phase A (explore) vs Phase B (ship)

| Phase | Goal | Evidence |
| --- | --- | --- |
| **A – explore** | Find the strongest constructive route; losing baselines trigger diagnosis and another attempt, not immediate claim reduction | Scratch artifacts under `$OMNIBIAS_SCRATCH`; notebooks / uncommitted sweeps OK |
| **B – ship** | Lock an acceptance-gated API + CI smoke + committed summary JSON with a `gates` block | Multi-seed distributions, named baseline, absolute skill floor, certificate when the math supports it |

A failed Phase-A run is a research input. Only after a concrete structural blocker
(or a measured compute wall) may Phase B narrow the public claim -- and only after
the constructive routes have been tried.

## Local vs submitted compute

- **Local CPU smoke** on `uv run python`: imports, tiny `n`, unit tests, a
  1-2 min sanity curve. This is the loop you iterate in.
- **Heavier / GPU work** via the optional `$OMNIBIAS_SUBMIT` wrapper when set;
  otherwise run locally (backgrounded with a durable log under
  `$OMNIBIAS_SCRATCH`). Big artifacts -> `$OMNIBIAS_SCRATCH/omnibias_runs/<name>/`;
  write only **summaries** (metrics JSON + small plots + a short md) back into
  the repo -- never large binaries in the tree.

```bash
# tiny local smoke first
uv run python benchmarks/<name>.py
# then the full multi-seed acceptance run
uv run python benchmarks/<name>.py --full
```

## Validity floor (absolute gates)

Ask these three questions in order before any arm comparison:

1. Is the **reference** physically valid? (maximum principle, known amplitude, …)
2. Does every prediction beat the **zero predictor**? (`skill_score > 0`)
3. Does absolute error clear a **named threshold**? (`rel_l2 <= tol`)

Shared helpers live in `benchmarks/_gates.py`. Every public artifact should emit
a `gates` block that self-declares pass/fail so a JSON can never look like a
result while encoding a divergence.

## What to measure, with acceptance gates

| Axis / package | Acceptance gate |
|---|---|
| `beta->inf` (logic, combinatorics, submodular, transport, packing) | certified bound **sandwiches** `brute_force_min` on small `n` across >=K seeds; `decode` energy <= oracle; torch<->jax parity `< tol`; **train-through strictly improves** the decoded decision vs a no-optimizer baseline; schedule chosen by *measured* convergence, not guessed |
| `delta->0` (`difference`) | interval-tower enclosure contains a **dense deterministic grid AND a random sample** of true values (the `verified-primitive` rule) |
| DP (`struct`) | `logsumexp_beta >= max` gap bound holds; matches exact DP on small instances |
| trees (`tab`) | monotonicity / robustness certificates pass; **2nd-order training beats a 1st-order baseline** on held-out tabular data |
| PINN causality (`pinn.train`) | two families (heat hard-cage + Krishnapriyan reaction soft IC); IC via `ic_fn` on marcher points; every arm `skill_score > 0`; heat best-arm clears named rel-L2; reaction marching beats whole-interval; `advance_policy="gate"` |
| PINN geometry (`pinn.domain`) | boundary identity by construction; hard interior skill > 0 and beats soft-penalty on named shapes |
| PINN operators (`pinn.operator`) | reference maximum principle; every arm skill > 0; conditioned median rel-L2 beats unconditioned **and** per-instance retrain |
| PINN spectral (`pinn` FBPINN/NTK/lstsq) | one-shot `lstsq` rel-L2 < 5e-6 through f=16; capacity falsification at high frequency; GD arms for mechanism evidence; publish `lstsq_matched` beside capacity-rich `lstsq`; speed claims require per-arm `wall_seconds` / `median_wall_seconds` (never invent a speedup); memory is structural `O(N H)` |

## Anti-overfitting rule (the sharpest one)

Calibrate every metric across **>= K seeds** (K ~ 5-20), never one instance.
Tuning a tolerance or a schedule to pass a single seed is *the* failure mode.
Concretely -- the qubo torch/jax parity story: a "frustrated" instance amplified
float64 reduction-order differences to ~1e-7; the honest fix was measuring parity
*across seeds* and pinning the tight-tol test to a well-determined instance, not
loosening the tol to hide the frustrated one.

## Named baselines (best-in-class must be concrete)

- combinatorics: Hungarian / LP relaxation (tight when integral).
- submodular: greedy (the `1 - 1/e` guarantee).
- logic (MaxSAT/#SAT): a classical solver as ground truth on small instances.
- qubo: brute force (small `n`) + the spectral / SOS bounds.
- trees / tabular: a gradient-boosting baseline on a held-out split.
- PINN spectral: the zero predictor and a plain MLP; the decisive arm is
  one-shot least-squares when the PDE / regression is linear in the readout.
  Quote wall-clock only from instrumented artifacts; state the `O(N H)`
  design-matrix memory cost honestly.

## The validation-report artifact (lock it in)

Each package ships a runnable example (mirror
`docs/examples/certified_differentiable_qubo.py`) that exercises **both** halves
-- the certified gap **and** a train-through improvement -- deterministic and
CPU-tiny so it wires in as a **CI smoke**. Alongside it: a metrics JSON and a
short markdown summary. The example is what reviewers run; the JSON / plots are
the evidence.

## Measure so the win is undeniable

- A weaker bound only **widens** the certified gap; never silently tighten a
  bound or loosen a soundness test to pass. Report blow-up / non-convergence as a
  finding that upgrades the validity floor.
- Label results **closed-form** vs **autodiff-exact** vs **numerical** (the
  `verified-primitive` convention).
- If the baseline wins, say so and either fix the implementation (preferred) or
  narrow the acceptance domain after the constructive routes have been tried.
  "Best-in-class" is earned per benchmark, then claimed plainly.
- Skills encode methodology + gates; wire the gates into a test or CI smoke so
  they cannot silently rot (the `test_concept_terminology.py` pattern).
