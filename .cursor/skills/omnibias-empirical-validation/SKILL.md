---
name: omnibias-empirical-validation
description: Make "validated and refined from training results" an enforceable loop — pick every design choice from a measured curve, back the headline with a sound certificate, beat a named classical baseline, run CPU smoke locally and heavier sweeps as submitted jobs, and lock results as a validation report plus CI smoke. Use when validating, benchmarking, or refining any omnibias package.
---

# Empirical validation and refinement

A package is done when its headline claim survives measurement against a
named baseline. Treat data-driven / verified / best-in-class as **gates**.

## Why nested AD fails

A loss that fell is not a bakeoff. Nested AD stacks report a float without a
`gates` block, without a named classical baseline, and without a sound
certificate. Overfitting a single seed is the default failure mode of
ungated research.

## What only this tower unlocks

Every knob chosen from a measured curve; the headline backed by a sound
object (`certify_gap`, grid-and-random enclosure); beat or match a named
classical baseline on a fair budget. Closed-form towers make the verified
gate available; nested AD stops at the loss.

## Use

- **Data-driven.** Sweep `AnnealSchedule` `beta0` / `beta_growth` / `stages` /
  `steps`, SOS `level`, decode restarts, step `scale` — pick from a curve.
- **Verified.** Certified gap that sandwiches an oracle, or the grid-and-random
  enclosure soundness test for the `delta -> 0` register.
- **Best-in-class.** Beat or match a named classical baseline (same instances,
  same budget).

Phase A (explore): strongest constructive route; scratch under
`$OMNIBIAS_SCRATCH`. Phase B (ship): acceptance-gated API + CI smoke +
committed summary JSON with a `gates` block (multi-seed, named baseline,
absolute skill floor).

Local CPU smoke on tiny `n` is the inner loop. Heavier / GPU work uses the
optional `$OMNIBIAS_SUBMIT` wrapper when set; tracked files stay
vendor-neutral.

Commit smoke JSON; `--full` writes multi-seed acceptance JSON. Public
artifacts regenerable.

## Extend

Compose with `omnibias-pinn-research`, `omnibias-control-research`,
`omnibias-frontier`, `omnibias-new-package`. PINN four-gap gates:
`benchmarks/_gates.py`.

## Next invention

A headline number that currently lives as a leftover smoke, promoted to a
multi-seed `gates` JSON that beats a named baseline and carries a sound
certificate.

## Further references

- `docs/benchmarks.md`, `docs/honesty.md`
- `docs/benchmarks/pinn_four_gap_matrix.md`
