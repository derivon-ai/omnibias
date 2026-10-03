# Hilbert XVI: kill-line Stage-A shrinking-rectangle wall

This companion continues [the tracked product](HILBERT16-ENTRY-EXIT-LEADING.md)
and [coalescing capture](HILBERT16-COALESCING-CAPTURE.md) §5.1.
On `lambda1 = -2` the slow-line midpoint is `rstar = 1` and
`r1 = 1 - sep/2`. With `theta = 1/8` the Stage-A walls are
`a = r1 - theta sep` and `bnd = r1 + theta sep`. The limiting
quadratic satisfies the exact identities

    B_-(a) = theta (1 + theta) sep^2,
    B_-'(bnd) = -sep (1 - 2 theta).

For `sep in [0, 1]` the left wall stays in `[3/8, 1]`. Interval wrapping
on that compact still yields `a > 1/4`, so a declared floor
`a_min = 1/4` holds. The leading `Psi_pre` factor
`B_-(a) / (B_-(0) sep^2)` is then in `(9/64, 3/16]` and below `1/4`.
This is the geometric core of the χ-rectangle, not
`dx_e / d kappa`, not a `chi` threshold, not Stage C, first-hit, G1, or
Hilbert XVI. Kill-line Stage B is [HILBERT16-STAGE-B.md](HILBERT16-STAGE-B.md).

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- The uniform-in-`chi` bound `dx_e / d kappa` is
  [HILBERT16-DX-E-UNIF.md](HILBERT16-DX-E-UNIF.md); leading factors are
  [HILBERT16-DX-E-LEADING.md](HILBERT16-DX-E-LEADING.md).
- Stage C first-root outgoing with uniform `a_min` is
  [HILBERT16-STAGE-C.md](HILBERT16-STAGE-C.md).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.

Lean: [Hilbert16StageA.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageA.lean).
Python: `omnibias.dynamics.stage_a`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_a.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_a
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageA
```
