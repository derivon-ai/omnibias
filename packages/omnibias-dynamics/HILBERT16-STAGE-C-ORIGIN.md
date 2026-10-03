# Hilbert XVI: kill-line matching-chart Lohner pack toward chart O

This companion continues [the Stage-C Lohner first-hit](HILBERT16-STAGE-C-ONESHOT.md)
and [the origin layer](HILBERT16-COALESCING-CAPTURE.md) §7. On
`lambda1 = -2`, `r1 = 1 - sep/2`, the origin layer is `sep -> 2`,
`r1 -> 0`. From the compact post-corridor start `(x, y) = (1/4, 1)`,
`certify_stopped_event` hits matching-chart `x = 4` uniquely and
transversely on `sep in {3/2, 7/4, 2}` (`r1 in {1/4, 1/8, 0}`) with
`step = 1/20` and `max_steps = 160`. A short horizon of 120 steps
does not certify at `sep = 2`. This is a finite origin pack from a
compact `x`-start, not every `r1`, not the shrinking interface
`x = r1(1+theta)`, not complete first-hit on chart O, C2, G1, or
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Parametric-sep matching-chart Lohner cover of `[3/2, 2]` is `stage_c_origin_span`.
- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCOrigin.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOrigin.lean).
Python: `omnibias.dynamics.stage_c_origin`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_origin.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_origin
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOrigin
```
