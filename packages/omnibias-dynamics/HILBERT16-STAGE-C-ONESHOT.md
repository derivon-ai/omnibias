# Hilbert XVI: kill-line Stage-C Lohner first-hit of matching-chart x=4

This companion continues [the Stage-C comparison first-hit of `E_out`](HILBERT16-STAGE-C-SEC.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the matching-chart height flow with independent `sigma`,
`dy/d sigma = x y` and

    dx/d sigma = eps y + eps (x-r1)(x-r2),

is polynomial. The section `x = rho/eps = 4` is the leading image of
`E_out`. From the declared Stage-C start `(x, y) = (1/4, 1)`,
`certify_stopped_event` hits `x = 4` uniquely and transversely on
`sep in {0, 3/5, 1}` with `step = 1/20` and `max_steps = 120`. A short
horizon of 80 steps does not certify at `sep = 0`. This is a finite
Lohner pack from Stage-C start, not a uniform-in-`eps` theorem, not
chart O, C2, `dx_e` off the kill line, G1, or Hilbert XVI. Matching-chart
Lohner `E_out` from the GRAZING start is
[HILBERT16-E-OUT-SECTION.md](HILBERT16-E-OUT-SECTION.md).

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps` (three squares are
  [HILBERT16-STAGE-C-ONESHOT-EPS.md](HILBERT16-STAGE-C-ONESHOT-EPS.md)).
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCOneshot.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOneshot.lean).
Python: `omnibias.dynamics.stage_c_oneshot`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_oneshot.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_oneshot
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshot
```
