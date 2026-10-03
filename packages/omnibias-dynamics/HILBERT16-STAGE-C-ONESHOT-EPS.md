# Hilbert XVI: shrinking-`eps` Stage-C Lohner pack of matching-chart `x=n/4`

This companion continues [the Stage-C Lohner first-hit at `eps=1/16`](HILBERT16-STAGE-C-ONESHOT.md).
On `lambda1 = -2`, `sep = 0`, the matching-chart height flow from
`(x, y) = (1/4, 1)` has unique transverse first-hit of `x = rho/eps = n/4`
at `eps = 1/n` for `n in {16, 20, 25}` with `step = 1/20` and declared
horizons `T = 6, 7, 8`. A short horizon of 120 steps at `n = 20` does
not certify. This is a finite shrinking pack, not a uniform-in-`eps`
theorem, not chart O, C2, `dx_e` off the kill line, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps` (a compact interval
  containing `1/16` is
  [HILBERT16-STAGE-C-EPS-SPAN.md](HILBERT16-STAGE-C-EPS-SPAN.md)).
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCOneshotEps.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOneshotEps.lean).
Python: `omnibias.dynamics.stage_c_oneshot_eps`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_oneshot_eps.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_oneshot_eps
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshotEps
```
