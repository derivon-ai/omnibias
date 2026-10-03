# Hilbert XVI: comparison first-hit for every `eps` in `(0, 1/16]`

This companion continues [the `eps` slab out to `x=8`](HILBERT16-STAGE-C-COMPARE.md).
On `lambda1 = -2` the root excess `2 r1 - r1^2` lies in `[0, 1]` for
every `r1` in `[0, 1]`. From `(x, y) = (1/4, 1)`, a phase-wise lower
bound on `dy/dx` keeps `y + x(x-2) > 1/2` on steps of `1/40` from
`x = 1/4` to `x = 2`, with `y(2) > 16`. For `x >= 2` the product
`x(x-2)` is nonnegative, so `dx/dσ >= eps/2` on `x >= 1/4` whenever
`eps` is in `(0, 1/16]`. Every such orbit hits the matching outgoing
section `x = (1/4)/eps`, in time at most `(1 - eps) / (2 eps^2)`.
At `eps = 1/16` that majorant equals `120`. Holding `y` at `1` stalls
on the phase through `x = 1`. This is a comparison first-hit from one
fixed start, not a Lohner tube, not `eps > 1/16`, not the shrinking
interface `x = r1(1+theta)`, not complete first-hit on chart O, C2,
G1, or Hilbert XVI. Every start in `(0, 1/2]`, including an interface
that lands there, is `stage_c_interface`.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCUniform.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCUniform.lean).
Python: `omnibias.dynamics.stage_c_uniform`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_uniform.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_uniform
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCUniform
```
