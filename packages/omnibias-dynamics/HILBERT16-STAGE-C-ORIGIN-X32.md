# Hilbert XVI: nearer-interface matching-chart Lohner cover of `[31/16, 2]`

This companion continues [the `x=1/16` cover](HILBERT16-STAGE-C-ORIGIN-NEAR.md).
On `lambda1 = -2`, with `sep` a `PolynomialFlow` parameter, the
matching-chart height flow from `(x, y) = (1/32, 1)` has unique
transverse first-hit of `x = 4` on eight slabs of width `1/128`
covering `[31/16, 2]` (`r1` from `1/32` down to `0`). At the left
endpoint `r1 = 1/32` equals the start `x`. A single slab over the
whole compact is unresolved. A short horizon of 160 steps at
`sep = 2` does not certify. This is enclosure continuation of a
declared `sep` compact containing chart-O `sep = 2` from a start
closer to the first root, not every `r1`, not the shrinking
interface `x = r1(1+theta)`, not complete first-hit on chart O,
C2, G1, or Hilbert XVI. The uniform-in-`r1` comparison on
`eps in [1/32, 1/16]` is `stage_c_compare`.

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

Lean: [Hilbert16StageCOriginX32.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOriginX32.lean).
Python: `omnibias.dynamics.stage_c_origin_x32`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_origin_x32.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_origin_x32
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginX32
```
