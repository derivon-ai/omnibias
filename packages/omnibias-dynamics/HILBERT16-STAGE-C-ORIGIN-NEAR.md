# Hilbert XVI: nearer-interface matching-chart Lohner cover of `[15/8, 2]`

This companion continues [the `x=1/8` cover](HILBERT16-STAGE-C-ORIGIN-IFACE.md).
On `lambda1 = -2`, with `sep` a `PolynomialFlow` parameter, the
matching-chart height flow from `(x, y) = (1/16, 1)` has unique
transverse first-hit of `x = 4` on eight slabs of width `1/64`
covering `[15/8, 2]` (`r1` from `1/16` down to `0`). At the left
endpoint `r1 = 1/16` equals the start `x`. A single slab over the
whole compact is unresolved. A short horizon of 160 steps at
`sep = 2` does not certify. This is enclosure continuation of a
declared `sep` compact containing chart-O `sep = 2` from a start
closer to the first root, not every `r1`, not the shrinking
interface `x = r1(1+theta)`, not complete first-hit on chart O,
C2, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Nearer-interface matching-chart Lohner cover of `[31/16, 2]` from `x=1/32` is `stage_c_origin_x32`.
- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCOriginNear.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOriginNear.lean).
Python: `omnibias.dynamics.stage_c_origin_near`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_origin_near.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_origin_near
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginNear
```
