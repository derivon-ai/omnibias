# Hilbert XVI: parametric-`sep` matching-chart Lohner cover of `[3/2, 2]`

This companion continues [the chart-O point pack](HILBERT16-STAGE-C-ORIGIN.md).
On `lambda1 = -2`, with `sep` a `PolynomialFlow` parameter, the
matching-chart height flow from `(x, y) = (1/4, 1)` has unique
transverse first-hit of `x = 4` on eight slabs of width `1/16`
covering `[3/2, 2]` (`r1` from `1/4` down to `0`). A single slab
over the whole compact is unresolved. This is enclosure continuation
of a declared `sep` compact containing chart-O `sep = 2`, not every
`r1`, not the shrinking interface `x = r1(1+theta)`, not complete
first-hit on chart O, C2, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Nearer-interface matching-chart Lohner cover of `[7/4, 2]` from `x=1/8` is `stage_c_origin_iface`.
- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCOriginSpan.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCOriginSpan.lean).
Python: `omnibias.dynamics.stage_c_origin_span`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_origin_span.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_origin_span
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginSpan
```
