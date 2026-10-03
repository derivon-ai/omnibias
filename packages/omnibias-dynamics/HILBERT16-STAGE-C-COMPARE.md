# Hilbert XVI: uniform-in-`r1` comparison first-hit on one `eps` slab

This companion continues [the `x=1/32` Lohner cover](HILBERT16-STAGE-C-ORIGIN-X32.md).
On `lambda1 = -2` the matching-chart root excess
`2 r1 - r1^2` lies in `[0, 1]` for every `r1` in `[0, 1]`. From
`(x, y) = (1/4, 1)`, a phase-wise Interval bound keeps
`x' >= eps (y + x(x-2))` positive on steps of `1/40` out to `x = 8`,
for every `eps` in `[1/32, 1/16]`. The value `8 = (1/4)/(1/32)` is the
matching outgoing section at the small end of that slab, so every
larger `eps` has already crossed `x = (1/4)/eps`. Freezing `y` at `1`
stalls at the neck `x = 1`. This is a comparison first-hit on one
`eps` slab and one start, not a Lohner tube, not every `eps`, not the
shrinking interface `x = r1(1+theta)`, not complete first-hit on
chart O, C2, G1, or Hilbert XVI. The comparison for every `eps` in
`(0, 1/16]` is `stage_c_uniform`.

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

Lean: [Hilbert16StageCCompare.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCCompare.lean).
Python: `omnibias.dynamics.stage_c_compare`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_compare.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_compare
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCCompare
```
