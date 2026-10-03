# Hilbert XVI: parametric-`eps` Stage-C Lohner cover of `[23/400, 1/16]`

This companion continues [the shrinking-eps point pack](HILBERT16-STAGE-C-ONESHOT-EPS.md).
On `lambda1 = -2`, `sep = 0`, with `eps` a `PolynomialFlow` parameter, the
matching-chart height flow from `(x, y) = (1/4, 1)` has unique
transverse first-hit of `4 eps x - 1 = 0` (the image of `E_out`) on
four slabs of width `1/800` covering `[23/400, 1/16]`. A single slab
over the whole compact is unresolved. This is enclosure continuation
of a declared `eps` compact containing `eps = 1/16`, not a
uniform-in-`eps` theorem for every `eps`, not chart O, C2, G1, or
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Chart-O matching-chart Lohner pack is `stage_c_origin`.
- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCEpsSpan.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCEpsSpan.lean).
Python: `omnibias.dynamics.stage_c_eps_span`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_eps_span.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_eps_span
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCEpsSpan
```
