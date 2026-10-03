# Hilbert XVI: comparison first-hit from every start in `(0, 1/2]`

This companion continues [the fixed start `x=1/4`](HILBERT16-STAGE-C-UNIFORM.md).
On `lambda1 = -2` the product `x(x-2)` on `(0, 1/2]` is at least
`-3/4`, so `y + x(x-2) >= 1/4` while `y >= 1`. Every start
`(x0, 1)` in that interval reaches `x = 1/2` with `y >= 1`. From that
worst entrance, `dy/dx` keeps the speed gap above `1/5` out to
`x = 2`, with `y(2) > 16`. For `x >= 2`,

    dx/dσ >= eps / 5

whenever `eps` is in `(0, 1/16]` and `r1` is in `[0, 1]`. The orbit
hits `x = (1/4)/eps`. The longest majorant, from `x = 0` at
`eps = 1/16`, equals `320`. Holding `y` at `1` stalls through `x = 1`.

Any interface `x = r1(1+theta)` that lands in `(0, 1/2]` is one of
these starts, including chart-O sequences `r1 -> 0`. This is not a
Lohner tube, not the height-section flag on `L = 1/n`, not
`eps > 1/16`, not an interface past `x = 1/2`, not complete first-hit
on chart O, C2, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line. The kill-line factor `sep * S_pre` on `(0, 1/2^16)` is [sep_spre](HILBERT16-SEP-SPRE.md).

Lean: [Hilbert16StageCInterface.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCInterface.lean).
Python: `omnibias.dynamics.stage_c_interface`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_interface.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_interface
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCInterface
```
