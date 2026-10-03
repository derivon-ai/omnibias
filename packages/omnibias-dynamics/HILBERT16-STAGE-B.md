# Hilbert XVI: kill-line Stage-B height inflation

This companion continues [the tracked product](HILBERT16-ENTRY-EXIT-LEADING.md).
On `lambda1 = -2` the slow-line midpoint is `rstar = 1` and
`r1 = 1 - sep/2`. For `sep in (0, 1]` the incoming wall lies in
`[1/2, 1]`. After Stage A, height inflation has `dx/dy = eps k / x`
with `k = 1`, independently of height. Interval Picard on

    eps in [0, 1/16],  x-guess [1/4, 4/3]

includes the image of the start box `[1/2, 1]` and encloses
`|Delta x| < 1/3`. The tracked exponent `2 - 2 alpha` with `alpha = 2 eps`
stays positive on this compact, so the sep-power of the tracked product
is at most 1 for `sep in (0, 1]`.

This is a sealed Stage-B displacement on the kill line, not Stage A/C,
not a C2 remainder, not first-hit, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Stage A `dx_e / d kappa`; the wall identities are
  [HILBERT16-STAGE-A.md](HILBERT16-STAGE-A.md), the `chi_b`
  threshold is [HILBERT16-CHI-B.md](HILBERT16-CHI-B.md), the
  leading `dx_e` factors are
  [HILBERT16-DX-E-LEADING.md](HILBERT16-DX-E-LEADING.md), and the
  uniform-in-`chi` bound is
  [HILBERT16-DX-E-UNIF.md](HILBERT16-DX-E-UNIF.md).
- Stage C first-root outgoing with uniform `a_min` is
  [HILBERT16-STAGE-C.md](HILBERT16-STAGE-C.md). The Stage-C exit
  energy is [HILBERT16-STAGE-C-EXIT.md](HILBERT16-STAGE-C-EXIT.md).
  Leading `T_h > 1/2` is
  [HILBERT16-STAGE-C-TH.md](HILBERT16-STAGE-C-TH.md).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.

Lean: [Hilbert16StageB.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageB.lean).
Python: `omnibias.dynamics.stage_b`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_b.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_b
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageB
```
