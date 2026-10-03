# Hilbert XVI: kill-line Stage-C comparison first-hit of E_out

This companion continues [the Stage-C height-section hit of `{h=1}`](HILBERT16-STAGE-C-HIT.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the Stage-C start box has `V in [-4 eps/3, -eps/4]`, so the leading
gap `rho - 4 eps/3` is `1/4 - 1/12 = 1/6 > 0`. Height corrections
still leave `E_out > 0` there. Along the sealed `T_h > 1/2`
continuation, `|V_h| = T_h/|V| >= (1/2)/2 = 1/4` on the rectangle
`|V| <= 2`, while the `E_out` height-correction derivative is at most
`1/64 + 1/256 = 5/256`. Interval wrapping keeps `dE_out/dh < 0` and
the height to `T = rho^2/2` below `1/8 < 1`. This is a unique
comparison first-hit of the selected large outgoing physical section
`E_out` from Stage-C start, not a Lohner orbit, not chart O, C2,
`dx_e` off the kill line, G1, or Hilbert XVI. Matching-chart Lohner
`E_out` from the GRAZING start is [HILBERT16-E-OUT-SECTION.md](HILBERT16-E-OUT-SECTION.md).

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- A Lohner first-hit of `E_out` from Stage-C start is
  [HILBERT16-STAGE-C-ONESHOT.md](HILBERT16-STAGE-C-ONESHOT.md).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCSec.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCSec.lean).
Python: `omnibias.dynamics.stage_c_sec`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_sec.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_sec
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCSec
```
