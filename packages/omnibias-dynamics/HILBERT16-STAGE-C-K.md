# Hilbert XVI: kill-line Stage-C C=2 tight `T(h)` ratio

This companion continues [the Stage-C C=2 `T(h)` majorant](HILBERT16-STAGE-C-INT.md)
and [the first-root note](HILBERT16-ROOT-SADDLE.md) §5. On `lambda1 = -2`
the integrating-factor exponent at `hmax = 1` is `6 eps log(1/eps)`.
The map `eps |-> eps log(1/eps)` increases on `(0, 1/16]`, so the
maximum is the compact edge. Interval `(3/8) ln(16)` encloses that
edge exponent and `exp` of it is `< 3`. Times the sealed
`T_e/eps^2 < 1` this is a prefactor `< 6`. The slope `16/7` sits
below `3`. Therefore

    T(h) <= 6 (eps^2 + h)

uniformly on the Stage-C compact, sharpening the coefficient-64
majorant. This is a sealed ratio bound, not `T-h = O(eps)`, first-hit,
C2, `dx_e` off the kill line, G1, or Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Outgoing height-section first-hit on chart O.
- A height-section first-hit of the Stage-C `T(h)` orbit
  (the `T-h < 1` margin is [the Stage-C bootstrap note](HILBERT16-STAGE-C-BOOT.md);
  continuation to `hmax` remains).
- `sep > 0` C2 of `log D'`.
- A uniform Lohner first-hit for every `eps`.
- Complete first-hit and Stage B/C continuation off the kill line.
- `dx_e` off the kill line, and the remaining `sep in (0, 1/2^16)`.

Lean: [Hilbert16StageCK.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16StageCK.lean).
Python: `omnibias.dynamics.stage_c_k`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_stage_c_k.py -q
uv run --no-sync python -m benchmarks.hilbert16_stage_c_k
lake build OmnibiasAnalytic.Dynamics.Hilbert16StageCK
```
