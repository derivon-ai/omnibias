# Hilbert XVI: shrinking-`eps` matching-chart `E_out` pack

This companion continues [matching-chart `E_out`](HILBERT16-E-OUT-SECTION.md).
On the kill line `L = 0`, `lambda1 = -2`, `rho = 1/4`, `C = 2`, matching
`V(0) = -eps`, `h(0) = 4 eps^3`, `certify_stopped_event` hits `E_out`
for `eps = 1/n` with `n in {16, 20, 25}` inside the declared majorant
horizon `T = n^2 / 8`. A short horizon at `n = 16` does not certify.
The kill-line matching slope is the exact identity

    Vdot + 3 eps^3 + (13/3) eps^4 + 4 eps^5 = 0.

This is a finite shrinking pack, not a uniform-in-`eps` theorem, not
GRAZING `E_sigma`, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- GRAZING `E_sigma` first-hit on the `V ≈ 1` chart.
- A uniform-in-`eps` first-hit theorem (this note is three squares
  `n in {16, 20, 25}`, not every `n` and not a limit; the comparison
  speed bound is [HILBERT16-E-OUT-SPEED.md](HILBERT16-E-OUT-SPEED.md)).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16EOutEps.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16EOutEps.lean).
Python: `omnibias.dynamics.e_out_eps`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_out_eps.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_out_eps
lake build OmnibiasAnalytic.Dynamics.Hilbert16EOutEps
```
