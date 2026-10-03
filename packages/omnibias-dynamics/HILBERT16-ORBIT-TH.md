# Hilbert XVI: actual-versus-comparison `T_h` gap

This companion continues [the height-mix note](HILBERT16-HEIGHT-MIX.md).
The first-root height continuation uses `T_h = q/h + k`. The comparison
is

    T_h = 1 + C eps T/h + C eps^3 / h.

Their difference splits exactly as

    (q/h + k) - (1 + C eps T/h + C eps^3 / h)
        = (k - 1) + (q - C eps (T + eps^2)) / h.

When `q` touches the comparison flux `C eps (T + eps^2)`, the gap is
`k - 1`. On `C = 0`, `A = 1`, `|k-1| <= 3 nu` for `|v| <= 2` and
`nu <= 1/8`.

This is the algebraic gap, not a validated integral of `T-h` along an
actual `(V,h)` orbit, not height-section first-hit, G1, or Hilbert XVI.
The comparison-bootstrap integral is [HILBERT16-TH-INTEGRAL.md](HILBERT16-TH-INTEGRAL.md).

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Integrated `T-h = O(epsilon)` along a validated `(V,h)` orbit
  (the comparison-bootstrap majorant is [HILBERT16-TH-INTEGRAL.md](HILBERT16-TH-INTEGRAL.md);
  this note is the pointwise gap).
- Height-section first-hit of the selected large first-root section
  (the transversal event).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16OrbitTh.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16OrbitTh.lean).
Python: `omnibias.dynamics.orbit_th`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_orbit_th.py -q
uv run --no-sync python -m benchmarks.hilbert16_orbit_th
lake build OmnibiasAnalytic.Dynamics.Hilbert16OrbitTh
```
