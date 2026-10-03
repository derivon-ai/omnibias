# Hilbert XVI: cubic `(V,h)` Lohner orbit and `V=-1/4` first-hit

This companion continues [the comparison-bootstrap integral](HILBERT16-TH-INTEGRAL.md).
The cubic truncation of the first-root normal field is the polynomial

    Vdot = f(V) + h g(V),   hdot = -V h,
    f(V) = -L eps^3 + lam1 eps^2 V - eps V^2 + (eps/3) V^3,
    g(V) = -1 + nu (V-1).

On the declared matching compact `eps = 1/16`, `lambda1 = -2`,
`L = 9/25`, `V(0) = -eps`, `h(0) = 4 eps^3`, a QR-Lohner prefix keeps
`V < 0`, `h > h(0)`, and `T-h < 9 eps`. Independently,
`certify_stopped_event` proves a unique transverse first hit of the
declared section `V = -1/4` at time in `[31, 31.5]`. That section is
not the physical `E_sigma` large first-root section, and the result is
not uniform in `r1 -> 0`. Not G1 and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Matching-chart first-hit of `E_out` on `L in {9/25, 1/16, 0}` is
  [HILBERT16-E-OUT-SECTION.md](HILBERT16-E-OUT-SECTION.md).
- GRAZING `E_sigma` first-hit on the `V ≈ 1` chart.
- Uniform first-hit as `eps -> 0`.
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16VhOrbit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16VhOrbit.lean).
Python: `omnibias.dynamics.vh_orbit`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_vh_orbit.py -q
uv run --no-sync python -m benchmarks.hilbert16_vh_orbit
lake build OmnibiasAnalytic.Dynamics.Hilbert16VhOrbit
```
