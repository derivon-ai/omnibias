# Hilbert XVI: matching-chart `E_out` first-hit

This companion continues [the cubic `(V,h)` Lohner orbit](HILBERT16-VH-ORBIT.md).
Under the first-root matching embedding `V = -eps x` with `nu = eps`, the
large physical section `x = rho/nu` maps to `V = -rho`. The height-corrected
matching-chart polynomial is

    E_out(V,h) = V + rho + nu rho h + C nu^2 rho h^2.

On the declared cubic compact `eps = 1/16`, `lambda1 = -2`,
`V(0) = -eps`, `h(0) = 4 eps^3`, `rho = 1/4`, `C = 2`,
`certify_stopped_event` proves a unique transverse first hit of `E_out`
for `L in {9/25, 1/16, 0}`, including the kill limit `L = 0`, at time
in `[31, 31.5]`. The GRAZING / HEIGHT-COMPARISON polynomial

    E_sigma(V,h) = V-1+sigma rho h+nu rho^2 h^2+C nu^2 sigma rho h^2

lives near `V = 1` and is excluded on this outgoing orbit, including at
`L = 0`. The `nu/C` terms in `E_out` are the HEIGHT-COMPARISON height
corrections transplanted to the matching chart; the leading term `V+rho`
is the exact embedding. Not uniform in `eps -> 0`, not G1, and not
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- GRAZING `E_sigma` first-hit on the `V ≈ 1` chart (this note hits
  matching-chart `E_out`, the image of `x = rho/nu` under `V = -eps x`).
  Uniform first-hit as `eps -> 0` (this note is one matching `eps = 1/16`;
  the finite shrinking pack is [HILBERT16-E-OUT-EPS.md](HILBERT16-E-OUT-EPS.md);
  the comparison speed bound is [HILBERT16-E-OUT-SPEED.md](HILBERT16-E-OUT-SPEED.md)).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16EOutSection.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16EOutSection.lean).
Python: `omnibias.dynamics.e_out_section`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_out_section.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_out_section
lake build OmnibiasAnalytic.Dynamics.Hilbert16EOutSection
```
