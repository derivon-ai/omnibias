# Hilbert XVI: incoming GRAZING-chart `V=1/4` first-hit

This companion continues [incoming GRAZING comparison speed](HILBERT16-E-SIGMA-SPEED.md).
GRAZING starts at `V = 0`, `h = 4 eps^3`. The reversed kill-line cubic
(`L in {9/25, 1/16, 0}`, `lambda1 = -2`, `nu = eps`) has unique
transverse first-hit of the declared wall `V = 1/4`. A short horizon
does not certify. Reverse `hdot = V h` and the matching reverse slope
`Vdot_rev = 4 eps^3 (1 + eps)` at `V = 0` are exact identities.

Lohner wrapping still refuses `E_sigma = V-1+sigma rho h+...` near
`V = 1`. This is an incoming `V`-wall first-hit, not GRAZING first-hit,
not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Certified GRAZING `E_sigma` first-hit on the `V ≈ 1` chart (this note
  hits the prefix wall `V = 1/4`; the declared-point `E_sigma` hit is
  [HILBERT16-E-SIGMA-HIT.md](HILBERT16-E-SIGMA-HIT.md)).
- A uniform Lohner first-hit for every `eps` on the outgoing matching
  chart.
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaIn.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaIn.lean).
Python: `omnibias.dynamics.e_sigma_in`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_in.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_in
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaIn
```
