# Hilbert XVI: uniform comparison GRAZING `E_sigma` on `eps in [0, 1/8]`

This companion continues [comparison from `V=0`](HILBERT16-E-SIGMA-FROM0.md).
The Taylor height majorant is pole-free after cancellation:

    I = V^2/den - (1+eps) V^2/(2 den^2) + (1+eps) eps V^3/(3 den^3),

    den = 1+eps-eps V,   h_up = 4 eps^3 + I.

At `eps = 0` this is `V^2/2`. On `V* = 6/5` and `eps in [0, 1/8]`,
eight equal Interval slabs each have `E_sigma(V*, h_up) > 0` and
`dE/dV >= 1 - rho V*/den > 0`. A single slab over the whole interval
wraps and does not separate `0`. `V* = 1` does not change sign.

This is a uniform comparison tube, not a Lohner event for every
`eps`, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Lohner `E_sigma` from the GRAZING start `V = 0` (a wall-box
  `h`-interval cover at `V=1/4` is
  [HILBERT16-E-SIGMA-BOX.md](HILBERT16-E-SIGMA-BOX.md)).
- A uniform Lohner first-hit for every `eps` (this note is a
  comparison tube, not `certify_stopped_event` on every square).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaUnif.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaUnif.lean).
Python: `omnibias.dynamics.e_sigma_unif`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_unif.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_unif
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaUnif
```
