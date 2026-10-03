# Hilbert XVI: declared-point GRAZING `E_sigma` first-hit

This companion continues [incoming `V=1/4` first-hit](HILBERT16-E-SIGMA-IN.md).
On the reverse cubic at `eps = 1/16`, `lambda1 = -2`, from the declared
incoming point `(V, h) = (3/4, 1/4)`, `certify_stopped_event` hits

    E_sigma = V-1+sigma rho h+nu rho^2 h^2+C nu^2 sigma rho h^2

with `sigma = -1`, `rho = 1/4`, `C = 2`, uniquely and transversely, for
`L in {9/25, 1/16, 0}`. A short horizon does not certify. The GRAZING
start `V = 0`, `h = 4 eps^3` still excludes `E_sigma` on this compact
horizon.

This is a declared incoming-point first-hit, not the GRAZING band from
`V = 0`, not uniform in `eps`, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Certified GRAZING `E_sigma` from the GRAZING start `V = 0` via
  Lohner (this note restarts at `(3/4, 1/4)`; an orbit-aligned
  restart inside the `V=1/4` box is
  [HILBERT16-E-SIGMA-WALL.md](HILBERT16-E-SIGMA-WALL.md)).
- A uniform Lohner first-hit for every `eps`.
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaHit.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaHit.lean).
Python: `omnibias.dynamics.e_sigma_hit`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_hit.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_hit
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaHit
```
