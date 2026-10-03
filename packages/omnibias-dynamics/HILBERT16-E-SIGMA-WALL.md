# Hilbert XVI: orbit-aligned GRAZING `E_sigma` from the `V=1/4` wall box

This companion continues [incoming `V=1/4` first-hit](HILBERT16-E-SIGMA-IN.md).
The GRAZING reverse cubic from `V=0`, `h=4 eps^3` has a certified
first-hit of `V=1/4`. On `L in {9/25, 1/16, 0}` that return box
contains the rational point `(V, h)=(1/4, 1/40)`. From that aligned
restart, `certify_stopped_event` hits GRAZING `E_sigma` uniquely and
transversely. A short horizon does not certify. The GRAZING start
`V=0` still excludes `E_sigma` on this compact horizon.

This is a two-segment chain (certified wall from `V=0`, then a point
in that wall box to `E_sigma`), not enclosure continuation of the
whole `h`-box, not a single Lohner run from `V=0`, not uniform in
`eps`, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Lohner `E_sigma` from the GRAZING start `V=0` in one run (this
  note restarts at a point inside the `V=1/4` box).
- Enclosure continuation of the whole wall `h`-box (a declared
  interior sub-box is [HILBERT16-E-SIGMA-BOX.md](HILBERT16-E-SIGMA-BOX.md)).
- A uniform Lohner first-hit for every `eps`.
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaWall.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaWall.lean).
Python: `omnibias.dynamics.e_sigma_wall`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_wall.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_wall
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaWall
```
