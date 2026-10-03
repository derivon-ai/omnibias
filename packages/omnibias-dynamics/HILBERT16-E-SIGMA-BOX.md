# Hilbert XVI: wall-box `h`-interval GRAZING `E_sigma` cover

This companion continues [orbit-aligned wall `E_sigma`](HILBERT16-E-SIGMA-WALL.md).
The GRAZING reverse cubic from `V=0`, `h=4 eps^3` has a certified
first-hit of `V=1/4`. On `L in {9/25, 1/16, 0}` those return boxes
all contain the rational interval `[1/50, 4/125]`. Split into twelve
equal slabs of width `1/1000`, `certify_stopped_event` hits GRAZING
`E_sigma` uniquely and transversely on every slab at `L=0`, and on
the aligned slab containing `h=1/40` at `L in {9/25, 1/16}`. A
single slab over the whole interval is unresolved. The GRAZING start
`V=0` still excludes `E_sigma` on this compact horizon.

This is enclosure continuation of a declared rational sub-box of the
wall intersection, not the whole wall `h`-interval, not a single
Lohner run from `V=0`, not uniform in `eps`, not G1, and not Hilbert
XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Lohner `E_sigma` from the GRAZING start `V=0` in one run (this
  note restarts at `V=1/4` on a positive-width `h`-box).
- Enclosure continuation of the whole wall `h`-interval (the `L=0`
  wall span is [HILBERT16-E-SIGMA-SPAN.md](HILBERT16-E-SIGMA-SPAN.md);
  the remaining L-pack walls are
  [HILBERT16-E-SIGMA-PACK.md](HILBERT16-E-SIGMA-PACK.md)).
- A uniform Lohner first-hit for every `eps`.
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaBox.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaBox.lean).
Python: `omnibias.dynamics.e_sigma_box`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_box.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_box
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaBox
```
