# Hilbert XVI: one-shot Lohner GRAZING `E_sigma` from `V=0`

This companion continues [shrinking-eps aligned pack](HILBERT16-E-SIGMA-EPS.md).
The reverse cubic from the GRAZING start `V=0`, `h=4 eps^3`, with
`step=1/4` and `max_steps=280`, has unique transverse first-hit of
GRAZING `E_sigma` on `L in {9/25, 1/16, 0}` at `eps=1/16`. A short
horizon of 200 steps does not certify. Coarser `step=1/2` wraps.

This is a single `certify_stopped_event` from `V=0` at one `eps`,
not a uniform-in-`eps` theorem, not `Z_x` C2, not G1, and not
Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- A uniform Lohner first-hit for every `eps` (a finite one-shot
  shrinking pack is [HILBERT16-E-SIGMA-ONESHOT-EPS.md](HILBERT16-E-SIGMA-ONESHOT-EPS.md)).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaOneshot.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaOneshot.lean).
Python: `omnibias.dynamics.e_sigma_oneshot`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_oneshot.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_oneshot
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaOneshot
```
