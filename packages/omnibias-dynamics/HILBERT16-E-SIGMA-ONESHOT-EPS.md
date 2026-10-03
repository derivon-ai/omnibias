# Hilbert XVI: shrinking-`eps` one-shot Lohner GRAZING `E_sigma` pack

This companion continues [one-shot from `V=0`](HILBERT16-E-SIGMA-ONESHOT.md).
On the kill line `L = 0`, the reverse cubic from GRAZING `V=0`,
`h = 4 eps^3`, with `step = 1/4`, has unique transverse first-hit of
GRAZING `E_sigma` at `eps = 1/n` for `n in {16, 20, 25}` inside the
declared horizons `T = 70, 100, 250`. A short horizon at `n = 16`
does not certify.

This is a finite shrinking pack of one-shot runs, not a
uniform-in-`eps` theorem, not `Z_x` C2, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- A uniform Lohner first-hit for every `eps` (this note is three
  squares, not every `n` and not a limit). The compact aligned
  `[1/25, 1/16]` cover is
  [HILBERT16-E-SIGMA-EPS-SPAN.md](HILBERT16-E-SIGMA-EPS-SPAN.md).
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaOneshotEps.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaOneshotEps.lean).
Python: `omnibias.dynamics.e_sigma_oneshot_eps`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_oneshot_eps.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_oneshot_eps
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaOneshotEps
```
