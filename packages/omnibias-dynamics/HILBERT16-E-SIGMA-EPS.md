# Hilbert XVI: shrinking-`eps` aligned GRAZING `E_sigma` pack

This companion continues [L-pack wall-span cover](HILBERT16-E-SIGMA-PACK.md).
On the kill line `L = 0`, `lambda1 = -2`, the aligned restart
`(V, h) = (1/4, 1/40)` lies in the certified GRAZING-from-`V=0`
`V=1/4` return box at `eps = 1/n` for `n in {16, 20, 25}`. From
that point, `certify_stopped_event` hits GRAZING `E_sigma` uniquely
and transversely inside the declared compact `T = 10`. A short
horizon at `n = 16` does not certify. The GRAZING start `V=0` still
excludes `E_sigma` on this compact horizon.

This is a finite shrinking pack of an aligned-point restart, not a
wall-span cover at every `n`, not a single Lohner run from `V=0`,
not a uniform-in-`eps` theorem, not G1, and not Hilbert XVI.

    g1_passed = false
    g4_passed = false
    full_graphic_cyclicity_proved = false
    full_hilbert16_solved = false

## 1. What remains for G1

- Lohner `E_sigma` from the GRAZING start `V=0` in one run
  ([HILBERT16-E-SIGMA-ONESHOT.md](HILBERT16-E-SIGMA-ONESHOT.md)).
- A wall-span cover at every `n` (this note is the aligned point).
- A uniform Lohner first-hit for every `eps`.
- Physical C2 with `Z_x` and `sep > 0`.
- Complete first-hit and Stage B/C continuation.

Lean: [Hilbert16ESigmaEps.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ESigmaEps.lean).
Python: `omnibias.dynamics.e_sigma_eps`.

## 2. Reproduction

```bash
uv run --no-sync python -m pytest \
  packages/omnibias-dynamics/tests/test_e_sigma_eps.py -q
uv run --no-sync python -m benchmarks.hilbert16_e_sigma_eps
lake build OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEps
```
