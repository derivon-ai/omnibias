# Spectral design (01-07)

Pack order is a **band selector**: differentiation multiplies the
Fourier transform by `(i ξ)^n`. This is a design calculator
(`BandPlan`, `peak_frequency`), not a Littlewood-Paley completeness
claim. Frames live in [frames.md](frames.md): `sigma'` is not
admissible.

G1–G2 are CI-gated. G3 (steps vs `MscaleMLP` on the spectral-bias arm)
is smoke-attempted and **not** in CI `all_passed`. Status is **gated**,
not shipped. See theory spec 01-07.

## Core algebra

::: omnibias.core.spectral_design
    options:
      show_root_heading: false
      heading_level: 3
