# OMBU frames (01-06)

`sigma'` is **not** admissible: `admissibility_constant(..., n=1)` returns
`None` and never guesses. Frames are not orthonormal and not compactly
supported; there is no O(N) fast transform. Pack order remains a **band
selector** ([spectral_design.md](spectral_design.md)), not a
Littlewood-Paley completeness claim. Atoms come from founding
`delta -> 0`.

G1–G3 are CI-gated. G4 (denoising skill) is smoke-earned, not in CI
`all_passed`. Status is **gated**, not shipped. See theory spec 01-06.

## Core algebra

::: omnibias.core.frames
    options:
      show_root_heading: false
      heading_level: 3
