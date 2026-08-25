# Adaptive pack refinement (03-13)

A bank of tempered packs can **birth** (new pack, `c = 0`), **grow**
(raise order by adding a zero-weight sibling, or call
`GrowableOMBU.grow` for literal `K`), and **die** (remove a pack whose
contribution is below a threshold). Birth and growth are bit-identical
zero-perturbation. Death is bounded: the report carries
`death_perturbation`. Scale `alpha` is the founding scaling law
`sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)`, a third axis — not
bias collapse and not temperature collapse.

Status is **gated**, not shipped. G1–G6 are CI-gated. G4 (10x vs a
matched-count fixed bank on the named boundary layer, five `eps`) is
**earned** through `propose_refinement`, not a hand-placed pack. See
theory spec 03-13. Singularity / scale-flow indicators use Domb-Sykes
and jet-ratio estimators; they are not the full 03-10 Padé tracker or
the 03-07 RG beta-function.

## Core algebra

::: omnibias.core.refine
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.refine
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.refine
    options:
      show_root_heading: false
      heading_level: 3
