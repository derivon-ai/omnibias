# Jet-KAN (02-03)

Each edge is a multi-pack
`φ(u) = Σ_g c_g σ^(n_g)(α_g u + μ_g)`. Exactness is of the **model
jet**, not the target. The Kolmogorov-Arnold theorem does **not**
justify the architecture.

Directional jets use `compose_jet`; mixed partials use
`compose_jet_mv` (the same Faà di Bruno kernel as `mlp_jet_mv`).
Refinement in this wave is zero-weight pack birth plus optional
GrowableOMBU order growth on torch; full 03-13 pack birth/death stays
designed.

G1/G3/G5 are CI-gated. G2 (jet vs autodiff timing) is smoke-earned, not
in CI `all_passed`. Status is **gated**, not shipped. See theory spec
02-03.

## PyTorch module

::: omnibias.torch.architectures.jetkan
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.architectures.jetkan
    options:
      show_root_heading: false
      heading_level: 3
