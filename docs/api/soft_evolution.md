# Soft-population evolution (03-01)

Four evolutionary variants that reuse omnibias structure instead of
treating the model as a black box. Status is **gated**, not shipped.
G1–G6 are CI-gated.

Selection `w = softmax(-beta E)` is **temperature collapse**
(`beta -> inf`, feasibility). Pack genes mutate the geometry of the
**founding bias collapse** (`delta -> 0` to `sigma^(K-1)`). Do not
conflate the two. Evolutionary algorithms are not new. The
contributions are the closed-form `log(P)/beta` selection gap,
geometry-aware mutation, exact-curvature polish with full evaluation
accounting, and a certify-loop stop. Nothing here is a P = NP claim,
and G2 targets **parity** with CMA-ES, not superiority.

Home: `omnibias.discrete.evolution`. No new package.

## API

::: omnibias.discrete.evolution
    options:
      show_root_heading: false
      heading_level: 3
