# Differentiable morphology (03-05)

Dilation is max-plus convolution. `logsumexp_beta` is the homotopy.
Status is **gated**, not shipped. G1–G6 are CI-gated.

`beta -> inf` is **temperature collapse** (feasibility), a soft max
hardening to a hard max. Pack structuring elements come from the
**founding bias collapse** (`delta -> 0`). Do not conflate the two.
The gap `compositions * log(N) / beta` is worst-case and tight only
when window values coincide. Soft dilation is a conservative upper
bound. Not a seventh `OperatorBlock` role.

Home: `omnibias.shape.morphology`. No new package.

## API

::: omnibias.shape.morphology
    options:
      show_root_heading: false
      heading_level: 3
