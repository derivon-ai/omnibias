# Pack-MoE (09-07)

Experts are OMBU packs. The router is slab mass (`integral` or
`band`), not a free softmax. The founding bias collapse
(`delta -> 0`) may appear only in an expert collapse head.
Temperature collapse (`beta -> inf`, feasibility) is recorded when
`beta != 1`. Do not conflate the two.

Status is **shipped**. Softmax raises unless
`allow_softmax`. Not ImageNet MoE. Not a 05-02 LightGBM reversal.
Not CCF stretch.

Homes: `omnibias.core.pack_moe`,
`omnibias.{torch,jax}.architectures.pack_moe`.

::: omnibias.core.pack_moe
    options:
      show_root_heading: false
      heading_level: 3
