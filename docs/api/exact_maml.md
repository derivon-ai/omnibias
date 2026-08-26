# Exact MAML (09-16)

The inner step is Gauss–Newton / Newton. The meta-gradient is the
implicit function theorem on inner stationarity. That is the chain
rule. The founding bias collapse (`delta -> 0`) supplies exact HVPs.
Temperature collapse (`beta -> inf`, feasibility) does not appear.
Do not conflate the two.

Status is **shipped**. Not ImageNet few-shot. Not CCF
stretch. `imagenet_claim` and `stretch_claim` stay false.
`theorem_prover_verified` is not asserted.

Homes: `omnibias.core.exact_maml`,
`omnibias.{torch,jax}.optim_maml`.

::: omnibias.core.exact_maml
    options:
      show_root_heading: false
      heading_level: 3
