# Jet-token transformer and jet distillation (09-02 / 09-19)

Tokens are N-jets. A mix is `compose_jet`, not a softmax of dots.
A student matches a teacher's N-jet, not logits. The founding bias
collapse (`delta -> 0`) supplies `sigma^(n)`. Temperature collapse
(`beta -> inf`, feasibility) does not appear in the default mix.
Do not conflate the two.

Status is **shipped** for 09-02 (G1–G3 CI). 09-19 stays gated
until its own gates are flipped. Exactness is of the **model** jet,
not the target. Not ImageNet, not Jet-KAN, not CCF stretch.
`imagenet_claim` stays false. `theorem_prover_verified` is not
asserted.

Homes: `omnibias.core.jet_token`,
`omnibias.{torch,jax}.architectures.jet_token`,
`omnibias.{torch,jax}.jet_distill`.

::: omnibias.core.jet_token
    options:
      show_root_heading: false
      heading_level: 3
