# Exact score matching (09-21)

Hyvärinen / denoising score matching on an OMBU score whose
divergence is Hutchinson-free. For `s(x) = -x` the divergence is
`-1`. For a tower score, `div` is a `sigma''` contraction from
founding bias collapse (`delta -> 0`). Temperature collapse
(`beta -> inf`, feasibility) does not appear.

This is a training objective. Exact CNF `div` (`integrate_cnf`) is
prior art and is not claimed as new. Status is **shipped**.
Not ImageNet. Not CCF stretch.

Homes: `omnibias.core.score_matching`,
`omnibias.score.{torch,jax}.score_matching`,
`omnibias.score.flow.{torch,jax}.score_matching`.

::: omnibias.core.score_matching
    options:
      show_root_heading: false
      heading_level: 3
