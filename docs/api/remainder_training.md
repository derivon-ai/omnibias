# Remainder training (09-18)

The loss is the Taylor remainder `R_N = f - T_N` after order `N`,
not a rewrite of the 03-10 diagnostic or the 03-13 architecture
search. An optional birth hook calls the 03-13 residual indicator
when `||R_N||` stays large; that hook is not this spec. The
founding bias collapse (`delta -> 0`) supplies the jet of `T_N`.
Temperature collapse (`beta -> inf`, feasibility) does not appear.
Do not conflate the two.

Status is **gated**, not shipped. `R_N` is of the model (or a named
analytic target), not of an unknown PDE solution. Not CCF stretch.

Homes: `omnibias.core.remainder_train`,
`omnibias.{torch,jax}.optim_remainder`.

::: omnibias.core.remainder_train
    options:
      show_root_heading: false
      heading_level: 3
