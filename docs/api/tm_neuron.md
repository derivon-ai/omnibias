# Taylor-model neuron (09-05)

The hidden state is a `TaylorModel`: affine image plus `sigma` via
the closed-form tower. The founding bias collapse (`delta -> 0`)
supplies the polynomial coefficients. Temperature collapse
(`beta -> inf`, feasibility) does not appear. Do not conflate the
two.

Status is **gated**, not shipped. Sound enclosure of a shallow cell.
Not a deep-net certificate. Not ImageNet. Not CCF stretch.
`theorem_prover_verified` is not asserted. Navier–Stokes stays
external.

Home: `omnibias.core.verified.tm_neuron`.

::: omnibias.core.verified.tm_neuron
    options:
      show_root_heading: false
      heading_level: 3
