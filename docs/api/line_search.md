# Exact jet line search (03-12)

A directional Taylor jet of `phi(s) = L(theta + s d)` turns line search
into root-finding on a known polynomial. The model is exact to order
`N` from one restriction; a Lagrange remainder radius is sound when a
bound on `|phi^(N+1)|` is supplied; `verify=True` is the never-worse
backstop.

Status is **gated**, not shipped. G1/G2/G3/G6 are CI-gated. G4
(target-loss step count vs strong Wolfe) is **leftover-recorded**
unearned (leftover #47; ratio `1.83`, need `2x`; Wolfe misses the
stiffest seed). G5 reports the order×depth crossover versus a
four-trial Wolfe budget (favourable at `N=2`, over budget from `N=4`)
and is **leftover-recorded** (leftover #48), **not** in CI
`all_passed`. The win is a constant factor in a specific regime, not
an asymptotic one. Jets come from the founding bias collapse
(`delta -> 0`). No temperature collapse appears. See theory spec 03-12.

`taylor_line_min` (order 2/3, no certified radius) is unchanged.

## Core algebra

::: omnibias.core.line_search
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.line_search
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.line_search
    options:
      show_root_heading: false
      heading_level: 3
