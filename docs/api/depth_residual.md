# Depth-causal residual (08-05)

After each hidden layer a linear decode is a field. The same local
PDE residual used at the last readout is formed from a closed-form
`layer_jet` / `jet_to_tower` (bias collapse, `delta -> 0`), optionally
wrapped by a hard Dirichlet factor, and a damped Gauss–Newton step
updates only that layer. Later layers see the corrected activations.

This marches the residual in **network depth**. It is not time marching
(`omnibias.pinn.train.march`) and not the 08-03 proxy residual. Status
is **gated**, not shipped. G1–G3 are CI-gated. Local GN is greedy, not
a global min, and not CCF stretch. Hilbert / nonlocal operators are
out of scope. Continuum Navier–Stokes regularity is not a claim.

## Core algebra

::: omnibias.pinn.train._core.depth_residual
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.pinn.train.torch.depth_residual
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.pinn.train.jax.depth_residual
    options:
      show_root_heading: false
      heading_level: 3
