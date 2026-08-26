# Bias scan (transverse convolution)

A scan shares one pack template across a bank of bias offsets on the
transverse coordinate `z = w · x`. There is no pixel grid. Equivariance is
along `w` only, and on a finite non-periodic bank it is an **interior
lattice shift** of the response (`R(z+Δ)[..., :-1]` vs `R(z)[..., 1:]`),
not a circular wrap of `tanh'` (which is not periodic).

The template's internal limit is founding bias collapse (`delta -> 0`).
Soft-argmax sharpness `gamma` is a softmax readout; driving it to infinity
would be temperature collapse (`beta -> inf`), a different limit. See
theory spec 01-02. G1–G4 are CI-gated (`all_passed`); G4 is a no-grid
MAE win against the voxelize-then-`cmbConv1d` pipeline at equal-or-lower
warmed-up wall, not a first-forward. Status is **shipped**.

## Core algebra

::: omnibias.core.scan
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch module

::: omnibias.torch.scan
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.scan
    options:
      show_root_heading: false
      heading_level: 3
