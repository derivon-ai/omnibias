# Composed-curvature joint Newton (08-02)

The order-2 chain rule `(f circ g)'' = f''(g)(g')^2 + f'(g)g''` is the
exact coupling between consecutive layers. A Newton step on the joint
block `(W_{ell-1}, W_ell)` can leave a critical point that is a minimum
of the current-layer slice and a saddle of the pair.

Status is **gated**, not shipped. G1–G4 are CI-gated. Escape is from a
*slice* critical point, locally. Not a global min of a deep nest, not
CCF stretch, and not Navier–Stokes regularity. `sigma''` comes from the
founding bias collapse (`delta -> 0`). No temperature collapse. See
theory spec 08-02.

`n_directions >= n_params` raises unless `allow_full=True`. The
subspace is `k << P`; a negative mode orthogonal to it is invisible.

## Core algebra

::: omnibias.core.composed_curvature
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.optim_composed
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.optim_composed
    options:
      show_root_heading: false
      heading_level: 3
