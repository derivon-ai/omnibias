# Recommended trainer stack (08-01)

One callable for the 08-01 recommended stack on a **one-layer**
Riccati MSE loss: closed-form Newton direction, 03-12 jet line search,
optional 08-04 Kantorovich accept on `φ'(s) = 0`, and 08-06 sharpness
as extra cubic damping. 08-02 is skipped (`used_composed` is false);
it is a two-layer slice rule.

An indefinite Hessian is damped by at least `|λ_min|` so the Newton
direction is a descent. A raw length-1 Newton step can still overshoot;
line search is required. Not a global min. Not CCF stretch.

Homes: `omnibias.core.train_stack`,
`omnibias.{torch,jax}.train_stack`.

::: omnibias.core.train_stack
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.train_stack
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.train_stack
    options:
      show_root_heading: false
      heading_level: 3
