# omnibias-curvature

Parameter curvature primitives.

`omnibias.curvature` supplies closed-form parameter derivatives for supported
model structures and curvature-based optimization utilities.

- `one_layer_param_grad`, `one_layer_param_hessian`: one-layer parameter derivatives.
- `mse_loss_hessian`, `mse_gauss_newton_fisher`: loss curvature constructions.
- `kfac_kron_factors`: factorized curvature information.
- `damped_solve`, `regularized_solve`: stabilized linear algebra.

Distinguish an exact Hessian from a Gauss–Newton or factorized approximation.
The one-layer formulas do not automatically cover arbitrary deep framework
modules. Validate conditioning and gradient behavior before using a curvature
step in a PINN training loop.

Install this distribution with `pip install omnibias-curvature`; select its
backend extras when needed. See [guarantees](../guarantees.md).
