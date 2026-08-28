# Antisymmetric Slater sampler + Bloch mixed partials

`omnibias.ferminet.antisymmetric` is a Gaussian Slater `log|det M|` on the
existing sampling `log_abs_psi_fn` contract, with a closed-form Laplacian of
`log|det M|`. `omnibias.ferminet.bloch` reads mixed `x`–twist partials from
the existing multivariate jet kernel (real twist observable, not a complex
Bloch phase).

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.ferminet.antisymmetric import gaussian_slater_log_abs_psi

params = {
    "centers": jnp.array([[0.0], [1.0]]),
    "alphas": jnp.array([0.5, 0.5]),
}
logabs = gaussian_slater_log_abs_psi(params, jnp.array([0.2, 0.8]))
assert jnp.isfinite(logabs)
```

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.ferminet.bloch import bloch_twist_mixed_partials

partials = bloch_twist_mixed_partials(jnp.array(0.3), jnp.array(0.1), order=2)
assert jnp.isfinite(partials[(1, 1)])
```

## API

::: omnibias.ferminet.antisymmetric
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.ferminet.bloch
    options:
      show_root_heading: false
      heading_level: 3
