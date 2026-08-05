# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""omnibias JAX quickstart.

Run:

    pip install omnibias-jax
    python docs/examples/quickstart_jax.py

Shows the closed-form value / gradient / Hessian of a one-layer scalar
field and checks the Laplacian against ``jax.hessian``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from omnibias.jax import (  # noqa: E402  (after x64 config)
    get_activation,
    neural_field_laplacian,
    neural_field_value_grad_hessian,
)


def main() -> None:
    key = jax.random.PRNGKey(0)
    H, D = 6, 3
    kW, kb, kc = jax.random.split(key, 3)

    W = jax.random.normal(kW, (H, D))
    beta = jax.random.normal(kb, (H,))
    c = jax.random.normal(kc, (H,))
    b = 0.25
    x = jnp.array([0.1, -0.2, 0.3])

    # Closed-form 3rd derivative of tanh at a point.
    spec = get_activation("tanh")
    print("tanh^(3)(0.5) =", float(spec.fastpath(jnp.array(0.5), 3)))

    # Fused value / grad / Hessian in one pass.
    val, grad, hess = neural_field_value_grad_hessian(x, W, beta, c, b, "tanh")
    print("value:", float(val))
    print("grad shape:", grad.shape, "hessian shape:", hess.shape)

    # The closed-form Laplacian must equal trace(jax.hessian(f)).
    lap = neural_field_laplacian(x, W, beta, c, "tanh")
    ref = jnp.trace(hess)
    print(f"laplacian closed-form vs trace(hessian): "
          f"{float(lap):.12f} vs {float(ref):.12f}")
    assert jnp.allclose(lap, ref, rtol=1e-10, atol=1e-12)
    print("OK: closed-form Laplacian matches autodiff Hessian trace.")


if __name__ == "__main__":
    main()
