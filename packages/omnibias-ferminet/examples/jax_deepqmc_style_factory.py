# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Minimal example: omnibias as a DeepQMC-style ``LaplacianFactory``.

This script shows the DeepQMC ``LaplacianFactory`` integration
pattern -- without requiring a DeepQMC install. It demonstrates:

* The DeepQMC ``LaplacianFactory`` protocol shape.
* That :func:`omnibias.jax.laplacian_factory` conforms to it
  (takes ``f: R^D -> R``, returns ``g: x -> (laplacian, gradient)``).
* That the result is numerically identical to a stock
  ``jax.linearize(jax.grad(f))`` baseline (the same path DeepQMC
  uses today via :func:`deepqmc.physics.laplacian`), and to
  ``jax.hessian``-derived references.

Run with::

    PYTHONPATH=. python examples/jax_deepqmc_style_factory.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnibias.jax import (  # noqa: E402  # x64 must be set before omnibias import
    closed_form_forward_laplacian,
    laplacian_factory,
    neural_field_value,
)

# ---------------------------------------------------------------------------
# A toy "neural wavefunction": an omnibias one-hidden-layer scalar field.
# ---------------------------------------------------------------------------


def make_params(D: int, H: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        W=jnp.asarray(rng.normal(size=(H, D)) * 0.3, dtype=jnp.float64),
        beta=jnp.zeros((H,), dtype=jnp.float64),
        c=jnp.asarray(rng.normal(size=(H,)) * 0.4, dtype=jnp.float64),
        b=jnp.float64(0.1),
    )


def log_psi(x: jnp.ndarray, params: dict, activation: str = "tanh") -> jnp.ndarray:
    return neural_field_value(
        x,
        params["W"],
        params["beta"],
        params["c"],
        params["b"],
        activation,
    )


# ---------------------------------------------------------------------------
# Pattern 1: DeepQMC LaplacianFactory style.
# ---------------------------------------------------------------------------


def pattern_1_deepqmc_factory(x, params):
    def f(r):
        return log_psi(r, params)

    lap_fn = laplacian_factory(f)
    lap, grad = lap_fn(x)
    return lap, grad


# ---------------------------------------------------------------------------
# Pattern 2: closed-form one-shot (omnibias-aware happy path).
# ---------------------------------------------------------------------------


def pattern_2_closed_form(x, params):
    res = closed_form_forward_laplacian(
        x,
        params["W"],
        params["beta"],
        params["c"],
        params["b"],
        "tanh",
    )
    return res.laplacian, res.dense_jacobian


# ---------------------------------------------------------------------------
# Reference: jax.hessian.
# ---------------------------------------------------------------------------


def pattern_ref(x, params):
    def f(r):
        return log_psi(r, params)

    grad = jax.grad(f)(x)
    lap = jnp.trace(jax.hessian(f)(x))
    return lap, grad


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    D = 24
    H = 32
    params = make_params(D=D, H=H, seed=42)
    rng = np.random.default_rng(99)
    x = jnp.asarray(rng.normal(size=(D,)) * 1.0, dtype=jnp.float64)

    lap_1, grad_1 = pattern_1_deepqmc_factory(x, params)
    lap_2, grad_2 = pattern_2_closed_form(x, params)
    lap_ref, grad_ref = pattern_ref(x, params)

    print("== omnibias as DeepQMC LaplacianFactory ==")
    print(f"D                       : {D}")
    print(f"H                       : {H}")
    print("activation              : tanh")
    print()
    print(f"laplacian  (pattern 1)  : {float(lap_1):+.12f}")
    print(f"laplacian  (pattern 2)  : {float(lap_2):+.12f}")
    print(f"laplacian  (jax.hessian): {float(lap_ref):+.12f}")
    print()
    print(f"|lap_1 - lap_ref|       : {abs(float(lap_1 - lap_ref)):.3e}")
    print(f"|lap_2 - lap_ref|       : {abs(float(lap_2 - lap_ref)):.3e}")
    print(f"|grad_1 - grad_ref|_inf : {float(jnp.max(jnp.abs(grad_1 - grad_ref))):.3e}")
    print(f"|grad_2 - grad_ref|_inf : {float(jnp.max(jnp.abs(grad_2 - grad_ref))):.3e}")
    print()

    # JIT-compile each and microbenchmark.
    def time_fn(fn, *args, warmup=3, repeats=20):
        f = jax.jit(fn)
        for _ in range(warmup):
            out = f(*args)
            jax.block_until_ready(out)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = f(*args)
            jax.block_until_ready(out)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1000

    t1 = time_fn(pattern_1_deepqmc_factory, x, params)
    t2 = time_fn(pattern_2_closed_form, x, params)
    tref = time_fn(pattern_ref, x, params)

    print("jit'd timing (single walker):")
    print(f"  pattern 1 (factory)     : {t1:.3f} ms")
    print(f"  pattern 2 (closed form) : {t2:.3f} ms  ({tref / t2:.2f}x vs jax.hessian)")
    print(f"  ref (jax.hessian)       : {tref:.3f} ms")


if __name__ == "__main__":
    main()
