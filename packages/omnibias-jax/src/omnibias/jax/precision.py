# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The precision precondition behind every "bit-identical twin" claim.

omnibias computes the same closed-form derivative tower on PyTorch and JAX from
one shared coefficient module, so the two backends agree **bit for bit** -- in
double precision. PyTorch uses ``float64`` for Python floats; JAX silently
truncates arrays to ``float32`` unless 64-bit mode is enabled. Under the JAX
default the twins are each internally consistent but agree only to ``float32``
tolerance.

That gap is not always cosmetic. Wherever a value feeds a threshold, a rounding
step or an ``argmax`` -- the decoders in ``omnibias.discrete`` / ``omnibias.qubo``
/ ``omnibias.struct``, the hardening step in ``omnibias.partition`` -- a
``1e-7`` disagreement can flip a bit and move the answer by ``1``, so the
divergence shows up as a different decision rather than a different last digit.

This module deliberately does **not** flip the flag for you. ``jax_enable_x64``
is process-global and cannot be changed once arrays exist, so a library that
sets it on import would silently re-specify the numerics of every other piece of
JAX code in the process. Enable it yourself, before the first JAX array::

    import jax
    jax.config.update("jax_enable_x64", True)

or set ``JAX_ENABLE_X64=1`` in the environment.
"""

from __future__ import annotations

X64_HINT = (
    'enable 64-bit JAX before the first array is created -- '
    'jax.config.update("jax_enable_x64", True) or JAX_ENABLE_X64=1'
)


def x64_enabled() -> bool:
    """Whether JAX is in 64-bit mode, i.e. whether bit-parity is achievable."""
    import jax

    return bool(getattr(jax.config, "jax_enable_x64", False))


def require_x64(context: str = "bit-identical torch/jax parity") -> None:
    """Raise unless 64-bit JAX is on.

    Call this at the top of a script or test whose correctness depends on the
    parity claim, so the precondition fails loudly instead of degrading into a
    silent ``float32`` comparison.
    """
    if not x64_enabled():
        raise RuntimeError(f"{context} requires 64-bit JAX: {X64_HINT}")


__all__ = ["X64_HINT", "require_x64", "x64_enabled"]
