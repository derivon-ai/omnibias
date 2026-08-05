# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multi-block FermiNet wrapper: :math:`\psi_\text{total}(r) = \sum_b \psi_b(r)`.

Multi-block design originally targeted the malonaldehyde
wrong-sign-barrier remediation; see ``docs/roadmap.md`` for the public
roadmap. Each "block" is an
independent FermiNet apply function with its own parameter tree;
the combined wavefunction is the *signed sum* of block wavefunctions.

The block sum is **not** the same as a wider single FermiNet (e.g.,
doubling the determinant count from K=16 to K=32 expands the orbital
basis within a single ansatz; the multi-block form, by contrast, lives
on a *direct sum* of independent ansätze, which is much better suited
to broken-symmetry minima where the wavefunction must localise on one
side or the other of a symmetric saddle point).

Forward-pass math
-----------------

Let each block return a ``(sign, log_abs)`` pair.  The combined
wavefunction is

.. math::

   \psi_\text{total}(r) \,=\, \sum_b s_b(r)\,|\psi_b|(r),

so :math:`|\psi_\text{total}|` can be assembled via a numerically
stable log-sum-exp-with-signs:

.. math::

   M(r) &= \max_b\, a_b(r),
   \\\
   \mathrm{sum}(r) &= \sum_b s_b(r)\,\exp\bigl(a_b(r)-M(r)\bigr),
   \\\
   s_\text{total}(r) &= \mathrm{sign}\bigl(\mathrm{sum}(r)\bigr),
   \\\
   a_\text{total}(r) &= M(r) + \log\bigl|\mathrm{sum}(r)\bigr|,

where :math:`a_b(r) := \log|\psi_b(r)|`.

The combined ``apply`` function returns ``(s_total, a_total)`` and
is plug-compatible with FermiNet's
``laplacian_method='omnibias_envelope'`` / ``'omnibias_total_relativistic'``
branches.

Pretrain strategy
-----------------

Each block is pretrained **independently** against a different HF
reference (e.g., block 0 → MIN_A HF, block 1 → MIN_B HF).  This is
done by calling FermiNet's standard pretrain loop on each block in
isolation (with the other blocks' contributions zeroed out for the
purpose of the pretrain loss) and then concatenating the resulting
parameter trees into a :class:`MultiBlockParams` named tuple.

The production-config end-to-end is documented in
``docs/api/ferminet.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# Type alias matching FermiNet's ``FermiNetLike`` interface.
_BlockApplyFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]


class MultiBlockParams(NamedTuple):
    """Pytree container for multi-block FermiNet parameters.

    ``blocks`` is a tuple (or list) of per-block parameter pytrees;
    the i-th element is what would normally be passed as the ``params``
    argument to the i-th block's apply function.

    JAX treats ``MultiBlockParams`` as a pytree (because it's a
    ``NamedTuple``), so optimisers, ``jax.jit``, ``jax.grad``, and
    ``kfac_jax`` see the full structure transparently.
    """

    blocks: tuple[Any, ...]


def make_multiblock_apply(
    block_apply_fns: Sequence[_BlockApplyFn],
) -> _BlockApplyFn:
    r"""Combine multiple FermiNet apply functions into a single one.

    Parameters
    ----------
    block_apply_fns
        Sequence of per-block apply functions with FermiNet's signature
        ``(params, positions, spins, atoms, charges) -> (sign, log_abs)``.

    Returns
    -------
    combined_apply
        A new apply function returning ``(sign, log_abs)`` of
        :math:`\psi_\text{total} = \sum_b \psi_b`.  ``params`` must be a
        :class:`MultiBlockParams` whose ``blocks`` tuple matches
        ``block_apply_fns`` in length and order.
    """
    n_blocks = len(block_apply_fns)
    if n_blocks < 2:
        raise ValueError(f"make_multiblock_apply requires >=2 blocks, got {n_blocks}")
    block_apply_fns = tuple(block_apply_fns)

    def combined_apply(
        params: MultiBlockParams,
        positions: jnp.ndarray,
        spins: jnp.ndarray,
        atoms: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        signs: list[jnp.ndarray] = []
        logabses: list[jnp.ndarray] = []
        for b, fn in enumerate(block_apply_fns):
            s, la = fn(params.blocks[b], positions, spins, atoms, charges)
            signs.append(s)
            logabses.append(la)
        signs_arr = jnp.stack(signs)  # (n_blocks,) possibly complex/real
        logabses_arr = jnp.stack(logabses)  # (n_blocks,) real

        # Numerically stable signed log-sum-exp.
        # Use jax.lax.stop_gradient on M to avoid spurious gradients
        # through the max; M itself is then added back unchanged.
        M = jax.lax.stop_gradient(jnp.max(logabses_arr))
        shifted = signs_arr * jnp.exp(logabses_arr - M)
        total = jnp.sum(shifted)
        sign_total = jnp.sign(total)
        # Add tiny epsilon to log to avoid -inf gradients when total == 0;
        # this is the same idiom used by FermiNet's `slog_sum_signed`.
        log_abs_total = M + jnp.log(jnp.abs(total) + 1e-300)
        return sign_total, log_abs_total

    return combined_apply


def split_multiblock_params(params: MultiBlockParams, block_idx: int) -> Any:
    """Extract the parameter pytree for block ``block_idx`` from a
    :class:`MultiBlockParams` container.

    Convenience for downstream code that wants to operate on a single
    block at a time (e.g., per-block pretrain).
    """
    return params.blocks[block_idx]


def assemble_multiblock_params(
    block_params: Sequence[Any],
) -> MultiBlockParams:
    """Bundle per-block parameter pytrees into a
    :class:`MultiBlockParams` container."""
    return MultiBlockParams(blocks=tuple(block_params))


def make_single_block_apply_from_multi(
    block_apply_fns: Sequence[_BlockApplyFn],
    block_idx: int,
) -> _BlockApplyFn:
    """Wrap ``block_apply_fns[block_idx]`` to accept a
    :class:`MultiBlockParams` (extracting just the block_idx-th
    element) so the same Pulay-swap / pretrain code can be reused.

    Useful when pretraining a single block in isolation, where the
    pretrain step expects a "FermiNetLike" callable but you want to
    keep the multi-block param tree intact.
    """
    if block_idx < 0 or block_idx >= len(block_apply_fns):
        raise IndexError(f"block_idx={block_idx} out of range [0, {len(block_apply_fns)})")
    fn = block_apply_fns[block_idx]

    def single_apply(
        params: MultiBlockParams,
        positions: jnp.ndarray,
        spins: jnp.ndarray,
        atoms: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return fn(params.blocks[block_idx], positions, spins, atoms, charges)

    return single_apply


__all__ = [
    "MultiBlockParams",
    "assemble_multiblock_params",
    "make_multiblock_apply",
    "make_single_block_apply_from_multi",
    "split_multiblock_params",
]
