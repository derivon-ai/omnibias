# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""FermiNet integration layer for the multi-block ansatz.

Wires :mod:`omnibias.ferminet.multiblock` into FermiNet's
``make_fermi_net`` infrastructure so that a multi-block wavefunction
:math:`\psi_\text{total} = \sum_b \psi_b` can be trained with FermiNet's
existing KFAC + Pulay-swap loop.

The integration is **opaque to FermiNet's interior**: each block is a
full ``make_fermi_net`` (independent equivariant blocks, independent
determinant orbitals, independent envelopes), and the only thing that
changes is the final ``(sign, log_abs)`` aggregation — which is
exactly what :func:`make_multiblock_apply` does.

The production workflow:

1. Build two ``ferminet.networks.make_fermi_net`` instances (block_A,
   block_B), each with K=16 determinants.
2. Pretrain each block independently against a different HF reference
   (block_A against MIN_A HF, block_B against MIN_B HF) by running
   FermiNet's standard pretrain loop on each block in isolation.
3. Bundle the resulting params into a
   :class:`~omnibias.ferminet.multiblock.MultiBlockParams`.
4. Hand the combined apply + bundled params to FermiNet's KFAC trainer.

Step 4 requires a small patch in ``ferminet/train.py`` (or equivalent
custom main script) to swap the standard ``signed_network`` apply with
the multi-block one.  This module provides the helper functions for
all four steps; the patch itself ships with the internal FermiNet
bring-up suite.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from omnibias.ferminet.multiblock import (
    MultiBlockParams,
    assemble_multiblock_params,
    make_multiblock_apply,
)


def build_multiblock_fermi_net(
    make_fermi_net_fn: Callable[..., Any],
    n_blocks: int,
    *,
    base_seed: int,
    init_kwargs_per_block: Sequence[dict[str, Any]] | None = None,
    **make_fermi_net_kwargs: Any,
) -> tuple[Callable[..., Any], Any]:
    r"""Build a multi-block FermiNet network from per-block specs.

    Returns a tuple ``(combined_init_fn, combined_network)`` where:

    * ``combined_init_fn(key)`` -> :class:`MultiBlockParams` containing
      independent per-block parameter trees.
    * ``combined_network`` is an object that mimics FermiNet's
      ``Network`` interface but whose ``apply`` aggregates per-block
      ``(sign, log_abs)`` via signed log-sum-exp.

    Parameters
    ----------
    make_fermi_net_fn
        FermiNet's :func:`ferminet.networks.make_fermi_net` (passed in
        as a callable to avoid importing FermiNet at omnibias module
        load time).  Must return a ``Network`` object with ``.init``,
        ``.apply``, ``.orbitals``, and ``.options`` attributes.
    n_blocks
        Number of blocks :math:`b = 0, \dots, n_\text{blocks}-1`.
    base_seed
        Random seed; each block's params are initialised with
        ``base_seed + b`` to break the symmetry between blocks.
    init_kwargs_per_block
        Optional list of ``n_blocks`` dicts, one per block, of
        block-specific overrides to ``make_fermi_net_fn``.
    make_fermi_net_kwargs
        Shared kwargs passed to every block's ``make_fermi_net_fn``.

    Returns
    -------
    combined_init_fn
        Callable producing a :class:`MultiBlockParams`.
    combined_network
        A minimal object with attributes:

        * ``init(key) -> MultiBlockParams``
        * ``apply(params, positions, spins, atoms, charges) -> (sign, log_abs)``
        * ``orbitals``: a list of the per-block orbital fns (for use
          during pretrain).
        * ``signed_network``: synonym for ``apply``.
        * ``block_networks``: the list of per-block ``Network`` objects
          (for direct access to FermiNet internals if needed).
    """
    if init_kwargs_per_block is None:
        init_kwargs_per_block = [dict() for _ in range(n_blocks)]
    if len(init_kwargs_per_block) != n_blocks:
        raise ValueError(
            f"init_kwargs_per_block must have length {n_blocks}, got {len(init_kwargs_per_block)}"
        )

    block_networks: list[Any] = []
    block_apply_fns: list[Callable[..., Any]] = []
    block_orbital_fns: list[Any] = []

    for b in range(n_blocks):
        kwargs = dict(make_fermi_net_kwargs)
        kwargs.update(init_kwargs_per_block[b])
        # FermiNet's make_fermi_net returns a Network object (NamedTuple-like)
        # with .init, .apply, .orbitals, .options attributes.
        network = make_fermi_net_fn(**kwargs)
        block_networks.append(network)
        block_apply_fns.append(network.apply)
        block_orbital_fns.append(getattr(network, "orbitals", None))

    combined_apply = make_multiblock_apply(block_apply_fns)

    def combined_init_fn(key: jnp.ndarray) -> MultiBlockParams:
        # Split the key into n_blocks subkeys for independent init.
        subkeys = jax.random.split(key, n_blocks)
        block_params = []
        for b in range(n_blocks):
            # Mix in base_seed so two consecutive runs with the same `key`
            # but different `base_seed` give different inits.
            block_key = jax.random.fold_in(subkeys[b], base_seed)
            block_params.append(block_networks[b].init(block_key))
        return assemble_multiblock_params(block_params)

    class _CombinedNetwork:
        """Minimal stand-in for ferminet.networks.Network."""

        def __init__(self) -> None:
            self.init = combined_init_fn
            self.apply = combined_apply
            self.signed_network = combined_apply
            self.orbitals = block_orbital_fns
            self.block_networks = block_networks
            # Match FermiNet Network's options attribute by deferring
            # to the first block's options.
            self.options = getattr(block_networks[0], "options", None)

        def apply_to_block(self, b: int) -> Callable[..., Any]:
            """Return apply for block `b` in isolation (for pretrain)."""

            def block_apply(
                params: MultiBlockParams,
                positions: Any,
                spins: Any,
                atoms: Any,
                charges: Any,
            ) -> Any:
                return block_apply_fns[b](params.blocks[b], positions, spins, atoms, charges)

            return block_apply

    return combined_init_fn, _CombinedNetwork()


def freeze_block_params(
    params: MultiBlockParams,
    frozen_block_idx: int,
    fresh_params_block: Any,
) -> MultiBlockParams:
    """Return a new MultiBlockParams with one block's params replaced.

    Useful for "pretrain block b independently while keeping the
    others fixed" workflows.
    """
    new_blocks = list(params.blocks)
    new_blocks[frozen_block_idx] = fresh_params_block
    return MultiBlockParams(blocks=tuple(new_blocks))


def pretrain_multiblock_hartree_fock(
    *,
    params: MultiBlockParams,
    positions: Any,
    spins: Any,
    atoms: Any,
    charges: Any,
    combined_network: Any,
    block_hf_references: Sequence[Any],
    sharded_key: Any,
    electrons: Any,
    iterations: int,
    batch_size: int,
    scf_fraction: float = 0.0,
    states: int = 0,
    pretrain_hartree_fock_fn: Callable[..., Any] | None = None,
) -> tuple[MultiBlockParams, Any]:
    r"""Pretrain each block of a multi-block ansatz against its own HF ref.

    Runs FermiNet's `pretrain_hartree_fock` independently on each block
    of a multi-block network.  Walker positions are shared (we pretrain
    block ``b+1`` starting from the positions produced by block ``b``'s
    pretrain so that walker statistics are continuous).

    This is the wiring for the malonaldehyde broken-symmetry diagnostic:
    block 0 pretrained against the proton-on-O₁ HF reference and
    block 1 pretrained against the proton-on-O₂ HF reference.  At
    the end of pretrain, the two blocks live in different basins of
    the broken-symmetry landscape; KFAC then refines the combined
    ψ = ψ_0 + ψ_1.

    Parameters
    ----------
    params
        :class:`MultiBlockParams` with ``n_blocks`` entries.  The
        per-block params are updated in place (functional update).
    positions, spins, atoms, charges
        Walker tensors as fed to ``pretrain.pretrain_hartree_fock``.
    combined_network
        The ``_CombinedNetwork`` produced by
        :func:`build_multiblock_fermi_net`.  We need ``block_networks``
        (with their ``.orbitals`` attributes) and ``apply_to_block``.
    block_hf_references
        List of HF reference objects, one per block.  These are the
        return values of ``ferminet.pretrain.get_hf`` for each
        per-block molecular config.  Length must equal n_blocks.
    sharded_key
        PRNG key (sharded across devices, matching FermiNet's pretrain
        signature).
    electrons, iterations, batch_size, scf_fraction, states
        Standard pretrain arguments.
    pretrain_hartree_fock_fn
        ``ferminet.pretrain.pretrain_hartree_fock`` (passed in to
        avoid importing FermiNet at module load time).

    Returns
    -------
    (updated_params, updated_positions)
        ``updated_params`` is a fresh :class:`MultiBlockParams` with
        each block's params having been advanced by FermiNet's
        pretrain loop against its HF reference.
        ``updated_positions`` are the final walker positions after
        the last block's pretrain.
    """
    if pretrain_hartree_fock_fn is None:
        raise ValueError(
            "pretrain_multiblock_hartree_fock: must pass "
            "pretrain_hartree_fock_fn=ferminet.pretrain.pretrain_hartree_fock"
        )

    n_blocks = len(params.blocks)
    if len(block_hf_references) != n_blocks:
        raise ValueError(
            f"block_hf_references must have length {n_blocks}, got {len(block_hf_references)}"
        )

    block_networks = combined_network.block_networks
    new_block_params = list(params.blocks)
    cur_positions = positions
    cur_key = sharded_key

    for b in range(n_blocks):
        # Build per-block batch_network and batch_orbitals using
        # block_networks[b] (a real FermiNet Network).
        block_net = block_networks[b]
        block_signed = block_net.apply

        def _block_batch_network(
            p: Any, pos: Any, sp: Any, at: Any, ch: Any, _bs: Any = block_signed
        ) -> Any:
            return _bs(p, pos, sp, at, ch)[1]

        batch_network = jax.vmap(
            _block_batch_network,
            in_axes=(None, 0, 0, 0, 0),
            out_axes=0,
        )
        batch_orbitals = jax.vmap(
            block_net.orbitals,
            in_axes=(None, 0, 0, 0, 0),
            out_axes=0,
        )

        # Run FermiNet's pretrain on this block.
        block_params_b = new_block_params[b]
        block_params_b, cur_positions = pretrain_hartree_fock_fn(
            params=block_params_b,
            positions=cur_positions,
            spins=spins,
            atoms=atoms,
            charges=charges,
            batch_network=batch_network,
            batch_orbitals=batch_orbitals,
            network_options=block_net.options,
            sharded_key=cur_key,
            electrons=electrons,
            scf_approx=block_hf_references[b],
            iterations=iterations,
            batch_size=batch_size,
            scf_fraction=scf_fraction,
            states=states,
        )
        new_block_params[b] = block_params_b

    return MultiBlockParams(blocks=tuple(new_block_params)), cur_positions


__all__ = [
    "build_multiblock_fermi_net",
    "freeze_block_params",
    "pretrain_multiblock_hartree_fock",
]
