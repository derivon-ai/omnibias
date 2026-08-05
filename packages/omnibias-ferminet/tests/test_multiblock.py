# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Regression tests for the multi-block FermiNet signed log-sum-exp combine.

The combined wavefunction is the *signed sum* of block wavefunctions, assembled
through a numerically stable signed log-sum-exp with a ``stop_gradient`` shift
``M`` (see :func:`omnibias.ferminet.multiblock.make_multiblock_apply`). These
tests pin the value and gradient against the shift-free reference
``log|sum_b s_b exp(a_b)|`` and check that the stable shift keeps both finite
under overflow-scale log-amplitudes and at a tie in the per-block maxima.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.ferminet.multiblock import (  # noqa: E402
    MultiBlockParams,
    make_multiblock_apply,
)

_DUMMY = jnp.zeros((2, 3))  # positions; spins/atoms/charges unused by test blocks


def _make_block(sign_val: float, offset: float):  # type: ignore[no-untyped-def]
    def apply(p, positions, spins, atoms, charges):  # type: ignore[no-untyped-def]
        a = p + offset + 0.0 * jnp.sum(positions)  # log_abs depends on block param
        return jnp.asarray(float(sign_val)), a

    return apply


def _combined(signs, offsets):  # type: ignore[no-untyped-def]
    return make_multiblock_apply(
        [_make_block(s, o) for s, o in zip(signs, offsets, strict=True)]
    )


def _params(values):  # type: ignore[no-untyped-def]
    return MultiBlockParams(blocks=tuple(jnp.asarray(float(v)) for v in values))


def _ref_log_abs(params, signs, offsets):  # type: ignore[no-untyped-def]
    a = jnp.stack([params.blocks[b] + offsets[b] for b in range(len(offsets))])
    s = jnp.asarray(signs)
    total = jnp.sum(s * jnp.exp(a))
    return jnp.log(jnp.abs(total) + 1e-300)


def test_multiblock_value_matches_signed_logsumexp() -> None:
    signs = (1.0, -1.0, 1.0)
    offsets = (0.2, -0.5, 0.1)
    params = _params((0.3, 0.7, -0.2))
    s_tot, a_tot = _combined(signs, offsets)(params, _DUMMY, _DUMMY, _DUMMY, _DUMMY)
    assert np.isclose(
        float(a_tot), float(_ref_log_abs(params, signs, offsets)), rtol=1e-12, atol=1e-12
    )
    a = np.array([float(params.blocks[b]) + offsets[b] for b in range(3)])
    total = float(np.sum(np.array(signs) * np.exp(a)))
    assert float(s_tot) == np.sign(total)


def test_multiblock_grad_matches_shiftfree_reference() -> None:
    signs = (1.0, -1.0, 1.0)
    offsets = (0.2, -0.5, 0.1)
    params = _params((0.3, 0.7, -0.2))
    combined = _combined(signs, offsets)
    g = jax.grad(lambda p: combined(p, _DUMMY, _DUMMY, _DUMMY, _DUMMY)[1].sum())(params)
    gref = jax.grad(lambda p: _ref_log_abs(p, signs, offsets))(params)
    for b in range(3):
        assert np.isclose(float(g.blocks[b]), float(gref.blocks[b]), rtol=1e-10, atol=1e-12)


def test_multiblock_grad_finite_and_correct_at_tie() -> None:
    # Two blocks share the max log_abs, so the jnp.max subgradient is ambiguous;
    # the stop_gradient(M) shift keeps the gradient finite and equal to the
    # shift-free reference.
    signs = (1.0, 1.0)
    offsets = (0.0, 0.0)
    params = _params((0.5, 0.5))
    combined = _combined(signs, offsets)
    g = jax.grad(lambda p: combined(p, _DUMMY, _DUMMY, _DUMMY, _DUMMY)[1].sum())(params)
    gref = jax.grad(lambda p: _ref_log_abs(p, signs, offsets))(params)
    for b in range(2):
        assert np.isfinite(float(g.blocks[b]))
        assert np.isclose(float(g.blocks[b]), float(gref.blocks[b]), rtol=1e-10, atol=1e-12)


def test_multiblock_stable_under_overflow_scale() -> None:
    # exp(720) overflows to inf in float64; the stable shift must keep the value
    # and gradient finite. value == log|exp(720) - exp(719.5)|.
    signs = (1.0, -1.0)
    offsets = (720.0, 719.5)
    params = _params((0.0, 0.0))
    combined = _combined(signs, offsets)
    _, a_tot = combined(params, _DUMMY, _DUMMY, _DUMMY, _DUMMY)
    assert np.isfinite(float(a_tot))
    expected = 720.0 + float(np.log1p(-np.exp(-0.5)))
    assert np.isclose(float(a_tot), expected, rtol=1e-12, atol=1e-9)
    g = jax.grad(lambda p: combined(p, _DUMMY, _DUMMY, _DUMMY, _DUMMY)[1].sum())(params)
    assert all(np.isfinite(float(g.blocks[b])) for b in range(2))
