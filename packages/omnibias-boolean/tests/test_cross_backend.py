# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity (torch <-> jax) for gates, spectrum, and the solver.

Bit-parity holds in float64 only, so jax x64 is enabled before importing the jax
ops (mirroring the omnibias-binary cross-backend tests).
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.boolean._core.truth_table import pm1_values  # noqa: E402
from omnibias.boolean.jax import ops as jb  # noqa: E402
from omnibias.boolean.torch import ops as tb  # noqa: E402


def _np(v):  # type: ignore[no-untyped-def]
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _tts(seed: int, ns=(1, 2, 3, 4)):  # type: ignore[no-untyped-def]
    rng = random.Random(seed)
    for n in ns:
        for _ in range(6):
            yield tuple(rng.randint(0, 1) for _ in range(1 << n))


def test_gates_parity() -> None:
    rng = random.Random(20)
    for _ in range(20):
        a, b = rng.random(), rng.random()
        for name in ("soft_and", "soft_or", "soft_xor", "soft_nand", "soft_implies"):
            tg = getattr(tb, name)
            jg = getattr(jb, name)
            tv = float(tg(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)))
            jv = float(jg(jnp.asarray(a, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64)))
            assert tv == pytest.approx(jv, rel=1e-9, abs=1e-11)


def test_mobius_parity() -> None:
    for tt in _tts(21):
        t = _np(tb.mobius_coeffs(torch.tensor(tt, dtype=torch.float64)))
        j = _np(jb.mobius_coeffs(jnp.asarray(tt, dtype=jnp.float64)))
        assert np.allclose(t, j, rtol=1e-9, atol=1e-11)


def test_walsh_and_influence_parity() -> None:
    for tt in _tts(22):
        vals = pm1_values(tt)
        tw = _np(tb.walsh_coeffs(torch.tensor(vals, dtype=torch.float64)))
        jw = _np(jb.walsh_coeffs(jnp.asarray(vals, dtype=jnp.float64)))
        assert np.allclose(tw, jw, rtol=1e-9, atol=1e-11)
        ti = _np(tb.influences_diff(torch.tensor(vals, dtype=torch.float64)))
        ji = _np(jb.influences_diff(jnp.asarray(vals, dtype=jnp.float64)))
        assert np.allclose(ti, ji, rtol=1e-9, atol=1e-11)


def test_jax_solver_gf2_and_anneal() -> None:
    lin = jb.BooleanSystem.from_predicates(
        [lambda a, b: (a ^ b) == 1, lambda a, b: b == 1], n=2
    )
    res = jb.solve(lin)
    assert res.method == "gf2" and res.verified and res.assignment == (0, 1)

    nonlin = jb.BooleanSystem.from_predicates(
        [lambda a, b, c: (a & b) == 1, lambda a, b, c: (b | c) == 1], n=3
    )
    res = jb.solve(nonlin, steps=200, restarts=12, seed=0)
    assert res.verified and res.assignment is not None
    assert nonlin.verify(res.assignment)


def test_design_loss_parity() -> None:
    vals = [0.1, 0.7, 0.3, 0.9]
    t = float(tb.degree_penalty(torch.tensor(vals, dtype=torch.float64)))
    j = float(jb.degree_penalty(jnp.asarray(vals, dtype=jnp.float64)))
    assert t == pytest.approx(j, rel=1e-9, abs=1e-11)
