# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""MultiPackUnit exactness, collapse, and torch/jax parity (theory 01-01)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.core.multipack import (
    MultiPackSpec,
    PackSpec,
    central_stencil_weights,
)
from omnibias.core.polynomials import sigmoid_polynomial_coeffs, tanh_polynomial_coeffs

mpmath = pytest.importorskip("mpmath")


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 2.2250738585072014e-308)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _mp_poly(base: str, z: float, n: int) -> float:
    """High-dps Horner of the shared polynomial coeffs (independent of float64)."""
    mpmath.mp.dps = 80
    if base == "tanh":
        s = mpmath.tanh(mpmath.mpf(z))
        coeffs = tanh_polynomial_coeffs(n) if n > 0 else None
    else:
        s = mpmath.mpf(1) / (mpmath.mpf(1) + mpmath.exp(-mpmath.mpf(z)))
        coeffs = sigmoid_polynomial_coeffs(n) if n > 0 else None
    if n == 0:
        return float(s)
    assert coeffs is not None
    acc = mpmath.mpf(0)
    for c in reversed(coeffs):
        acc = acc * s + mpmath.mpf(c)
    return float(acc)


def test_worked_example_matches_spec() -> None:
    """Spec §5 closed form at z=0 for tanh support."""
    from omnibias.torch.multipack import MultiPackUnit

    torch.set_default_dtype(torch.float64)
    spec = MultiPackSpec((PackSpec(1, -0.5, 1.0), PackSpec(2, 0.5, 0.25)))
    unit = MultiPackUnit(
        1, spec, base="tanh", learnable_means=False, learnable_weights=False
    )
    z = torch.zeros(1, 1, dtype=torch.float64)
    got = float(unit(z).item())
    t1 = math.tanh(-0.5)
    s1 = 1.0 - t1 * t1
    t2 = math.tanh(0.5)
    s2 = -2.0 * t2 * (1.0 - t2 * t2)
    expected = 1.0 * s1 + 0.25 * s2
    assert got == pytest.approx(
        expected, rel=0, abs=4 * np.finfo(float).eps * max(abs(expected), 1.0)
    )


def test_g1_exactness_and_order_ceiling() -> None:
    """G1: measure ulp vs order; gate at the honest ceiling (plan §Gates).

    Pre-registered domain: ``z in [-1, 1]``, single-pack at mean 0. Spec §11
    warns ``P_n`` loses accuracy at high order; we record the highest order
    where worst ulp ``<= 4`` rather than loosening the threshold.
    """
    from omnibias.torch.multipack import MultiPackUnit

    torch.set_default_dtype(torch.float64)
    grid = np.linspace(-1.0, 1.0, 81)
    ceilings: dict[str, int] = {}
    curves: dict[str, list[float]] = {}
    for base in ("sigmoid", "tanh"):
        worsts: list[float] = []
        ceiling = -1
        for n in range(0, 7):
            spec = MultiPackSpec((PackSpec(n, 0.0, 1.0),))
            unit = MultiPackUnit(
                1, spec, base=base, learnable_means=False, learnable_weights=False
            )
            worst = 0.0
            for z in grid:
                pred = float(unit(torch.tensor([[float(z)]], dtype=torch.float64)).item())
                ref = _mp_poly(base, float(z), n)
                worst = max(worst, _ulp_error(pred, ref))
            worsts.append(worst)
            if worst <= 4.0:
                ceiling = n
        curves[base] = worsts
        ceilings[base] = ceiling
        # Floor: at least order 1 must earn 4 ulp on this domain.
        assert ceiling >= 1, (
            f"{base}: G1 order ceiling {ceiling} < 1; worsts={worsts}"
        )
    # Multi-pack worked example on the same domain.
    spec = MultiPackSpec((PackSpec(1, -0.5, 1.0), PackSpec(2, 0.5, 0.25)))
    unit = MultiPackUnit(
        1, spec, base="tanh", learnable_means=False, learnable_weights=False
    )
    worst_mp = 0.0
    for z in grid:
        pred = float(unit(torch.tensor([[float(z)]], dtype=torch.float64)).item())
        ref = sum(
            p.weight * _mp_poly("tanh", float(z) + p.mean, p.order) for p in spec.packs
        )
        worst_mp = max(worst_mp, _ulp_error(pred, ref))
    assert worst_mp <= 4.0, f"worked-example multipack worst_ulp={worst_mp}"
    # Stash for the benchmark to echo (importable constants).
    test_g1_exactness_and_order_ceiling.ceilings = ceilings  # type: ignore[attr-defined]
    test_g1_exactness_and_order_ceiling.curves = curves  # type: ignore[attr-defined]
    print(f"G1 ceilings={ceilings} curves={curves} worked_ulp={worst_mp:.3g}")


def test_g2_collapse_no_closed_form_growth() -> None:
    """Closed form stable as delta shrinks; FD error grows then breaks."""
    from omnibias.torch.multipack import MultiPackUnit

    torch.set_default_dtype(torch.float64)
    spec = MultiPackSpec((PackSpec(1, -0.5, 1.0), PackSpec(2, 0.5, 0.25)))
    unit = MultiPackUnit(
        1, spec, base="tanh", learnable_means=False, learnable_weights=False
    )
    z0 = 0.0
    closed = float(unit(torch.tensor([[z0]], dtype=torch.float64)).item())
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    fd_errs = []
    for delta in deltas:
        acc = 0.0
        for p in spec.packs:
            offsets, signs = central_stencil_weights(p.order, delta)
            for off, s in zip(offsets, signs, strict=True):
                acc += p.weight * s * math.tanh(z0 + p.mean + off)
        fd_errs.append(abs(acc - closed))
    assert fd_errs[0] > fd_errs[2] or fd_errs[2] < 1e-8
    closed2 = float(unit(torch.tensor([[z0]], dtype=torch.float64)).item())
    assert closed2 == closed
    assert fd_errs[-1] > fd_errs[2]


def test_g3_torch_jax_bit_identical() -> None:
    import jax
    import jax.numpy as jnp
    from omnibias.jax.multipack import init_multipack, multipack_apply
    from omnibias.torch.multipack import MultiPackUnit

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    # Worked-example support; z chosen where native torch/jax tanh agree
    # bit-for-bit (see test_sigmoid_tail_parity for the documented tail
    # contract when they do not).
    spec = MultiPackSpec((PackSpec(1, -0.5, 1.0), PackSpec(2, 0.5, 0.25)))
    unit = MultiPackUnit(
        2, spec, base="tanh", learnable_means=False, learnable_weights=False
    )
    z_np = np.array([[0.0, 0.0], [0.25, -0.25], [0.5, 0.5]], dtype=np.float64)
    torch_out = unit(torch.as_tensor(z_np)).detach().numpy()
    _act, means, weights, orders, mean_index = init_multipack(
        2, spec, base="tanh", share_means=True
    )
    jax_out = np.asarray(
        multipack_apply(
            jnp.asarray(z_np), means, weights, orders, "tanh", mean_index=mean_index
        )
    )
    np.testing.assert_array_equal(torch_out, jax_out)


def test_negative_order_raises() -> None:
    with pytest.raises(ValueError):
        PackSpec(-1, 0.0)
