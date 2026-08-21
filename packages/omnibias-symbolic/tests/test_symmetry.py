# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-11: Lie point-symmetry discovery, gates G1–G6."""

from __future__ import annotations

import numpy as np
from omnibias.symbolic.symmetry import (
    DISCLAIMER,
    Generator,
    LinearPoly,
    affine_basis,
    designed_samples,
    determining_matrix,
    discover_symmetries,
    eta_t,
    eta_xx,
    heat_known_coeffs,
    honesty_payload,
    matrix_condition,
    negative_control,
    noether_wave_residual,
    pr_for,
    pr_heat,
    pr_heat_fd,
    suite,
)


def _heat_scale() -> Generator:
    return Generator(LinearPoly(cx=1.0), LinearPoly(ct=2.0), LinearPoly(), "scale")


def test_worked_heat_scale() -> None:
    from omnibias.symbolic.symmetry._core import Sample, _restrict_heat

    raw = Sample(0.4, 0.3, 1.2, 0.5, 0.0, 0.8, 0.1, 0.2, 0.3, 0.05, 0.0, 0.0)
    s = _restrict_heat(raw)
    g = _heat_scale()
    assert abs(eta_t(g, s) + 2.0 * s.ut) < 1e-14
    assert abs(eta_xx(g, s) + 2.0 * s.uxx) < 1e-14
    assert abs(pr_heat(g, s)) < 1e-14


def test_g1_in_ansatz_dimensions_and_heat_generators() -> None:
    basis = affine_basis()
    samples = designed_samples(32)
    for pde in suite():
        res = discover_symmetries(pr_for(pde.name), pde.restrict, basis, samples)
        assert res.algebra_dim == pde.expected_dim, pde.name
        assert res.basis.name == "affine_xtu"
        assert res.disclaimer == DISCLAIMER
    heat = next(p for p in suite() if p.name == "heat")
    mat = determining_matrix(pr_heat, heat.restrict, basis, samples)
    assert matrix_condition(mat) < 1e4
    for vec in heat_known_coeffs():
        assert float(np.linalg.norm(mat @ vec)) <= 1e-12 * max(1.0, float(np.linalg.norm(mat)))


def test_g2_rank_separation() -> None:
    basis = affine_basis()
    samples = designed_samples(32)
    for pde in suite():
        res = discover_symmetries(pr_for(pde.name), pde.restrict, basis, samples)
        assert res.separation > 1e6, (pde.name, res.separation)


def test_g3_fd_rank_wrong() -> None:
    basis = affine_basis()
    samples = designed_samples(32)
    heat = next(p for p in suite() if p.name == "heat")
    exact = discover_symmetries(pr_heat, heat.restrict, basis, samples)
    fd = discover_symmetries(pr_heat_fd, heat.restrict, basis, samples)
    assert exact.algebra_dim == 5
    assert fd.algebra_dim != exact.algebra_dim


def test_g4_threshold_stable() -> None:
    basis = affine_basis()
    samples = designed_samples(32)
    heat = next(p for p in suite() if p.name == "heat")
    dims = [
        discover_symmetries(pr_heat, heat.restrict, basis, samples, threshold=thr).algebra_dim
        for thr in (1e-12, 1e-10, 1e-8)
    ]
    assert dims == [5, 5, 5]


def test_g5_noether_wave() -> None:
    dt = Generator(LinearPoly(), LinearPoly(c0=1.0), LinearPoly(), "dt")
    dx = Generator(LinearPoly(c0=1.0), LinearPoly(), LinearPoly(), "dx")
    junk = Generator(LinearPoly(ct=1.0), LinearPoly(), LinearPoly(c0=1.0), "junk")
    assert noether_wave_residual(dt) <= 1e-10
    assert noether_wave_residual(dx) <= 1e-10
    assert noether_wave_residual(junk) > 0.1


def test_g6_negative_control() -> None:
    spec = negative_control()
    res = discover_symmetries(pr_for(spec.name), spec.restrict, affine_basis(), designed_samples(32))
    assert res.algebra_dim == 0
    assert res.generators == ()


def test_honesty() -> None:
    payload = honesty_payload()
    assert payload["point_symmetries_only"] is True
    assert payload["ns_regularity"] is False
    assert payload["temperature_collapse"] is False
    assert "ansatz" in DISCLAIMER
