# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 05-02 G6/G7: field Euler bound + topology-regularized shape quality."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from omnibias.shape.topology import (
    SoftCount,
    cubical_euler_grad,
    cubical_euler_value,
    digital_genus,
    euler_value_bound,
    field_euler_characteristic,
    field_euler_pair,
    occupancy_euler,
    regularize_occupancy,
    soft_euler_characteristic,
)


def _hard_face_euler(masses: list[float], dims: list[int]) -> float:
    hard = 0.0
    for mass, dim in zip(masses, dims, strict=True):
        sign = -1.0 if int(dim) % 2 else 1.0
        hard += sign * (1.0 if mass >= 0.5 else 0.0)
    return hard


def test_softcount_never_value_alone() -> None:
    sc = occupancy_euler(np.ones((3, 3), dtype=np.float64))
    assert isinstance(sc, SoftCount)
    value, bound = sc
    assert value == pytest.approx(sc.value)
    assert bound == pytest.approx(sc.gap_bound)
    assert euler_value_bound(sc) == (sc.value, sc.gap_bound)
    assert sc.as_pair() == (sc.value, sc.gap_bound)
    with pytest.raises(TypeError):
        float(sc)  # type: ignore[arg-type]


def test_public_euler_signatures_carry_bound() -> None:
    pair = field_euler_pair(lambda x, y: 0.25 - x**2 - y**2, beta=12.0, grid=np.linspace(-1.0, 1.0, 11))
    assert isinstance(pair, tuple) and len(pair) == 2
    sc = field_euler_characteristic(lambda x, y: 0.25 - x**2 - y**2, beta=12.0, grid=np.linspace(-1.0, 1.0, 11))
    assert isinstance(sc, SoftCount)
    face = soft_euler_characteristic((0.9, 0.1, 0.8), (0, 1, 0))
    assert isinstance(face, SoftCount)
    for name, fn in (
        ("field_euler_characteristic", field_euler_characteristic),
        ("field_euler_pair", field_euler_pair),
        ("occupancy_euler", occupancy_euler),
        ("soft_euler_characteristic", soft_euler_characteristic),
    ):
        hint = inspect.signature(fn).return_annotation
        text = str(hint)
        assert "float" not in text or "tuple" in text or "SoftCount" in text, name


def test_cubical_euler_grad_matches_fd() -> None:
    rng = np.random.default_rng(4)
    u = rng.uniform(0.05, 0.95, size=(5, 6))
    numeric = np.empty_like(u)
    eps = 1e-6
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            up = u.copy()
            um = u.copy()
            up[i, j] += eps
            um[i, j] -= eps
            numeric[i, j] = (cubical_euler_value(up) - cubical_euler_value(um)) / (2.0 * eps)
    assert np.max(np.abs(numeric - cubical_euler_grad(u))) < 2e-8


def test_g6_gap_contains_true_integer() -> None:
    rng = np.random.default_rng(5)
    grid = np.linspace(-1.0, 1.0, 13)
    violations = 0
    n = 0
    for _ in range(48):
        masses = [float(v) for v in rng.uniform(0.0, 1.0, size=8)]
        dims = [int(d) for d in rng.integers(0, 3, size=8)]
        sc = soft_euler_characteristic(masses, dims)
        true = _hard_face_euler(masses, dims)
        n += 1
        if abs(sc.value - true) > sc.gap_bound + 1e-12:
            violations += 1
    for _ in range(24):
        field = rng.normal(0.0, 0.35, size=(13, 13))
        sc = field_euler_characteristic(field, beta=6.0, grid=grid)
        from omnibias.shape.topology import cubical_faces_2d, field_to_occupancy

        occ = field_to_occupancy(field, beta=6.0, grid=grid)
        masses, dims = cubical_faces_2d(occ)
        true = _hard_face_euler(masses, dims)
        n += 1
        if abs(sc.value - true) > sc.gap_bound + 1e-12:
            violations += 1
    disk = field_euler_characteristic(lambda x, y: 0.36 - x**2 - y**2, beta=16.0, grid=grid)
    n += 1
    if disk.gap_bound < 0.0:
        violations += 1
    assert violations == 0
    assert n == 73


def test_digital_genus_disk_and_annulus() -> None:
    xs = np.linspace(-1.0, 1.0, 25)
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    rho2 = xx**2 + yy**2
    disk = 1.0 / (1.0 + np.exp(-20.0 * (0.55**2 - rho2)))
    hole = 1.0 / (1.0 + np.exp(-20.0 * (0.28**2 - rho2)))
    annulus = disk * (1.0 - hole)
    assert digital_genus(disk) == 0
    assert digital_genus(annulus) == 1


def test_g7_regularizer_opens_hole() -> None:
    xs = np.linspace(-1.0, 1.0, 25)
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    rho2 = xx**2 + yy**2
    r_out, r_in = 0.70, 0.32
    disk = 1.0 / (1.0 + np.exp(-18.0 * (r_out**2 - rho2)))
    hole = 1.0 / (1.0 + np.exp(-18.0 * (r_in**2 - rho2)))
    truth = disk * (1.0 - hole)
    pos = truth >= 0.5
    exterior = rho2 >= (r_out + 0.08) ** 2
    rec = regularize_occupancy(disk, pos_mask=pos, exterior_mask=exterior, target_chi=0.0)
    assert digital_genus(disk) == 0
    assert digital_genus(rec) == 1
    inter = np.logical_and(rec >= 0.5, truth >= 0.5).sum()
    union = np.logical_or(rec >= 0.5, truth >= 0.5).sum()
    iou_reg = inter / max(int(union), 1)
    inter_b = np.logical_and(disk >= 0.5, truth >= 0.5).sum()
    union_b = np.logical_or(disk >= 0.5, truth >= 0.5).sum()
    iou_base = inter_b / max(int(union_b), 1)
    assert iou_reg >= 0.95 * iou_base
    assert math.isfinite(iou_reg)
