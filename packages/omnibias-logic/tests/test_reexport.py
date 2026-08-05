# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""omnibias-logic re-exports the MaxSAT surface and substrate helpers unchanged."""

from __future__ import annotations

import omnibias.logic as logic
import pytest


def test_maxsat_surface_is_the_same_object() -> None:
    from omnibias.discrete.maxsat import Clause, MaxSATProblem, WeightedCNF, max_sat

    assert logic.max_sat is max_sat
    assert logic.MaxSATProblem is MaxSATProblem
    assert logic.WeightedCNF is WeightedCNF
    assert logic.Clause is Clause


def test_substrate_helpers_are_the_same_object() -> None:
    from omnibias.discrete import (
        AnnealSchedule,
        DiscreteSolution,
        GapCertificate,
        brute_force_min,
        certify_gap,
        decode,
    )

    assert logic.decode is decode
    assert logic.brute_force_min is brute_force_min
    assert logic.certify_gap is certify_gap
    assert logic.GapCertificate is GapCertificate
    assert logic.AnnealSchedule is AnnealSchedule
    assert logic.DiscreteSolution is DiscreteSolution


def test_all_is_sorted_and_has_version() -> None:
    assert logic.__all__ == sorted(logic.__all__)
    assert "__version__" in logic.__all__
    assert logic.__version__ == "0.1.0a1"


def test_backend_relaxation_reexports_are_identical() -> None:
    pytest.importorskip("torch")
    from omnibias.discrete.maxsat.torch import maxsat_relaxation as disc_torch
    from omnibias.logic.torch import maxsat_relaxation as logic_torch

    assert logic_torch is disc_torch

    pytest.importorskip("jax")
    from omnibias.discrete.maxsat.jax import maxsat_relaxation as disc_jax
    from omnibias.logic.jax import maxsat_relaxation as logic_jax

    assert logic_jax is disc_jax
