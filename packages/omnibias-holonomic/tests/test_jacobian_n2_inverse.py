# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gabber inverse-degree test for n=2 Keller maps."""

from __future__ import annotations

from omnibias.holonomic._core.poly_n import PolyN
from omnibias.holonomic.jacobian_n2 import (
    n2_counterexample_earned,
    n2_violation_payload,
    shear_map,
)
from omnibias.holonomic.jacobian_n2_inverse import (
    compose_map,
    formal_inverse,
    gabber_inverse_degree_bound,
    gabber_n2_test,
    is_identity_map,
    map_degree,
    shift_to_origin,
)


def _xy() -> tuple[PolyN, PolyN]:
    return PolyN.var(2, 0), PolyN.var(2, 1)


def test_gabber_bound_is_degree_for_n2() -> None:
    assert gabber_inverse_degree_bound(1, nvars=2) == 1
    assert gabber_inverse_degree_bound(5, nvars=2) == 5
    assert gabber_inverse_degree_bound(3, nvars=3) == 9


def test_shear_passes_gabber() -> None:
    shear = shear_map(0, (0, 0, 1))
    result = gabber_n2_test(shear)
    assert result.keller is True
    assert result.jacobian_constant == 1
    assert result.degree == 2
    assert result.bound == 2
    assert result.inverse_ok is True
    assert result.fails is False


def test_translated_shear_passes_gabber() -> None:
    x, y = _xy()
    translated = (x + 3, y + x**3 - 2)
    result = gabber_n2_test(translated)
    assert result.keller is True
    assert result.inverse_ok is True
    assert result.fails is False


def test_tame_composition_passes_gabber() -> None:
    x, y = _xy()
    first = (x + y**2, y)
    second = (x, y + x**2)
    composed = compose_map(first, second)
    assert map_degree(composed) == 4
    result = gabber_n2_test(composed)
    assert result.keller is True
    assert result.bound == 4
    assert result.inverse_ok is True
    assert result.fails is False


def test_fold_is_not_keller() -> None:
    x, y = _xy()
    result = gabber_n2_test((x**2, y))
    assert result.keller is False
    assert result.fails is False
    assert result.inverse_ok is False


def test_low_jet_is_not_an_inverse() -> None:
    shear = shear_map(0, (0, 0, 0, 0, 1))
    shifted = shift_to_origin(shear)
    short = formal_inverse(shifted, max_degree=2)
    assert not is_identity_map(compose_map(shifted, short))
    full = formal_inverse(shifted, max_degree=4)
    assert is_identity_map(compose_map(shifted, full))
    assert is_identity_map(compose_map(full, shifted))


def test_payload_records_gabber_pass() -> None:
    payload = n2_violation_payload(shear_map(1, (0, 0, 1)))
    assert payload["gabber_fails"] is False
    assert payload["gabber_inverse_ok"] is True
    assert n2_counterexample_earned(payload) is False


def test_gabber_fail_payload_earns_n2() -> None:
    assert n2_counterexample_earned(
        {
            "jacobian_identity": "identical",
            "jacobian_nonzero_constant": True,
            "gabber_fails": True,
        }
    )
    assert not n2_counterexample_earned(
        {
            "jacobian_identity": "identical",
            "jacobian_nonzero_constant": False,
            "gabber_fails": True,
        }
    )
