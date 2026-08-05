# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The operator catalog must stay in lock-step with the dispatch surface.

If you add an op to ``{torch,jax}/_ops_dispatch.py`` you must catalogue it (and
vice versa); these tests fail otherwise, which is the point.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omnibias.fields import (
    DOMAINS,
    get_operator,
    list_operators,
    operator_names,
)

_OPERATORS_MD = Path(__file__).resolve().parents[3] / "docs" / "operators.md"


def _dispatch_names(backend: str) -> set[str]:
    if backend == "jax":
        from omnibias.fields.jax import _ops_dispatch as d
    else:
        from omnibias.fields.torch import _ops_dispatch as d
    return set(d.list_ops())


def test_catalog_matches_jax_dispatch_surface_exactly():
    assert operator_names() == _dispatch_names("jax")


def test_catalog_matches_torch_dispatch_surface_exactly():
    assert operator_names() == _dispatch_names("torch")


def test_backends_agree_on_op_surface():
    assert _dispatch_names("jax") == _dispatch_names("torch")


def test_every_operator_has_a_known_domain():
    for op in list_operators():
        assert op.domain in DOMAINS, op


def test_every_domain_is_populated():
    for domain in DOMAINS:
        assert list_operators(domain=domain), f"domain {domain!r} has no operators"


def test_domain_filter_partitions_the_catalog():
    total = sum(len(list_operators(domain=d)) for d in DOMAINS)
    assert total == len(list_operators())
    assert total == len(operator_names())


def test_list_operators_is_sorted_by_domain_then_name():
    ops = list_operators()
    keys = [(DOMAINS.index(o.domain), o.name) for o in ops]
    assert keys == sorted(keys)


def test_list_operators_rejects_unknown_domain():
    with pytest.raises(ValueError, match="unknown domain"):
        list_operators(domain="nonsense")


def test_get_operator_roundtrips_and_raises():
    info = get_operator("faraday_residual")
    assert info.domain == "electromagnetism"
    assert info.formula
    with pytest.raises(KeyError):
        get_operator("does_not_exist")


def test_field_operator_surface_is_all_closed_form_torch_jax():
    # The omnibias-fields surface is exact: every op is the closed-form sigma
    # tower or an exact composition of it, on both backends.
    for op in list_operators():
        assert op.closed_form is True, op
        assert op.backends == ("torch", "jax"), op


def test_aliases_are_catalogued_with_their_primaries():
    names = operator_names()
    for alias, primary in (("rot", "curl"), ("div", "divergence"), ("wave_operator", "dalembertian")):
        assert alias in names and primary in names
        assert get_operator(alias).domain == get_operator(primary).domain


def test_docs_operators_md_documents_every_op():
    """The reference page must mention every catalogued operator (no doc drift)."""
    if not _OPERATORS_MD.is_file():
        pytest.skip("docs/operators.md not present (package installed without repo docs)")
    text = _OPERATORS_MD.read_text(encoding="utf-8")
    missing = sorted(name for name in operator_names() if f"`{name}`" not in text)
    assert not missing, f"docs/operators.md is missing operators: {missing}"
