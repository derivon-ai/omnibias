# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The deprecation schedule is data: well formed, consistent with the installed release, and warns when due."""

from __future__ import annotations

import importlib.metadata as md
import warnings

import pytest
from omnibias.core import deprecation as dep
from omnibias.core.deprecation import Alias, version_tuple


def test_version_tuple_ignores_prerelease_suffixes():
    assert version_tuple("0.5.0rc1") == (0, 5, 0) == version_tuple("0.5.0")
    assert version_tuple("1.2") < version_tuple("1.10")
    with pytest.raises(ValueError):
        version_tuple("rc1")


def test_schedule_must_be_ordered_and_dotted():
    with pytest.raises(ValueError, match="since <= warn_from < remove_in"):
        Alias("a.b", "c.d", since="0.5.0", warn_from="0.7.0", remove_in="0.6.0")
    with pytest.raises(ValueError, match="dotted"):
        Alias("old", "c.d", since="0.5.0", warn_from="0.6.0", remove_in="0.7.0")


def test_warning_starts_at_warn_from_and_names_the_replacement(monkeypatch):
    monkeypatch.setattr(dep, "ALIASES", {})
    alias = dep.register(Alias("pkg.old.f", "os.path.join", "0.5.0", "0.6.0", "0.7.0"))
    assert dep.register(alias) is alias  # the same schedule again is a no-op
    with pytest.raises(ValueError, match="different schedule"):
        dep.register(Alias("pkg.old.f", "os.path.join", "0.5.0", "0.6.1", "0.8.0"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dep.warn_if_due("pkg.old.f", "0.5.0rc1") is False
        assert (
            dep.resolve("pkg.old.f", "0.5.9") is __import__("os").path.join
        )  # silent alias: the same object
    with pytest.warns(DeprecationWarning, match=r"removed in 0\.7\.0; use os\.path\.join"):
        assert dep.resolve("pkg.old.f", "0.6.0rc1") is __import__("os").path.join
    assert alias.due("0.6.0") and not alias.expired("0.6.9") and alias.expired("0.7.0")
    with pytest.raises(AttributeError):
        dep.resolve("pkg.unknown", "0.6.0")


def test_installed_release_has_no_expired_alias():
    pytest.importorskip("omnibias.jax")
    pytest.importorskip("omnibias.torch")
    import omnibias.jax.operators  # noqa: F401
    import omnibias.torch.operators  # noqa: F401

    current = md.version("omnibias-core")
    assert dep.ALIASES, "the back ends register their promoted names"
    for alias in dep.ALIASES.values():
        assert not alias.expired(
            current
        ), f"{alias.old} should have been removed in {alias.remove_in}"
        module, _, attr = alias.new.rpartition(".")
        assert hasattr(__import__(module, fromlist=[attr]), attr), alias.new
