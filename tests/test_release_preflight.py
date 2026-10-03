# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Release failures must stop before upload, including reruns and index outages."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_preflight', ROOT / 'scripts/release_preflight.py')
assert spec and spec.loader
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_matching_package_tag() -> None:
    assert release.select(ROOT, '', 'omnibias-core-v0.4.0', 'push') == ['core']


@pytest.mark.parametrize('tag', ['omnibias-core-v9.9.9', 'v0.4.0', 'omnibias-pinn-v0.1.0'])
def test_rejects_wrong_or_external_tag(tag: str) -> None:
    with pytest.raises(ValueError):
        release.select(ROOT, '', tag, 'push')


def test_manual_selection_is_validated() -> None:
    assert release.select(ROOT, 'core core jax', '', 'workflow_dispatch') == ['core', 'jax']
    with pytest.raises(ValueError):
        release.select(ROOT, '../core', '', 'workflow_dispatch')


def test_only_missing_release_permits_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args, **kwargs):
        raise HTTPError('https://pypi.org', 404, 'missing', {}, None)
    monkeypatch.setattr(release, 'urlopen', missing)
    release.ensure_unpublished('omnibias-core', '0.5.0', 'pypi')


@pytest.mark.parametrize('error', [HTTPError('url', 403, 'denied', {}, None), URLError('offline')])
def test_index_failure_is_not_treated_as_absence(monkeypatch: pytest.MonkeyPatch, error) -> None:
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(release, 'urlopen', broken)
    with pytest.raises(type(error)):
        release.ensure_unpublished('omnibias-core', '0.5.0', 'pypi')


def test_existing_release_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext
    monkeypatch.setattr(release, 'urlopen', lambda *a, **kw: nullcontext())
    with pytest.raises(ValueError, match='already exists'):
        release.ensure_unpublished('omnibias-core', '0.4.0', 'pypi')
