# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Select release artifacts; reject mismatched tags and occupied index versions."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import tomllib
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ('core', 'torch', 'jax', 'keras', 'fields')


def select(root: Path, requested: str, tag: str, event: str) -> list[str]:
    config = tomllib.loads((root / 'pyproject.toml').read_text())
    members = config['tool']['uv']['workspace']['members']
    allowed = {Path(member).name.removeprefix('omnibias-') for member in members}
    names = requested.split() or list(DEFAULT)
    tagged_version = None
    if event == 'push':
        match = re.fullmatch(r'omnibias-([a-z0-9-]+)-v(.+)', tag)
        if match:
            names = [match[1]]
            tagged_version = Version(match[2])
        else:
            raise ValueError('Use an independent omnibias-<package>-v<version> tag')
    if not names or any(name not in allowed for name in names):
        raise ValueError('Release selections must name retained workspace packages')
    names = list(dict.fromkeys(names))
    for name in names:
        project = tomllib.loads(
            (root / f'packages/omnibias-{name}/pyproject.toml').read_text()
        )['project']
        if tagged_version is not None and Version(project['version']) != tagged_version:
            raise ValueError(f'Tag does not match {project["name"]} metadata version')
    return names


def ensure_unpublished(name: str, version: str, repository: str) -> None:
    host = {'pypi': 'pypi.org', 'testpypi': 'test.pypi.org'}[repository]
    try:
        with urlopen(f'https://{host}/pypi/{name}/{version}/json', timeout=30):
            pass
    except HTTPError as error:
        if error.code == 404:
            return
        raise
    raise ValueError(f'{name} {version} already exists on {repository}; choose a new version')


def main() -> None:
    names = select(ROOT, os.getenv('REQUESTED_PACKAGES', ''),
                   os.getenv('RELEASE_TAG', ''), os.getenv('RELEASE_EVENT', ''))
    repository = os.getenv('RELEASE_REPOSITORY') or 'testpypi'
    for name in names:
        project = tomllib.loads(
            (ROOT / f'packages/omnibias-{name}/pyproject.toml').read_text()
        )['project']
        ensure_unpublished(project['name'], project['version'], repository)
    print(json.dumps({'packages': names, 'repository': repository}))
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
        print('pkgs=' + ' '.join(names), file=output)


if __name__ == '__main__':
    main()
