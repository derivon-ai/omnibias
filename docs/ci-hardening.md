# CI supply-chain hardening

Short notes on the GitHub Actions posture for this repository. See also
[CONTRIBUTING.md](https://github.com/derivon-ai/omnibias/blob/main/CONTRIBUTING.md)
and [AGENTS.md](https://github.com/derivon-ai/omnibias/blob/main/AGENTS.md)
(repository root; not part of the MkDocs nav).

## Permissions

Every workflow defaults to least privilege:

```yaml
permissions:
  contents: read
```

Jobs escalate only when required (`pages: write` + `id-token: write` for docs
deploy, `security-events: write` for CodeQL / Scorecard SARIF upload,
`id-token: write` for PyPI Trusted Publishing).

## SHA-pinned actions

Third-party actions are pinned to a full commit SHA with a version comment, e.g.

```yaml
uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
```

Pins were resolved from GitHub tag refs at hardening time. Dependabot's
`github-actions` ecosystem keeps them current via weekly PRs.

| Action | Pinned tag |
| --- | --- |
| `actions/checkout` | v4.2.2 |
| `actions/setup-python` | v5.6.0 |
| `astral-sh/setup-uv` | v5.4.2 |
| `actions/upload-artifact` | v4.6.2 |
| `actions/download-artifact` | v4.3.0 |
| `actions/cache` | v4.2.3 |
| `actions/upload-pages-artifact` | v3.0.1 |
| `actions/deploy-pages` | v4.0.5 |
| `lycheeverse/lychee-action` | v2.3.0 |
| `pypa/gh-action-pypi-publish` | v1.12.4 |
| `contributor-assistant/github-action` | v2.6.1 |
| `github/codeql-action/*` | v3.28.19 |
| `ossf/scorecard-action` | v2.4.2 |
| `actions/dependency-review-action` | v4.7.1 |
| `anchore/sbom-action` | v0.20.1 |

## Timeouts and caching

Every job sets `timeout-minutes` (lint ~20, package tests ~45, docs ~30, wheels
~60, Lean ~90). Jobs that use `astral-sh/setup-uv` enable `enable-cache: true`.

## Security workflows

| Workflow | Purpose |
| --- | --- |
| `codeql.yml` | CodeQL Python analysis on push/PR to `main` |
| `dependency-review.yml` | PR dependency review (`fail-on-severity: high`) |
| `scorecard.yml` | OpenSSF Scorecard + SARIF upload |
| `sbom.yml` | CycloneDX SBOM on release tags (artifact upload) |

## Release / Trusted Publishing

`release.yml` publishes curated-core to TestPyPI (tag push / dispatch) or
production PyPI (`workflow_dispatch` with `repository=pypi`) using **OIDC
Trusted Publishing only** -- no `TEST_PYPI_API_TOKEN` / long-lived secrets.

Before the first publish, configure:

1. GitHub Environments `testpypi` and `pypi`.
2. A PyPI / TestPyPI trusted publisher per project pointing at this repository,
   workflow file, and environment.

## Coverage floor

The `core` CI job runs with `--cov=omnibias.core --cov-fail-under=60`. Raise the
floor after a coverage pass; do not lower it to paper over regressions.

## Python interpreter policy

- Dev target: **3.14**; supported floor: **3.10** (`requires-python` unchanged).
- `core` / `torch` / `jax` matrix: `{3.10, 3.11, 3.12, 3.13, 3.14}`.
- Keras tensorflow leg: Python **<=3.13** (no stable TensorFlow cp314 wheels).
- JAX may drop older CPythons upstream; exclude those matrix cells rather than
  changing omnibias `requires-python`.

## Lockfile

`uv.lock` is **tracked** for enterprise / CI reproducibility. It does not
constrain published wheel dependency ranges.
