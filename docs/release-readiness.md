<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (C) 2026 Derivon -->

# Release readiness -- omnibias curated core

**Founder / release owner:** Vardan Grigoryants (vardan@derivon.ai)
**Copyright holder:** Derivon (info@derivon.ai)
**Target repository:** `github.com/derivon-ai/omnibias`
**Verification run:** 2026-08-05 (CPU gate, post-README / public-benchmarks polish)
**Verdict:** **GO on the engineering gate** -- every automated gate below is
green. Three **owner actions** remain before a tag can be pushed; they are
external to the repository and are listed under
[Blocking owner actions](#blocking-owner-actions).

This is the single, authoritative go / no-go sign-off for the public release.
Companion documents:

- Licensing (two tiers, and the invariant between them): [`LICENSING.md`](../LICENSING.md)
- Governance and maintainer roster: [`GOVERNANCE.md`](../GOVERNANCE.md), [`MAINTAINERS.md`](../MAINTAINERS.md)
- Blocker / limitation register: [`release_blockers.md`](release_blockers.md)
- API-stability contract: [`docs/stability.md`](docs/stability.md)
- Reproduction (pinned toolchain, seeds, selectors): [`docs/reproducibility.md`](docs/reproducibility.md)

## Curated-core release set

Published first, under the [API-stability contract](docs/stability.md). The
`pyproject` metadata is the single source of truth for each version; every
distribution derives `__version__` from the installed metadata. Use
`python scripts/bump_version.py` rather than editing versions by hand.

| Distribution      | Version  | Import package      | Licence tier | RC tag                        |
|-------------------|----------|---------------------|--------------|-------------------------------|
| omnibias-core     | 0.4.0    | `omnibias.core`     | Apache-2.0   | `omnibias-core-v0.4.0rc1`     |
| omnibias-torch    | 0.4.0    | `omnibias.torch`    | Apache-2.0   | `omnibias-torch-v0.4.0rc1`    |
| omnibias-jax      | 0.4.0    | `omnibias.jax`      | Apache-2.0   | `omnibias-jax-v0.4.0rc1`      |
| omnibias-ferminet | 0.2.0    | `omnibias.ferminet` | Apache-2.0   | `omnibias-ferminet-v0.2.0rc1` |
| omnibias-keras    | 0.0.1a1  | `omnibias.keras`    | Apache-2.0   | `omnibias-keras-v0.0.1a1`     |
| omnibias-fields   | 0.1.0    | `omnibias.fields`   | Apache-2.0   | `omnibias-fields-v0.1.0rc1`   |
| omnibias-pinn     | 0.1.0    | `omnibias.pinn`     | Apache-2.0   | `omnibias-pinn-v0.1.0rc1`     |
| omnibias-geometry | 0.2.0    | `omnibias.geometry` | Apache-2.0   | `omnibias-geometry-v0.2.0rc1` |

The whole curated core is permissive, which is the point of the split: the
adoption surface carries no copyleft obligation. The remaining 34 distributions
are Alpha on the extended track and are **not** under the stability contract.
See [`docs/packages.md`](docs/packages.md) for the full grouped inventory.

## Verification gate (all green)

Run on CPU: Python 3.11 / 3.14, torch CPU wheels, `JAX_PLATFORMS=cpu`,
`KERAS_BACKEND=torch`. The GPU-cluster fidelity legs (G1-G7 in
[`release_blockers.md`](release_blockers.md)) are **not**
release-gating for the versions shipped today and are tracked separately.

| Gate                    | Command                                                                 | Result                                   |
|-------------------------|-------------------------------------------------------------------------|------------------------------------------|
| Lint                    | `ruff check packages tests scripts`                                     | pass                                     |
| Type -- T1              | `mypy --strict {core,torch,jax,ferminet}/src`                           | pass (114 source files)                  |
| Type -- curated beta    | `python scripts/mypy_strict_check.py`                                   | pass (13 authored-strict)                |
| Docs                    | `mkdocs build --strict`                                                  | pass (no broken anchors)                 |
| Executable docs         | `pytest tests/test_docs_snippets.py`                                    | pass (27 run, 58 opted out with reasons) |
| Workspace pytest        | `pytest packages/{core,torch,jax,ferminet}/tests tests`                 | pass (3540 passed, 79 skipped)           |
| **Licence consistency** | `pytest .../test_license_consistency.py`                                | **pass (11 checks)**                     |
| **SPDX headers**        | `python scripts/license_headers.py --check`                             | **pass (0 drifted of 1786)**             |
| Release guards          | placeholders / version / leakage / lineage / `__all__` / `py.typed` / terminology / registry | pass |
| Build + metadata        | `python -m build` (x42) + `twine check`                                 | pass (42 wheels + 42 sdists, 84 PASSED)  |
| **PEP 639 metadata**    | `License-Expression` in every wheel matches its tier                    | **pass (42/42)**                         |
| Packaging hygiene       | `python scripts/check_packaging.py`                                     | pass (no junk in any dist)               |
| Wheel import smoke      | all 42 wheels into one clean venv + `scripts/import_smoke.py`           | pass (42/42 imported)                    |
| Agent-skill drift       | `omnibias-skills install --check`, `scripts/sync_skills.py --check`     | pass (12 up-to-date, mirror in sync)     |

Package suites re-run for this gate after the licensing and concept-fidelity
changes: `pinn`, `symbolic`, `sos`, `formal` (1558 passed, 7 skipped);
`verify`, `discrete`, `convex`, `fields`, `geometry`, `struct`, `tab`,
`partition` (1635 passed, 2 skipped); `holonomic`, `timescale`, `qcalculus`
(356 passed). The full 42-package matrix runs as independent per-package CI
jobs, one per distribution.

## Licensing posture (new since the previous sign-off)

omnibias is **open-core**, in two tiers recorded in
`[tool.omnibias.license_tiers]` in the root [`pyproject.toml`](pyproject.toml):

- **Tier P -- `Apache-2.0`** (28 packages): the derivative tower and everything
  built directly on it. Express patent grant; usable in closed-source and
  hosted products; **never** requires a commercial licence.
- **Tier C -- `AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial`**
  (14 packages): the certified-decision layer, where the commercial offer lives.

Three mechanical properties are enforced rather than documented:

1. **The DAG invariant.** No permissive package depends on a copyleft package,
   through `dependencies` or any `optional-dependencies` extra. Verified across
   all 42 distributions;
   [`test_license_consistency.py`](packages/omnibias-core/tests/test_license_consistency.py)
   fails the build if such an edge is ever added, and self-tests that the check
   is not a tautology.
2. **Single source of truth.** The tier table drives the per-package `LICENSE`
   file, the PEP 639 `license` expression, and the SPDX header on all 1,786
   `.py` files. `scripts/license_headers.py` is the only writer.
3. **Published metadata is correct.** All 42 wheels carry Metadata-Version 2.4
   with the right `License-Expression`, including the `LicenseRef-` dual
   expression, which `twine check` and `packaging`'s SPDX canonicaliser both
   accept.

Repository layout follows the [REUSE Specification](https://reuse.software):
full texts in `LICENSES/`, root `LICENSE` left as AGPL so automated detectors
report the strongest licence in the tree.

## Enforced on every public tag (CI)

Static, backend-free guards that must stay green to tag (they run in the
`guards` and `lint` jobs and in `omnibias-core`'s test job):

- **Licence consistency** -- tier / `LICENSE` / SPDX header / PEP 639 agreement,
  and the permissive-never-depends-on-copyleft invariant
  (`test_license_consistency.py`, plus `scripts/license_headers.py --check`).
- **Placeholders** -- no residual `<...>` identity token in any release-surface
  file (`test_no_placeholders.py`).
- **Version single-source-of-truth** -- `pyproject` matches every in-code
  `__version__`, no stray submodule markers, and **every sibling pin is
  satisfiable by the version in the tree** so no package becomes uninstallable
  on publication (`test_version_consistency.py`).
- **Leakage / secrets** -- no local paths, batch-scheduler commands / directives
  / environment variables from any vendor, site mounts, personal accounts or
  emails, corporate names, private-tree references, or credentials in any
  readable file (`test_no_leakage.py`, which self-tests its blocklist against
  synthetic bait for every leak family).
- **Conceptual lineage** -- each package's declared `__lineage__` agrees with
  its own description and docstrings, so bias collapse and temperature collapse
  can never be conflated in shipped prose (`test_lineage_declared.py`,
  `test_terminology.py`, `test_concept_terminology.py`).
- **Sorted `__all__`** -- every export list is ordered (`test_all_sorted.py`).
- **`py.typed`** -- every distribution ships a wired-up PEP 561 marker.
- **Package registry** -- folded modules (`flow` / `pde` / `gauge`) are not
  mis-declared as distributions (`test_package_registry.py`).
- **Packaging cleanliness** -- `build_wheels` builds and twine-checks every
  distribution and runs `scripts/check_packaging.py`.
- **Wheel import smoke + curated clean-install matrix** -- every wheel installs
  into a clean venv and imports.

## Supply-chain posture

- **Publishing:** PyPI trusted publishing (OIDC) only. No long-lived API token
  exists in secrets.
- **Provenance:** [`release.yml`](.github/workflows/release.yml) emits SLSA
  build-provenance attestations for every artifact. Consumers verify with
  `gh attestation verify <dist> --repo derivon-ai/omnibias`.
- **Scanning:** CodeQL, OpenSSF Scorecard, dependency review, and SBOM
  generation run in CI. All third-party actions are SHA-pinned.
- **Contributions:** gated by a CLA bot writing to `derivon-ai/cla-signatures`.

## Blocking owner actions

These are external to the repository and cannot be automated from here.

- [ ] **Create `github.com/derivon-ai/omnibias`** and set the description to:
      *Closed-form n-th derivatives of neural activations in a single forward
      pass — bit-identical across PyTorch, JAX, and Keras 3.* Suggested topics:
      `automatic-differentiation`, `higher-order-derivatives`,
      `scientific-machine-learning`, `pytorch`, `jax`, `keras`,
      `physics-informed-neural-networks`, `numerical-analysis`,
      `formal-verification`, `computational-mathematics`.
- [ ] **Reserve all 43 PyPI names** (`omnibias` plus the 42 `omnibias-*`), and
      configure a trusted publisher per project pointing at `release.yml` and
      the `testpypi` / `pypi` GitHub Environments. Do one TestPyPI upload of
      `omnibias-core` and one copyleft-tier package to confirm the index
      accepts the `LicenseRef-` dual expression end to end; `twine check` and
      the `packaging` canonicaliser already accept it locally.
- [ ] **Create `derivon-ai/cla-signatures`** and the fine-grained PAT the CLA
      workflow writes with.

Recommended, not blocking: file **"omnibias"** as a trademark under Derivon.
For a mathematics library that is a stronger defensibility lever than the
licence.

## Known limitations (documented, accepted -- not blockers)

Full register in [`release_blockers.md`](release_blockers.md).
Highlights:

- **omnibias-keras** ships curated at Alpha (`0.0.1a1`): the activation-level
  math is bit-identical to the torch/jax core by construction (same
  `omnibias-core` coefficients), but the Keras 3 wrapper surface may still shift
  between alpha releases.
- **Extended-set API instability**: the 34 alpha distributions may change their
  public surface between alpha releases; only the curated core is frozen.
- **`mypy --strict` scope**: enforced on T1 (`core/torch/jax/ferminet`) and the
  curated-beta authored-strict `_core` substrate; full-alpha strict typing is
  tracked per package and grows through
  `scripts/mypy_strict_allowlist.txt`.
- **Solo maintainer**: bus factor 1, stated plainly in
  [`MAINTAINERS.md`](../MAINTAINERS.md). Mitigated by executable invariants and a
  self-describing repository, not eliminated.

## Cutting the release candidates (human sign-off required)

Tagging triggers the TestPyPI publish workflow
([`.github/workflows/release.yml`](.github/workflows/release.yml)), so it is a
deliberate release-owner action and is intentionally **not** automated here.
From a clean, green `main`, with the owner actions above complete:

```bash
# one PEP 440 RC tag per curated-core distribution
git tag omnibias-core-v0.4.0rc1
git tag omnibias-torch-v0.4.0rc1
git tag omnibias-jax-v0.4.0rc1
git tag omnibias-ferminet-v0.2.0rc1
git tag omnibias-keras-v0.0.1a1
git tag omnibias-fields-v0.1.0rc1
git tag omnibias-pinn-v0.1.0rc1
git tag omnibias-geometry-v0.2.0rc1
git push origin --tags
```

**RC versioning note:** the workflow publishes the version currently in each
`pyproject`. To ship a distinct RC *version string* on TestPyPI (e.g.
`0.4.0rc1`), bump on a release branch first --
`python scripts/bump_version.py core --set 0.4.0rc1` -- otherwise the tag
publishes the final version shown in the table above. `skip-existing` keeps
re-tags idempotent. `workflow_dispatch` can publish a chosen subset without a
tag.

## Sign-off

- [x] All release-gating guards green (see the verification table).
- [x] All 42 distributions build clean, pass `twine check`, install into one
      clean venv, and import.
- [x] Licence split implemented, published in metadata, and enforced by a CI
      invariant.
- [x] Founder attribution (Vardan Grigoryants) on all 42 distributions,
      `CITATION.cff`, `.zenodo.json`, `NOTICE`, and `README.md`; no personal
      address published.
- [x] No internal-reference leakage; no unfilled identity placeholders.
- [x] Honesty / terminology / lineage guards green (no over-claims).
- [ ] **Release owner** completes the three blocking owner actions.
- [ ] **Release owner** cuts the RC tags above (final human action).

*Prepared for Derivon. Every gate above is reproducible via
[`docs/reproducibility.md`](docs/reproducibility.md).*
