# Open-core migration validation

This is a validation record for the prospective transition from commit
`115a417d6bee7687137f2f8a36a88dbe7df12c52`, not a release announcement.
The [historical inventory](../license-transition-baseline.json) preserves the
previous license and version of every distribution. The
[transition guide](../license-transition.md) explains the installation changes.

## Context ownership

One capability rule owns the agent-facing mechanisms and measured benchmark
summary. Maintenance instructions use one canonical skill directory in each
repository. There are 16 primitive skills and 31 consumer/research/supporting
repository skills, with 195–220 words per skill. The standalone installer bundles
43 distribution/research skills; the four application/laboratory skills stay in
their owning repositories.

| Context measure | Before | After |
| --- | ---: | ---: |
| Main always-loaded guidance, bytes | 8,021 | 4,467 |
| Main active guidance including all optional skills, bytes | 12,763 | 34,779 |
| Scoped ecosystem active guidance, bytes | 191,963 | 101,869 |
| Active agent benchmark blocks | 4 | 1 |

Always-loaded guidance shrank **44.31%**. The optional main catalog grew because
it now provides 16 focused maintenance skills instead of two broad skills. These
are byte measurements, not tokenizer counts; discovery descriptions also cost
context according to the harness. Loading a relevant skill is useful task context,
not startup waste. Historical archives are excluded from these measurements.

Independent maintenance exercises selected Torch, JAX, PINN and geometry context
and skipped numerical guidance for unrelated website work. Across the exercises,
nine files were consulted and all 34 local references resolved. The automated
guard checks skill coverage, unique descriptions, copied paragraphs, retired
discovery roots, references and the sole benchmark owner.

## Installation and legal boundaries

The main repository contains nine Apache and seven dual-licensed distributions.
The complete ecosystem contains **22 Apache, 21 dual and one mixed distribution**.
All 44 versions match the historical inventory. Existing Apache grants remain
available; changed source editions require new versions before any publication.

The independent wheel audit checks SPDX expressions, bundled license bytes,
RECORD hashes and disjoint file ownership. It also follows base and all-declared-
extras dependency closures for each of the eight published Apache packages.
All 16 metadata scenarios contain only Apache omnibias dependencies. Fourteen
existing dual packages now bundle the commercial references already named by
their licenses; their existing legal texts were preserved.

Twenty-five integration modules moved into the existing certified add-on: the
13 planned bridges plus 12 symbolic-dependent PINN discovery/certification
modules discovered by the import audit. Their Apache headers and import paths
remain intact. FBPINN uses its existing weighted sum unconditionally. Exact
recurrence fitting has one Apache implementation, reused by symbolic and
holonomic. All 29 local source maps are minimal: **147 overrides** after the
transition, including the laboratory and certified add-on.

## Validation evidence

All 44 wheels build from fresh source archives and pass packaging, Twine,
metadata and ownership checks. Isolated installations run outside checkouts
with source overrides disabled, no editable packages and checked import origins.
Each certified extra also installs independently without the test extra.

| Installed-wheel numerical profile | Passed | Skipped |
| --- | ---: | ---: |
| Apache FermiNet | 15 | 1 |
| Apache PINN | 171 | 3 |
| Apache geometry | 21 | 0 |
| Apache holonomic without symbolic | 10 | 0 |
| Certified representative suite | 123 | 0 |
| Remaining relocated integrations | 131 | 0 |

The optional skips are recorded by the existing tests, not new migration
exemptions. Separate checks cover FBPINN values and parameter gradients, 26
installer regressions, 17 ecosystem license-guard cases and stale setuptools
build caches. Installer cases cover selective installation/removal, shared
resources, edited and retired files, legacy manifests, and non-mutating checks.

The machine-readable [validation summary](open-core-validation.json) records
the wheel hashes and scoped results. Full local logs remain under
`artifacts/open-core-transition/`. Reproduction commands:

```bash
uv run --no-sync python scripts/check_agent_context.py --projects-root ../omnibias_projects
uv run --no-sync python scripts/generate_inventory.py --check
uv run --no-sync python scripts/generate_capability_evidence.py --check
uv run --no-sync python ../omnibias_projects/omnibias-skills/scripts/generate_bundle.py --check
uv run --no-sync python scripts/validate_wheels.py --projects-root ../omnibias_projects --numerical
```

The shared validator and per-repository profiles are reusable CI entry points.
No package was published, no version changed, and no application remote was
created. Existing CLA requirements remain in force. Commercial offers are
nonbinding and need counsel-reviewed agreements before grants are executed.
