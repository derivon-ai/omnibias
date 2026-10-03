# Prospective license transition

The main repository retains 16 distributions: nine Apache and seven
AGPL-or-commercial. The separated ecosystem retains 44: 22 Apache, 21 dual and
one mixed Apache/dual distribution. Package metadata is authoritative for each
source edition; installed wheels carry the corresponding license texts.

## Existing grants

The [pre-transition inventory](license-transition-baseline.json) records versions,
licenses, repository commits and manifest hashes before this change. Source
attribution and Git authorship were inspected for the seven changing distributions;
the maintainer confirmed control of the relevant rights. This audit is evidence,
not a substitute for legal review of employer rights or third-party material.

Previously distributed Apache snapshots remain available under their original
grants. The transition changes prospective source editions, not history. No
existing PyPI artifact is replaced, no version is reused for publication, and no
release is made as part of this migration. A subsequent release requires an
explicit version decision and the normal publication checks.

## Upgrade optional integrations

The published core, torch, jax, keras, fields, FermiNet, PINN and geometry
packages retain Apache installation paths. Advanced integrations are explicit:

| Previous installation | Prospective installation |
| --- | --- |
| `omnibias-ferminet[curvature]` | `omnibias-research-certified[ferminet]` |
| `omnibias-geometry[atlas]` | `omnibias-research-certified[geometry]` |
| `omnibias-pinn[partition]` | `omnibias-research-certified[pinn]` |
| Research information geometry | `omnibias-research-certified[information]` |

These commands describe the coordinated future release, not currently published
PyPI availability. Use built local wheels for pre-release validation. Integration
module paths remain the same when the add-on is installed. Apache packages do not
install the add-on automatically, including through test or development extras.
The add-on preserves Apache bridge headers and the expression
`Apache-2.0 AND (AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial)`.

The PINN add-on also owns the symbolic-dependent `omnibias.pinn.certified`
and `omnibias.pinn.jax.discovery` entry points. The import audit moved their
12 integration modules alongside the 13 originally identified bridges; pure
Apache leaf utilities remain in PINN. FBPINN always uses its Apache weighted-sum
implementation, independently of which optional engines happen to be installed.

Future symbolic, hopfield and shape source editions use the dual tier.
Holonomic instead uses the exact Apache recurrence fitter in
`omnibias.difference.recurrence`; symbolic retains compatible re-exports.

The [licensing policy](https://github.com/derivon-ai/omnibias/blob/codex/pinn-substrate-split/LICENSING.md)
and [commercial offer](https://github.com/derivon-ai/omnibias/blob/codex/pinn-substrate-split/COMMERCIAL-LICENSE.md)
explain the available grants. Commercial terms never revoke historical Apache
permissions. Contributions still require the existing CLA process.
