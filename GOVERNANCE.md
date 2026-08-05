<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (C) 2026 Derivon -->

# Governance

This document says who decides what in omnibias, and how. It is deliberately
short and deliberately honest: the project has one maintainer, and pretending
otherwise would mislead anyone evaluating it as a dependency.

## Roles

| Role | Who | What they decide |
|---|---|---|
| **Founder / Lead Maintainer** | Vardan Grigoryants | technical direction, API and licence-tier changes, releases, who becomes a maintainer |
| **Steward (copyright holder)** | Derivon | legal matters: licensing, the CLA, trademark, commercial agreements |
| **Maintainers** | see [`MAINTAINERS.md`](MAINTAINERS.md) | review and merge within their area |
| **Contributors** | anyone who has signed the [CLA](docs/CLA.md) | propose changes |

## The decision model, stated honestly

**Benevolent-dictator, solo.** The lead maintainer decides. There is no voting
body, no steering committee, and no formal RFC process, because with one
maintainer those would be ceremony rather than governance.

What constrains that authority in practice is not a committee but the test
suite. The project's load-bearing invariants are *executable*, so a decision
that violates one fails CI regardless of who made it:

- cross-backend numerics are bit-identical by construction (parity tests);
- no permissive package may depend on a copyleft package
  (`test_license_consistency.py`);
- no internal paths, scheduler tokens, or secrets in tracked files
  (`test_no_leakage.py`);
- terminology and conceptual lineage stay accurate (`test_terminology.py`,
  `test_lineage_declared.py`);
- `theorem_prover_verified` is set only by a genuine Lean kernel pass, and
  asserting it without one blocks the verdict.

Those are the real governance. Adding a maintainer changes who can merge;
it does not change what the build will accept.

## How changes are proposed

1. **Small fixes** — open a pull request. Read
   [`CONTRIBUTING.md`](CONTRIBUTING.md) first; every behavioural change needs a
   regression test.
2. **Larger or structural changes** — open an issue describing the problem
   before writing code. Anything touching the derivative-tower contract, the
   public API surface, the licence tiers, or a certificate's soundness needs
   agreement on the approach first.
3. **New packages** must earn independent existence — a distinct domain, a
   distinct dependency / maturity tier, or a distinct audience. If a proposal
   fails that test it ships as a submodule of an existing package. See the
   `omnibias-dev-new-package` skill and the "Don't" list in
   [`AGENTS.md`](AGENTS.md).

## Disagreement

Argue it in the issue or pull request, on the technical merits, in public. If
no agreement is reached, the lead maintainer decides and says why. Both the
disagreement and the reasoning stay on the record.

## Releases

Each of the 42 distributions is versioned independently. The lead maintainer
cuts releases; publishing runs through PyPI trusted publishing (OIDC) from
[`.github/workflows/release.yml`](.github/workflows/release.yml), so no
long-lived credential exists to be shared or leaked. The gate a release must
pass is recorded in [`docs/release-readiness.md`](docs/release-readiness.md).

## Licence and relicensing

The tier of each package is recorded in `[tool.omnibias.license_tiers]` in the
root [`pyproject.toml`](pyproject.toml) and explained in
[`LICENSING.md`](LICENSING.md). Moving a package between tiers is a **steward**
decision, made possible by the CLA's grant, and is announced in
[`CHANGELOG.md`](CHANGELOG.md).

Moving a package *from* copyleft *to* permissive is always available. Moving
one from permissive to copyleft applies only to future versions: releases
already published under Apache-2.0 stay Apache-2.0 forever, and nothing here
can retract them.

## Contact

| Topic | Address |
|---|---|
| Steward / commercial / legal | <info@derivon.ai> |
| Founder / lead maintainer | <vardan@derivon.ai> |
| Security reports | see [`SECURITY.md`](SECURITY.md) |
| Code of conduct | <info@derivon.ai> (subject `CODE OF CONDUCT:`) |

## Code of conduct

Enforcement is handled by the lead maintainer. See
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Security

Vulnerability reports follow [`SECURITY.md`](SECURITY.md) and take priority
over feature work.

## If the maintainer becomes unavailable

Derivon holds the copyright and the CLA grants, and can appoint a new
maintainer. Should the project be abandoned outright, the permissive tier is
already Apache-2.0 and can be forked and continued by anyone, with no action
required from us. That is a deliberate property of the tier split, not an
accident.
