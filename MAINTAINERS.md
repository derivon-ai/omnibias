<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (C) 2026 Derivon -->

# Maintainers

| Role | Name | Contact | Scope |
|---|---|---|---|
| Founder, Lead Maintainer | Vardan Grigoryants | <vardan@derivon.ai> | all 42 packages |

Copyright is held by **Derivon** (<info@derivon.ai>), which is the project
steward and the counterparty to the [CLA](docs/CLA.md) and to commercial
licences.

## Current state: solo maintainer

omnibias has **one** maintainer. That is stated plainly rather than dressed up,
because it is the single most important thing for you to know when deciding
whether to depend on it. The practical consequences:

- **Review latency is one person's calendar.** Pull requests are reviewed as
  time allows; there is no rota and no on-call.
- **The bus factor is 1.** Mitigations in place: everything runs in public CI,
  every invariant that matters is a test rather than a convention, and the repo
  is self-describing ([`AGENTS.md`](AGENTS.md), `.cursor/skills/`) so a
  successor can pick it up. That reduces the risk; it does not eliminate it.
- **Security reports still get priority.** See [`SECURITY.md`](SECURITY.md) for
  the disclosure process and response targets.
- **Commercial support is a contract, not goodwill.** If you need a response
  commitment, that is what the commercial tier is for — see
  [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md).

## Area ownership

With one maintainer, [`.github/CODEOWNERS`](.github/CODEOWNERS) routes every
path to the same reviewer. It is structured per package anyway, so that adding
a maintainer for a subtree is a one-line change rather than a reorganisation.

## Becoming a maintainer

There is no committee to petition. The path is ordinary and evidence-based:

1. Land several non-trivial pull requests that meet the bar in
   [`CONTRIBUTING.md`](CONTRIBUTING.md) — regression test included, numerics
   bit-identical across backends, honest claims about what is proved.
2. Review others' pull requests substantively.
3. Take responsibility for a package or subsystem over time.

Commit access is then offered by the lead maintainer. See
[`GOVERNANCE.md`](GOVERNANCE.md) for the decision model this fits into.

## Emeritus

None yet.
