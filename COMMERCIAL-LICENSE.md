<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (C) 2026 Derivon -->

# omnibias Commercial License (summary)

> This document is a **plain-language summary** of the commercial license
> offer. It is **not** the binding contract and is **not legal advice**. The
> binding terms are in the signed commercial license agreement issued by
> **Derivon**. To request the agreement, contact **info@derivon.ai**.

## First: check whether you actually need one

**Most of omnibias does not require a commercial license, ever.** The 28
packages in the permissive tier — the derivative tower and everything built
directly on it, including `omnibias-core`, `omnibias-torch`, `omnibias-jax`,
`omnibias-keras`, `omnibias-fields`, `omnibias-pinn`, and `omnibias-geometry` —
are **Apache-2.0**. Ship them in a closed-source product, run them behind a
hosted API, redistribute them: no copyleft, no §13 disclosure, no conversation
with us required.

This page is only about the **14 copyleft-tier packages**:

`omnibias-verify`, `omnibias-formal`, `omnibias-sos`, `omnibias-dynamics`,
`omnibias-convex`, `omnibias-discrete`, `omnibias-qubo`, `omnibias-logic`,
`omnibias-nphard`, `omnibias-submodular`, `omnibias-combinatorics`,
`omnibias-routing`, `omnibias-tab`, `omnibias-control`.

Their SPDX expression is `AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial`.
See [`LICENSING.md`](LICENSING.md) for the full tier table.

Note also that no permissive package depends on a copyleft one, through
required or optional dependencies — enforced in CI. So installing a Tier-P
package with any set of extras cannot pull an AGPL package into your tree and
cannot create an obligation by accident.

## When you need this

You need a commercial license if you use one of the 14 packages above and do
**not** want to comply with the [GNU AGPL-3.0](LICENSES/AGPL-3.0-or-later.txt)
— most commonly because you intend to:

- ship a copyleft-tier package inside a **proprietary / closed-source**
  application;
- run a **hosted or SaaS** offering on top of a modified copyleft-tier package
  without publishing your modified source (the AGPL §13 trigger);
- redistribute a copyleft-tier package under terms other than the AGPL; or
- require a **warranty, indemnity, or support commitment**.

If you are doing open-source work, internal research, or are happy to comply
with the AGPL, you do **not** need this — just use the AGPL branch.

## What the commercial license grants (typical terms)

Subject to the signed agreement, a commercial license typically grants:

- a non-exclusive, worldwide, royalty-bearing (or paid-up) license to use,
  modify, and distribute the covered packages **without** the AGPL copyleft /
  §13 source-disclosure obligations;
- the right to distribute them in **object/binary** form as part of a larger
  proprietary work;
- optional **support, maintenance, and warranty** terms; and
- optional **patent assurances** and **indemnification**, by negotiation.

## What it does not grant

- Any rights over the permissive tier — those are already Apache-2.0 and a
  commercial agreement neither adds to nor restricts them.
- Rights to the **"omnibias" trademark** beyond fair / nominative use
  (see [`TRADEMARKS.md`](TRADEMARKS.md)).
- Any ownership of the copyright, which is retained by **Derivon**.

## Pricing & tiers

Pricing is by agreement and typically scales with company size and use case
(startup / SMB / enterprise / OEM-redistribution). Academic and
single-developer rates may be available.

## How to obtain a license

1. Email **info@derivon.ai** with your company name, which copyleft-tier
   packages you need, your intended use (embedded vs hosted), and expected
   scale.
2. You will receive the commercial license agreement and a quote.
3. On signature and payment, you receive a written grant and, if purchased, a
   support channel.
