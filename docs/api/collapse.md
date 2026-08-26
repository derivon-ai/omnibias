# Named-collapse schema

The founding three senses of collapse stay exactly those three:

- **bias** (`delta -> 0`) yields a derivative
- **temperature** (`beta -> inf`) yields a 0/1 feasibility indicator
- **enclosure** (`width -> 0` of a *sound enclosure*) yields a point plus a proof, or `Inconclusive`

`omnibias.core.collapse` catalogues them and refuses a new name that is
a rebrand (same moving parameter *and* same surviving object). A later
named collapse earns a slot only when it mints a different object.
This is not a package. A float residual is never a proof.
`theorem_prover_verified` stays false unless a genuine `lake build`
earned it. Do not conflate the three founding senses.

Home: `omnibias.core.collapse` (schema + registry). No new distribution.
The prove/disprove router over these kinds is
[`omnibias.core.proof.engine`](proof_engine.md).

Verdict collapse (`omnibias.core.collapse.verdict`) is the first named
sense that earned a slot: a sound residual enclosure of a finite
obligation becomes `PROVED` only at `{0}`, `DISPROVED` when `0` is
excluded, and `BLOCKED` otherwise. A float residual is not a proof.

Identity collapse (`omnibias.core.collapse.identity`) decides a germ
identity: exact `Q` coefficient agreement is `{0}`; a float `||R_N||`
is not a proof.

Winding collapse (`omnibias.core.collapse.winding`) encloses
`Δarg / 2π` on a circle and accepts only a unique integer. Not a
blow-up proof.

Pairing collapse (`omnibias.core.collapse.pairing`) is a certified
weak residual on a finite test pack, not a strong solution.

Rank collapse (`omnibias.core.collapse.rank`) is an exact `Q` syzygy.
A float SVD is not a proof.

Duality / gap collapse was evaluated and **rejected**: a primal-dual
sandwich that collapses at `L = U` is Enclosure Collapse of `OPT`.

Honorable mentions that were evaluated and **not shipped**: cocycle
(identity of a jet defect), path (already homotopy 09-20),
`q -> 1` / `μ -> 0` specialization (recovers the ordinary derivative),
scale / RG (03-07 tempering, not a collapse), Morse (already 03-09),
and policy entropy (temperature collapse of a search heuristic).
Founding surviving objects (`derivative`, `indicator`,
`point_plus_proof`) cannot be reused.

::: omnibias.core.collapse
    options:
      show_root_heading: false
      heading_level: 3
