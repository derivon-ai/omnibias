# Forest projections with fixed boundary vertices

The finite API <code>gauge_forest_projection</code> checks when available
vertex gauge transformations act transitively on an observed set of links.
It supplies a conditional implication for a gauge-invariant probability
measure and observable. It does **not** construct that measure or certify an
actual vacuum.

For a nontrivial compact gauge group, the exact criterion is:

1. The observed multigraph is a forest.
2. Each observed connected component contains at most one fixed vertex.

The empty observed set passes. Observed selfloops and pairs of parallel links
are cycles and fail the criterion. Cycles involving unobserved links do not
matter. A failed gate returns <code>INCONCLUSIVE</code> for projection collapse,
with an explicit cycle or fixed-terminal path obstruction.

## 1. Gauge action and the exact topological criterion

Orient each original link \(e=(u,v)\). The available gauge group acts by

\[
 U_e\longmapsto g_uU_eg_v^{-1},\qquad
 g_v=1\quad\hbox{at every fixed vertex}.
\tag{1}
\]

All other vertex transformations must be independently available. The API
uses zero-based vertex IDs and positive, one-based observed edge IDs.
Link orientation is retained as data; reversing it does not change the
transitivity test.

Suppose an observed component is a tree with at most one fixed vertex.
Root it at that fixed vertex, or at any vertex if none is fixed, and set
the root gauge to the identity. Traverse away from the root. For a
parent-to-child oriented token \(t\), set

\[
 g_{\rm child}=g_{\rm parent}U_{|t|}^{\operatorname{sign}(t)}.
\tag{2}
\]

The transformed link is the identity, in either original orientation.
Every nonroot vertex is available, so no forbidden gauge transformation is
used. Applying this independently to the observed components sends every
observed configuration to the all-identity configuration. The action is
therefore transitive. Gauge transformations may change unobserved links;
those links are integrated out, not held fixed by this operation.

The converse has explicit invariants. Around an observed cycle, the holonomy
transforms by conjugation. Its conjugacy class distinguishes identity from a
nonidentity holonomy. Along a tree path between two fixed vertices, the
holonomy itself is invariant. Thus either obstruction prevents transitivity
for every nontrivial compact gauge group. In SU(2), a trace distinguishes the
two configurations obtained by assigning \(I\), respectively \(-I\), to one
path or cycle edge and identity to the other edges. Their traces are \(2\)
and \(-2\).

This argument also handles a selfloop, whose one-edge holonomy is conjugated,
and a parallel-link cycle, whose two-edge holonomy need not be the identity.
These are valid multigraph obstructions, not malformed input.

The witness records each observed component, its cycle rank, fixed vertices,
a root, and an oriented spanning-tree gauge order. In failed components it
also records a closed oriented cycle or a path between fixed endpoints.
Spanning-tree data on a failed component are diagnostic; they do not assert
that all fixed vertices can be kept fixed during elimination.

## 2. Conditional expectation and the physical subspace

Let \(\mu\) be a probability measure on the full original-link space, invariant
under every transformation in (1). Let \(F\in L^1(\mu)\) have the same
invariance, and let \(P_TF=\mathbb E_\mu[F\mid U_T]\), where \(T\) is the observed
edge set. Gauge transformations preserve the observed sigma-algebra. The
uniqueness of conditional expectation therefore implies equivariance:

\[
 P_T(F\circ g)=(P_TF)\circ g
 \quad\hbox{as equivalence classes modulo }\mu.
\tag{3}
\]

Consequently \(P_TF\) is available-gauge invariant. If the finite criterion
passes, the observed action is transitive, so

\[
 \boxed{\mathbb E_\mu[F\mid U_T]=\mathbb E_\mu F.}
\tag{4}
\]

There is no hidden pointwise-version assumption in this step. Average the
observed function over Haar probability on the compact available gauge group.
Equivariance leaves its \(L^1(\mu)\) class unchanged. Transitivity makes the
averaged function constant; integrating identifies the constant as
\(\mathbb E_\mu F\). Equivalently the invariant observed marginal is the
unique invariant probability on the transitive orbit. For a rooted forest
this is product Haar on its observed links.

For \(F\in L^2(\mu)\), (4) means that the observed orthogonal projection equals
the constant projection on the available-gauge-invariant subspace. On its
centered part the projection is zero. No density lower bound, spectral gap,
perturbative expansion, Wilson Gibbs assumption or ground-state estimate is
needed for this conditional implication.

For a gauge-invariant Hamiltonian whose actual vacuum measure is independently
known to have the stated invariance, the conclusion applies to that measure.
This API does not perform that source identification. Its
<code>actual_vacuum_verified</code>, <code>actual_measure_verified</code> and
<code>input_gauge_invariance_verified</code> flags always remain false.

The current [physical-block theta example](gauge-physical-blocks.md) illustrates
the distinction: both observed blocks there are globally forests, so their
projections vanish on centered globally physical functions, even though the
unrestricted scalar overlap can be nonzero. Numerical Haar minorants are
unnecessary for that symmetry-restricted statement.

## 3. Frozen exterior: the necessary qualification

In an exterior-conditioned application, \(\mu\) in (4) is the actual conditional
measure \(\mu_z\), and only gauge transformations preserving the specified
exterior \(z\) may be used. Every vertex whose transformation changes frozen
data must be listed as fixed. The ordinary internal-vertex choice makes all
vertices incident to frozen exterior links fixed; further restrictions must
also be recorded. The checker cannot infer an omitted ambient embedding.

Under these premises, (4) becomes

\[
 \mathbb E_{\mu_z}[F_z\mid U_T]=\mathbb E_{\mu_z}F_z.
\tag{5}
\]

The right side may depend on \(z\). This does not replace a conditional
projection by a global constant.

A forest component with two fixed terminals retains their open-path
transport. These are physical boundary-flux degrees of freedom for the
internal gauge group. Deleting them by imposing an additional neutral
boundary condition would change the Hilbert space.

For a concrete counterexample, start with a single square and its physical
Wilson trace. Freeze one complementary path to the identity. The remaining
observed path is a tree with two fixed endpoints; the Wilson trace becomes
the trace of that open-path holonomy and is nonconstant. It survives its
observed conditional projection. This applies, in particular, to the
strictly positive actual finite-square vacuum density. The forest theorem
with the full vertex group cannot be reused after these endpoint gauges
have been frozen.

## 4. Exact negative controls and their scope

The recorded cycle and path words provide a particularly simple measure
control: normalized product Haar on every original SU(2) link. A simple
cycle or simple path uses distinct links, so its product is Haar. The
fundamental character has

\[
 \int\chi_{1/2}\,dH=0,\qquad
 \int|\chi_{1/2}|^2\,dH=1.
\tag{6}
\]

The observable is invariant under all available gauges: take its trace
around a cycle, or its trace between fixed path endpoints. It is already
observed, so \(P_TF=F\ne0=\mu F\). This disproves a universal projection
collapse implication for the failing topology. It does not assert
noncollapse for every particular unknown measure, which could be supported
on a single orbit, and it is not a supplied Yang–Mills vacuum.

Independent tests enumerate the entire \(\mathbb Z_2\) vertex gauge action for
all 64 observed subgraphs of a four-vertex complete graph and all 16 fixed
vertex sets. They also exhaust a small multigraph containing selfloops and
parallel links. Transitivity is checked by directly enumerating the orbit of
the identity, without reusing the spanning-tree algorithm. Other tests apply
the recorded elimination to noncommuting rational SU(2) quaternions and check
the nonconstant path and cycle controls.

These finite tests are regression evidence. The arbitrary compact-group
statement is the constructive proof (2), not an extrapolation from
\(\mathbb Z_2\) or from a random sample.

## 5. API and replay

    from omnibias.geometry.gauge.transfer.forest_projection import (
        gauge_forest_projection,
        replay_gauge_forest_projection_certificate,
    )

    forest = gauge_forest_projection(
        3, [(0, 1), (1, 2)], [1, 2], fixed_vertices=[0],
    )
    assert forest["status"] == "PASS"
    assert not forest["actual_measure_verified"]
    assert replay_gauge_forest_projection_certificate(forest["certificate"])

    path = gauge_forest_projection(
        3, [(0, 1), (1, 2)], [1, 2], fixed_vertices=[0, 2],
    )
    assert path["status"] == "INCONCLUSIVE"
    assert path["obstructions"][0]["kind"] == "multiple_fixed_terminals"

Vertex count and IDs must be exact Python integers; floats, booleans,
fractions and text are rejected. Endpoint pairs and ID inputs must be
sequences. Duplicate observed or fixed IDs, out-of-range endpoints and
negative vertex counts are errors. An empty graph with zero vertices is
legal. Input ID lists are canonicalized by sorting; original edge order and
orientation are preserved.

The sealed schema is <code>gauge_forest_projection_v1</code>. Canonical replay
rebuilds the entire certificate, including input order, graph witnesses,
obstructions, conditional premises, metadata and honesty flags. Rehashing
forged source or parent claims does not make them valid. No Hamiltonian gap,
infinite-volume construction, continuum theorem, Lean tier or Clay parent
is earned by this finite topology certificate.
