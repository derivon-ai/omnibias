# H3 direct Log-Noetherian format barrier

This note tests whether the actual singular return family can be put directly
into one uniformly bounded Log-Noetherian/exp format.  The earlier
[LN-passage assessment](HILBERT16-LN-PASSAGE.md) certified coordinate-chain
algebra but left the physical map, analytic extension, and norm bounds open.
Here the corrected kill sequence gives a decisive negative result for the
direct `tau`/`W` representation.

## Exact kill-sequence data

Take

\[
\epsilon=n^{-1},\qquad \operatorname{sep}=e^{-n^2}.
\]

For the existing outgoing section \(h_{\max}=\epsilon^3\), the natural
logarithmic matching functions satisfy

\[
\tau=\epsilon\log(1/\operatorname{sep})=n,\qquad
\log(W_{\max}/W_e)=2n.
\]

On every finite truncation \(1<|u|<N\), these functions form an exact
two-function LN chain:

\[
u\partial_u\tau=\tau,\qquad
u\partial_u\log(W_{\max}/W_e)=\log(W_{\max}/W_e).
\]

The chain length is two, its differential-polynomial degree is one, and its
largest explicit coefficient is two.  Thus the algebraic part does not
degenerate.  The analytic data do:

\[
\sup|\tau|+\sup|\log(W_{\max}/W_e)|=3N,
\]

and the annulus outer radius is \(N\).  Both grow without bound.  Since an LN
format includes the cell format and the supremum of its chain functions, no
parameter-independent format bound exists for this direct representation.

## What the result does not exclude

This is not a representation-independent impossibility theorem.  Multiplying
a displacement by an exact positive factor preserves its zeros.  A normalized
physical displacement could therefore avoid carrying the divergent matching
factor as a chain function.  To pass G3, such a proposal must still provide:

1. an exact zero-equivalence identity for the physical return displacement;
2. a differential-polynomial chain for the normalized physical map;
3. a uniform complex cell extension, including admission predicates and
   branches;
4. uniform coefficient and chain-function norm bounds.

No such normalization is currently constructed.  A variational upper bound
cannot be peeled off as though it were an exact factor.

## Verdict

`omnibias.dynamics.ln_format_barrier` certifies all four finite format data on
each bounded truncation and proves exact linear growth of the missing uniform
data:

```text
finite_truncations_certified = true
direct_bounded_format = false
actual_return_ln_membership_proved = false
normalized_zero_equivalent_route_open = true
g3_passed = false
full_hilbert16_solved = false
```

Thus H3 fails in its direct-return formulation.  The normalized route remains
an explicit open falsifier, so this is not a theorem against all LN or
multisummable approaches and not a solution of Hilbert's sixteenth problem.

Reproduce the finite assessment with:

```bash
python benchmarks/hilbert16_ln_format_barrier.py
```
