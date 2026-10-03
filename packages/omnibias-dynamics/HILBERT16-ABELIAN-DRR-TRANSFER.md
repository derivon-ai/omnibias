# H4 Picard--Fuchs to physical-return transfer

The holonomic and Abelian tracks already certify exact Picard--Fuchs
annihilators and instance-tight zero counts for named regular elliptic
Hamiltonian ovals.  This note tests the missing step proposed in H4: whether
that result closes an individual Design--Roussarie--Rousseau graphic.

It does not.  The test does produce a useful new conditional transfer theorem
and isolates exactly which physical data are absent.

## Conditional transfer theorem

Suppose a physical displacement has the uniform expansion

\[
\Delta_\epsilon(h)
=\epsilon I(h)+\epsilon^2R_\epsilon(h).
\]

Assume a certified cover proves:

- \(I\) has exactly \(N\) root boxes;
- in each root box, \(I'\) has fixed sign and
  \(\lvert I'\rvert\ge m>0\);
- the endpoint values have magnitude at least \(b>0\);
- off the root boxes, \(\lvert I\rvert\ge g>0\);
- uniformly, \(\lvert R_\epsilon\rvert\le M\) and
  \(\lvert\partial_hR_\epsilon\rvert\le K\).

Then the normalized displacement
\(\Delta_\epsilon/\epsilon=I+\epsilon R_\epsilon\) has exactly the same
\(N\) simple real zeros whenever

\[
0<|\epsilon|<
\min\left\{\frac bM,\frac gM,\frac mK\right\}.
\]

Endpoint signs give existence, the derivative margin gives uniqueness, and
the off-root margin excludes additional zeros.  The implementation seals this
threshold arithmetic over \(\mathbb Q\); it does not silently manufacture the
analytic hypotheses.

## Why the open DRR cases do not pass

The named Abelian certificate concerns a regular oval of

\[
H(x,y)=y^2+x^3+px+q
\]

strictly between critical energies.  The open `I_2^1` and `I_4^1` cases are
nilpotent saddle-node graphics at infinity.  For either case, the repository
does not currently supply:

1. an exact reduction of its physical return displacement to the named
   Hamiltonian Abelian integral;
2. a source-derived \(O(\epsilon^2)\) remainder with uniform value and
   derivative bounds;
3. capture of the singular endpoint and all incident graphic sides.

A Picard--Fuchs annihilator controls the supplied Abelian integral.  It cannot
replace these three physical-return obligations.

## Verdict

`omnibias.dynamics.abelian_return_transfer` records:

```text
picard_fuchs_verified = true
conditional_transfer_verified = true
named_graphic_reduction_supplied = false
physical_remainder_derived = false
endpoint_capture_proved = false
drr_transfer_certified = false
full_hilbert16_solved = false
```

H4 therefore succeeds as a mechanized infinitesimal and conditional-transfer
artifact, but fails as a route to an open DRR graphic or general Hilbert XVI.
This agrees with the known scope of the solved infinitesimal problem.

Reproduce the assessment with:

```bash
python benchmarks/hilbert16_abelian_return_transfer.py
```
