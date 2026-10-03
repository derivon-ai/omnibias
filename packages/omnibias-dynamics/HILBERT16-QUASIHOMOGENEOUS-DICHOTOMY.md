# H2 quasi-homogeneous scale dichotomy

This note tests whether the frozen-section obstruction can be upgraded from
several failed scales to a theorem over all quasi-homogeneous blow-up weights.
The maximal true result is a one-scale no-go theorem.  The proposed theorem
over every weighted atlas is false.

## Kill sequence and exact logarithms

Take

\[
\epsilon=n^{-1},\qquad \operatorname{sep}=e^{-n^2},\qquad
\sigma=\epsilon^a\operatorname{sep}^b.
\]

For the event factor and a frozen section
\(h_{\max}=\epsilon^N\), the exact logarithmic expressions are

\[
\log(\sigma\kappa)=(1-b)n^2-a\log n,
\]

\[
\log\frac{W_{\max}}{W_e}
=2bn+(2a+3-N)\frac{\log n}{n}.
\]

The event factor is bounded exactly when \(b>1\), or \(b=1\) and
\(a\geq0\).  The frozen-section ratio is bounded above only when
\(b\leq0\).  The conditions are incompatible.  This proves the no-go result
for every rational monomial weight, not merely the six weights previously
tested.

The same leading-order argument is not tied to a monomial parameterization:
if a single positive scale makes \(\sigma\kappa\) bounded on the kill
sequence, then \(\sigma\) is at most a constant times \(e^{-n^2}\).
Its separately bounded frozen-section factor contains
\(\sigma^{-2/n}\), which then grows at least exponentially in \(n\).

## Why this is not an atlas impossibility theorem

For a moving section

\[
h_{\max}=\epsilon^3\operatorname{sep}^2
\]

and \((a,b)=(0,1)\), both scalar logarithms are exactly zero.  Thus the
moving section is a counterexample to the proposed theorem over all
quasi-homogeneous atlases.  The separate
[weighted-section assessment](HILBERT16-WEIGHTED-SECTION.md) shows why this
scalar escape still fails ordinary C2 matching at `D intersect C` and loses
the chart-O interface margin.

## Verdict

`omnibias.dynamics.quasihomogeneous_dichotomy` seals three exact rational
logarithmic identities and the complete boundary classification:

```text
frozen_section_scale_no_go = true
all_quasihomogeneous_atlases_excluded = false
g1_passed = false
full_hilbert16_solved = false
```

H2 therefore yields a publishable local no-go statement for a frozen section,
but its broader falsifier succeeds.  It does not prove physical C2, G1,
graphic cyclicity, or Hilbert's sixteenth problem.

Reproduce the finite assessment with:

```bash
python benchmarks/hilbert16_quasihomogeneous_dichotomy.py
```
