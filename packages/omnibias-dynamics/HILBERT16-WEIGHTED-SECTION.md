# H1 weighted-section assessment

This note tests the proposed separation-scale closing section

\[
h=\epsilon^3\operatorname{sep}^2\eta_0.
\]

It is a negative assessment of that route, not G1, G4, graphic cyclicity,
or Hilbert's sixteenth problem.

## What works inside chart D

With `sigma=sep`, `eta=y/sep^2`, and
`x=rstar+sep*xi`, the exact chart equation is

\[
\eta_\tau=x\eta.
\]

On a compact box with `x >= m > 0`, the section `eta=eta0` is therefore
hit uniquely and transversely, conditional on the orbit remaining in the
box.  This is a valid value-level first hit for every fixed `sep>0`.

## Why the weighted margin does not match D to C

Put `q=sep^2`.  The scalar normal equation

\[
\eta_\tau=r\eta,\qquad \eta(0)=q\eta_i
\]

has hit time

\[
\tau(q)=\frac1r\log\frac{\eta_0}{q\eta_i}.
\]

Hence

\[
r q\tau_q=-1,\qquad r q^2\tau_{qq}=1.
\]

The weighted derivatives are finite, but the ordinary coefficient
derivatives grow as `q^-1` and `q^-2`.  At `q=0`, the physical section is
`h=0`; a positive-height orbit does not hit it in finite time.  Thus the
intrinsic section does not provide the ordinary C2 coefficient matching
required on `D intersect C`.  It also does not replace the still-open
physical C2 remainder of `log D'`.

## Why the margin also fails at chart O

Use the exact rational shrinking-root path

\[
r_1=q,\quad r_2=2-q,\quad
\operatorname{sep}=2-2q,\quad L=q(2-q).
\]

At the matching interface `x=r1(1+theta)`,

\[
\eta_\tau=q(1+\theta)\eta_0\longrightarrow0.
\]

The chart-intrinsic margin therefore collapses at the `O` overlap itself.
Moving first to a fixed positive `x` would require the outgoing physical
first-hit and overlap theorem that G1 already lacks.

## Verdict

`omnibias.dynamics.weighted_section` seals three exact rational identities:
the weighted first and second hit-time derivatives and the chart-O speed
factorization.  The result falsifies H1 as a G1 discharge.  It does **not**
prove that every possible new closing map fails.

The parent flags remain derived and false:

```text
weighted_section_obstruction = true
weighted_section_closes_g1 = false
g1_passed = false
full_hilbert16_solved = false
```

Reproduce the finite assessment with:

```bash
python benchmarks/hilbert16_weighted_section.py
```
