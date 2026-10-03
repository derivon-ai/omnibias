# Hilbert XVI H6: Songling four-cycle reproduction audit

H6 proposed producing a certificate-backed \(H(2)\ge4\) witness. The lower
bound is important, but it is not new: Shi proved four cycles in 1980, and
Galias--Tucker proved with rigorous interval arithmetic in 2022 that the
Songling system has exactly four limit cycles
([DOI 10.1016/j.amc.2021.126691](https://doi.org/10.1016/j.amc.2021.126691)).

## Exact source

The audited quadratic field is

\[
\begin{aligned}
\dot x&=\lambda x-y-10x^2+(5+\delta)xy+y^2,\\
\dot y&=x+x^2+(-25+8\epsilon-9\delta)xy,
\end{aligned}
\]

with

\[
\lambda=-10^{-200},\qquad \epsilon=-10^{-52},\qquad
\delta=-10^{-13}.
\]

`omnibias.dynamics.songling_lower_bound.songling_field` represents every
coefficient as an exact `Fraction`.

The four published section roots lie at scales approximately
\(10^{-2},10^{-8},10^{-21},10^{-75}\). The smallest endpoint displacement
sign used in the published existence proof is \(5.03\cdot10^{-295}\), and the
later uniqueness/hyperbolicity computation used up to 2048-bit arithmetic.

## Exact backend falsifier

The current omnibias interval substrate has binary64 endpoints. For the
`xy` coefficient in the second equation, exact arithmetic gives

\[
(-25+8\epsilon-9\delta)-(-25-9\delta)=8\epsilon=-8\cdot10^{-52}.
\]

After outward binary64 injection and subtraction, the current backend returns

```text
[-3.552713678800502e-15, 3.552713678800502e-15],
```

which contains zero. Thus the backend cannot preserve even the sign of a
governing perturbation under the dominant cancellation, before integrating a
single return orbit. Running four ordinary `certify_stopped_event` calls would
therefore be incapable of reproducing the published proof.

This does not show that the proof is impossible in omnibias. It gives the
concrete prerequisite: an outward arbitrary-precision interval ODE and
variational backend, followed by four disjoint first-return interval-Newton
certificates whose multipliers exclude one.

```text
published H(2) >= 4 = true
published exact cycle count = 4
published theorem replayed by omnibias = false
four hyperbolic returns certified = false
full_hilbert16_solved = false
```

Reproduce the audit with:

```bash
python -m pytest \
  packages/omnibias-dynamics/tests/test_songling_lower_bound.py -q
python benchmarks/hilbert16_songling_lower_bound.py
```
