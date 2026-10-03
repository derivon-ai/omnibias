# Hilbert XVI: complex fixed-time normal flow

The H3 bounded-format route needs control of the physical return family on a
complex neighborhood. Real interval first-hit certificates do not provide
that. `omnibias.core.verified.complex_ode` supplies the first missing substrate:
an outward-rounded Taylor integrator for

\[
Y'(t)=F(Y(t)),\qquad t\in\mathbb R,
\]

uniformly over a rectangular complex initial set.

## Certified local cell

`omnibias.dynamics.complex_normal_flow` applies the integrator to the cubic
Hilbert-XVI normal-form comparison field

\[
\dot V=-L\epsilon^3+\lambda_1\epsilon^2V-\epsilon V^2+
\frac{\epsilon}{3}V^3+h[-1+\epsilon(V-1)],\qquad
\dot h=-Vh,
\]

with \(L=9/25\), \(\lambda_1=-2\), \(V(0)=-\epsilon\), and
\(h(0)=4\epsilon^3\). It encloses the flow through \(t=1/2\) for every

\[
\epsilon\in
\left[\frac1{16}-10^{-4},\frac1{16}+10^{-4}\right]
+i[-10^{-4},10^{-4}].
\]

Dense deterministic and seeded random complex parameters are checked against
independent high-precision trajectories.

## Local complex event branch

`omnibias.dynamics.complex_event_branch` augments the complexified field with
a frozen complex time scale and applies parametric complex interval Newton to

\[
V(T,\epsilon)+\frac{629534}{10^7}=0.
\]

For

\[
\epsilon\in
\left[\frac1{16}-10^{-6},\frac1{16}+10^{-6}\right]
+i[-10^{-6},10^{-6}],
\]

it isolates one event-time branch inside

\[
\operatorname{Re}T\in[0.49886,0.50114],\qquad
\operatorname{Im}T\in[-0.00114,0.00114],
\]

with the complex event derivative excluding zero. A separate
`certify_stopped_event` computation derives first- and second-parameter
variations from the exact polynomial source and proves that the branch zero is
the first transverse target crossing on the real epsilon slice. Dense and
random complex parameters are independently checked with high-precision
integration.

## Separation-limit cover

`omnibias.dynamics.complex_separation_cover` imposes the exact relation

\[
L=1-\frac{\mathrm{sep}^2}{4},\qquad \lambda_1=-2.
\]

The single complex rectangle

\[
\operatorname{Re}(\mathrm{sep})\in[0,2],\qquad
\operatorname{Im}(\mathrm{sep})\in[-10^{-3},10^{-3}]
\]

contains both the D--C endpoint \(\mathrm{sep}=0\) and the chart-O endpoint
\(r_1=(2-\mathrm{sep})/2=0\). Complex interval Newton isolates the same regular
\(V\)-event branch throughout that rectangle, with its event derivative
separated from zero. An exact-source stopped-event replay independently proves
the first real crossing for every \(\mathrm{sep}\in[0,2]\).

This removes those two limits as obstructions for this chosen regular event of
the cubic comparison field. It does not repair the physical height/entry-exit
sections whose transversality and matching fail in H1.

## Physical outgoing section

`omnibias.dynamics.complex_physical_e_out` replaces the regular \(V\)-event by
the matching-chart image of the physical outgoing section:

\[
E_{\rm out}(V,h)
=V+\rho+\nu\rho h+C\nu^2\rho h^2=0,
\qquad
(\nu,\rho,C)=\left(\frac1{16},\frac14,2\right).
\]

Eight closed rational cells of width \(1/4\) cover
\(\operatorname{Re}(\mathrm{sep})\in[0,2]\), each with the common complex
thickness \(|\operatorname{Im}(\mathrm{sep})|\le10^{-4}\). The flow is first
enclosed to the fixed real anchor \(T=30\); complex interval Newton then
isolates the remaining event time on every cell. Adjacent cells share a
parameter boundary and the same time domain. Their root enclosures overlap, so
uniqueness matches the branches across the complete cover.

An independent exact-source stopped-event run proves first transverse
crossing on the same eight real cells. Thus the physical outgoing branch now
continues from the D--C endpoint through the chart-O endpoint for this cubic
comparison source.

This is not complete physical overlap matching. The incoming
\(E_\sigma\) branch, the entry-to-exit composition, the nontruncated quadratic
field, and bounded Log-Noetherian format remain open.

## What remains open

These are fixed-time flow enclosures and local event branches of the cubic
comparison field. They are not:

- the full physical quadratic singular passage;
- a complex incoming \(E_\sigma\) branch matched to \(E_{\rm out}\);
- a complete return map between every atlas section;
- an exact differential-polynomial chain for that map;
- a parameter-independent Log-Noetherian format; or
- G3 or Hilbert XVI.

The next analytic obligation is to replace the local cubic comparison source
with the full physical quadratic passage and continue compatible branches
across the D-C and chart-O overlaps. Only a finite uniform cover with bounded
complex flow/event enclosures can be offered to the Log-Noetherian format
machinery; one local event branch is not exact differential-polynomial
membership.

Reproduce the local result with:

```bash
python -m pytest \
  packages/omnibias-core/tests/verified/test_complex_ode.py \
  packages/omnibias-core/tests/verified/test_complex_rootfind.py \
  packages/omnibias-dynamics/tests/test_complex_event_branch.py \
  packages/omnibias-dynamics/tests/test_complex_normal_flow.py \
  packages/omnibias-dynamics/tests/test_complex_physical_e_out.py \
  packages/omnibias-dynamics/tests/test_complex_separation_cover.py -q
```
